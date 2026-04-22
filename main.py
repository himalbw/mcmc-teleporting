import csv
import os
import numpy as np

from scripts.generate_data import make_scenarios, make_scaling_scenarios
from samplers.parallel_tempering import (
    ParallelTemperingMCMC,
    grid_search_temperatures,
    optimize_temperatures,
)
from samplers.teleporting_mcmc import TeleportingMCMC, gaussian_q_density, gaussian_q_sample
from samplers.hybrid_teleporting_nuts import HybridTeleportingNUTS
from samplers.vanilla_mcmc import VanillaMCMC
from diagnostics import summary, plot_comparison, save_metrics_table, ess, r_hat

# ------------------------------------------------------------------
# Output layout:
#
#   results/
#     comparison/
#       {slug}.png            ← 1×3 (or 2×3) per-scenario comparison
#       metrics_table.png     ← colour-coded TVD / ESS / R-hat summary
#       metrics_table.csv
#     tvd_summary.csv         ← backward-compat TVD-only CSV
# ------------------------------------------------------------------

SLUGS = [
    "standard", "correlated", "bimodal_moderate",
    "bimodal_large", "unequal_weight", "different_scale",
]

NUM_ITER = 5000
WARMUP_FRACTION = 0.25

# Dimension-scaling experiment
SCALING_DIMS = [1, 2, 5, 10, 20]
SCALING_ITER = 3000


def _setup_dirs():
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("results/hybrid", exist_ok=True)


def _fig_path(slug):
    return f"results/comparison/{slug}.png"


def _save_hybrid_fig(chains, scenario):
    """Save per-scenario density figure(s) for the hybrid sampler."""
    import matplotlib.pyplot as plt
    from diagnostics import plot_against_truth

    d    = scenario["d"]
    slug = scenario["slug"]

    fig, axes = plt.subplots(1, d, figsize=(6 * d, 4), squeeze=False)

    for dim in range(d):
        ax = axes[0, dim]
        if d == 1:
            pi_fn_1d  = scenario["pi_fn"]
            x_range   = scenario["x_range"][0]
            chains_1d = chains
            dim_label = "x"
        else:
            pi_fn_1d  = scenario["marginal_pi_fns"][dim]
            x_range   = scenario["x_range"][dim]
            chains_1d = chains[:, :, dim:dim + 1]
            dim_label = f"x[{dim}]"

        tvd = plot_against_truth(chains_1d, pi_fn_1d,
                                 x_range=x_range, param_name=dim_label, ax=ax)
        ax.set_title(f"Hybrid (T+NUTS)  —  TVD = {tvd:.4f}", fontsize=10)

    fig.suptitle(scenario["label"], fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = f"results/hybrid/{slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


# ------------------------------------------------------------------
# Per-scenario runner
# ------------------------------------------------------------------

def run_scenario(scenario, rng, num_iter=NUM_ITER):
    slug           = scenario["slug"]
    pi_fn          = scenario["pi_fn"]
    d              = scenario["d"]
    x_range        = scenario["x_range"]
    proposal_sigma = scenario["proposal_sigma"]
    warmup         = int(num_iter * WARMUP_FRACTION)
    N_walkers      = 8

    param_names = ["x"] if d == 1 else [f"x[{i}]" for i in range(d)]

    # --- initial positions ---
    if d == 1:
        mu     = scenario["mu"]
        sigma2 = scenario["sigma2"]
        mode_idx = rng.integers(len(mu), size=N_walkers)
        x0 = rng.normal(
            loc=mu[mode_idx], scale=np.sqrt(sigma2.mean())
        ).reshape(N_walkers, 1)
    else:
        # unimodal (correlated Gaussian): start walkers near origin with small spread
        x0 = rng.normal(loc=0.0, scale=1.0, size=(N_walkers, d))

    # Isotropic Gaussian proposal (works for all d)
    q_sample_fn  = lambda x, r: gaussian_q_sample(x, proposal_sigma, r)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, proposal_sigma)

    row = {"scenario": scenario["label"]}

    # ----------------------------------------------------------------
    # 1. Teleporting MCMC
    # ----------------------------------------------------------------
    print("  [Teleporting MCMC]")
    t_sampler = TeleportingMCMC(pi_fn, q_sample_fn, q_density_fn, rng=rng)
    t_result  = t_sampler.run(x0, num_iter)
    t_chains  = t_result["samples"].transpose(1, 0, 2)[:, warmup:, :]

    print(
        f"    accept={t_result['acceptance_rate']:.3f}  "
        f"teleport_proposal={t_result['teleport_proposal_rate']:.3f}  "
        f"teleport_accept={t_result['teleport_accept_rate']:.3f}"
    )
    summary(t_chains, param_names=param_names)
    row["ess_teleporting"]  = round(float(ess(t_chains).mean()),  1)
    row["rhat_teleporting"] = round(float(r_hat(t_chains).mean()), 3)

    # ----------------------------------------------------------------
    # 2. Hybrid Teleporting-NUTS
    # ----------------------------------------------------------------
    print("\n  [Hybrid Teleporting-NUTS]")
    init_step = proposal_sigma / 5.0   # reasonable starting point; adapted during warmup
    h_sampler = HybridTeleportingNUTS(
        pi_fn          = pi_fn,
        q_sample_fn    = q_sample_fn,
        q_density_fn   = q_density_fn,
        init_step_size = init_step,
        max_tree_depth = 5,
        target_accept  = 0.65,
        rng            = rng,
    )
    h_result = h_sampler.run(x0, num_iter, num_warmup=warmup)
    h_chains  = h_result["samples"].transpose(1, 0, 2)[:, warmup:, :]

    print(
        f"    accept={h_result['acceptance_rate']:.3f}  "
        f"teleport_proposal={h_result['teleport_proposal_rate']:.3f}  "
        f"teleport_accept={h_result['teleport_accept_rate']:.3f}  "
        f"nuts_local={h_result['local_nuts_rate']:.3f}  "
        f"calibrated_ε={h_result['calibrated_step_size']:.4f}"
    )
    summary(h_chains, param_names=param_names)
    row["ess_hybrid"]  = round(float(ess(h_chains).mean()),  1)
    row["rhat_hybrid"] = round(float(r_hat(h_chains).mean()), 3)
    _save_hybrid_fig(h_chains, scenario)

    # ----------------------------------------------------------------
    # 4. Parallel Tempering — grid search then adaptive refinement
    # ----------------------------------------------------------------
    print("\n  [Parallel Tempering — grid search]")
    best_betas, _ = grid_search_temperatures(
        pi_fn          = pi_fn,
        x0_single      = x0[0],
        num_replicas_grid = [3, 4, 5, 6, 7],
        beta_min_grid  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
        num_iter       = 400,
        proposal_scale = proposal_sigma,
        rng            = rng,
        verbose        = True,
    )

    print("\n  [Parallel Tempering — adaptive temperature refinement]")
    best_betas, last_swap_rates = optimize_temperatures(
        pi_fn              = pi_fn,
        x0_single          = x0[0],
        betas_init         = best_betas,
        num_rounds         = 4,
        num_iter_per_round = 400,
        target_rate        = 0.25,
        proposal_scale     = proposal_sigma,
        rng                = rng,
        verbose            = True,
    )
    print(f"    refined betas = {best_betas.round(4)}")
    print(f"    swap rates    = {last_swap_rates.round(3)}")

    print("\n  [Parallel Tempering — full run]")
    pt_x0      = np.tile(x0[0], (len(best_betas), 1))
    pt_sampler = ParallelTemperingMCMC(
        pi_fn, best_betas, proposal_scale=proposal_sigma, rng=rng
    )
    pt_result  = pt_sampler.run(pt_x0, num_iter=num_iter)
    cold_chain = pt_result["cold_samples"].transpose(1, 0, 2)[:, warmup:, :]

    swap_str = ", ".join(f"{r:.3f}" for r in pt_result["swap_acceptance_rates"])
    print(f"    betas={best_betas.round(3)}  swap_rates=[{swap_str}]")
    summary(cold_chain, param_names=param_names)
    row["ess_parallel_tempering"]  = round(float(ess(cold_chain).mean()),  1)
    row["rhat_parallel_tempering"] = round(float(r_hat(cold_chain).mean()), 3)

    # ----------------------------------------------------------------
    # 5. Vanilla MCMC (PyMC / NUTS)
    # ----------------------------------------------------------------
    print("\n  [Vanilla MCMC — PyMC NUTS]")
    vanilla_kwargs = {"vanilla_type": scenario["vanilla_type"], "seed": 221}
    if scenario["vanilla_type"] == "mixture_1d":
        vanilla_kwargs.update(
            pi=scenario["pi"], mu=scenario["mu"], sigma2=scenario["sigma2"]
        )
    else:  # mvnormal
        vanilla_kwargs.update(mu_vec=scenario["mu_vec"], cov=scenario["cov"])

    vanilla  = VanillaMCMC(**vanilla_kwargs)
    v_result = vanilla.run(
        num_draws=num_iter, num_chains=4, num_tune=warmup, progressbar=False
    )
    summary(v_result["samples"], param_names=param_names)
    row["ess_vanilla"]  = round(float(ess(v_result["samples"]).mean()),  1)
    row["rhat_vanilla"] = round(float(r_hat(v_result["samples"]).mean()), 3)

    # ----------------------------------------------------------------
    # Comparison figure (grouped by scenario)
    # ----------------------------------------------------------------
    method_chains = {
        "teleporting":        t_chains,
        "hybrid":             h_chains,
        "parallel_tempering": cold_chain,
        "vanilla":            v_result["samples"],
    }
    tvds = plot_comparison(
        method_chains=method_chains,
        scenario=scenario,
        save_path=_fig_path(slug),
    )

    for method in ["teleporting", "hybrid", "parallel_tempering", "vanilla"]:
        avg_tvd = float(np.mean(tvds[method]))
        row[f"tvd_{method}"] = round(avg_tvd, 4)

    return row


# ------------------------------------------------------------------
# Dimension-scaling helpers
# ------------------------------------------------------------------

def _compute_tvd_1d(chains, pi_fn_1d, x_range, param_idx=0, n_grid=400):
    """Estimate TVD between sampler KDE and the 1D marginal truth."""
    from scipy.stats import gaussian_kde
    samples_1d = chains[:, :, param_idx].ravel()
    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx   = grid[1] - grid[0]
    kde  = gaussian_kde(samples_1d, bw_method="scott")
    p_kde      = kde(grid)
    p_true_raw = np.array([pi_fn_1d(np.array([v])) for v in grid])
    p_true     = p_true_raw / (p_true_raw.sum() * dx)
    return float(0.5 * np.sum(np.abs(p_true - p_kde)) * dx)


def run_scaling_scenario(scenario, rng, num_iter=SCALING_ITER):
    """Run all four methods on a single d-dim product bimodal scenario."""
    import time
    pi_fn          = scenario["pi_fn"]
    d              = scenario["d"]
    x_range        = scenario["x_range"]
    proposal_sigma = scenario["proposal_sigma"]
    warmup         = int(num_iter * WARMUP_FRACTION)
    N_walkers      = 8
    marginal_pi_fn = scenario["marginal_pi_fn"]

    # Initialise half walkers near each mode
    x0 = np.zeros((N_walkers, d))
    for i in range(N_walkers):
        center = -5.0 if i < N_walkers // 2 else 5.0
        x0[i]  = center + rng.normal(scale=1.0, size=d)

    q_sample_fn  = lambda x, r: gaussian_q_sample(x, proposal_sigma, r)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, proposal_sigma)

    row = {"d": d}

    # ---- 1. Teleporting MCMC ----
    t0 = time.perf_counter()
    t_sampler = TeleportingMCMC(pi_fn, q_sample_fn, q_density_fn, rng=rng)
    t_result  = t_sampler.run(x0, num_iter)
    t_chains  = t_result["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["time_teleporting"] = round(time.perf_counter() - t0, 2)
    row["tvd_teleporting"]  = round(_compute_tvd_1d(t_chains, marginal_pi_fn, x_range[0]), 4)
    row["ess_teleporting"]  = round(float(ess(t_chains).mean()), 1)
    row["rhat_teleporting"] = round(float(r_hat(t_chains).mean()), 3)
    print(
        f"    [Teleport] TVD={row['tvd_teleporting']:.4f}  "
        f"ESS={row['ess_teleporting']:.0f}  time={row['time_teleporting']:.1f}s"
    )

    # ---- 2. Hybrid Teleporting-NUTS ----
    t0 = time.perf_counter()
    h_sampler = HybridTeleportingNUTS(
        pi_fn=pi_fn, q_sample_fn=q_sample_fn, q_density_fn=q_density_fn,
        init_step_size=proposal_sigma / 5.0, max_tree_depth=5,
        target_accept=0.65, rng=rng,
    )
    h_result = h_sampler.run(x0, num_iter, num_warmup=warmup)
    h_chains = h_result["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["time_hybrid"] = round(time.perf_counter() - t0, 2)
    row["tvd_hybrid"]  = round(_compute_tvd_1d(h_chains, marginal_pi_fn, x_range[0]), 4)
    row["ess_hybrid"]  = round(float(ess(h_chains).mean()), 1)
    row["rhat_hybrid"] = round(float(r_hat(h_chains).mean()), 3)
    print(
        f"    [Hybrid]   TVD={row['tvd_hybrid']:.4f}  "
        f"ESS={row['ess_hybrid']:.0f}  time={row['time_hybrid']:.1f}s"
    )

    # ---- 3. Parallel Tempering (fixed betas — no grid search) ----
    betas = np.array([1.0, 0.2, 0.04, 0.008])
    t0    = time.perf_counter()
    pt_x0      = np.tile(x0[0], (len(betas), 1))
    pt_sampler = ParallelTemperingMCMC(pi_fn, betas, proposal_scale=proposal_sigma, rng=rng)
    pt_result  = pt_sampler.run(pt_x0, num_iter=num_iter)
    cold_chain = pt_result["cold_samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["time_parallel_tempering"] = round(time.perf_counter() - t0, 2)
    row["tvd_parallel_tempering"]  = round(_compute_tvd_1d(cold_chain, marginal_pi_fn, x_range[0]), 4)
    row["ess_parallel_tempering"]  = round(float(ess(cold_chain).mean()), 1)
    row["rhat_parallel_tempering"] = round(float(r_hat(cold_chain).mean()), 3)
    print(
        f"    [PT]       TVD={row['tvd_parallel_tempering']:.4f}  "
        f"ESS={row['ess_parallel_tempering']:.0f}  time={row['time_parallel_tempering']:.1f}s"
    )

    # ---- 4. Vanilla PyMC NUTS ----
    t0 = time.perf_counter()
    vanilla_kwargs = {"vanilla_type": scenario["vanilla_type"], "seed": 221}
    if scenario["vanilla_type"] == "mixture_1d":
        vanilla_kwargs.update(pi=scenario["pi"], mu=scenario["mu"], sigma2=scenario["sigma2"])
    else:
        vanilla_kwargs["nd_d"] = d
    vanilla  = VanillaMCMC(**vanilla_kwargs)
    v_result = vanilla.run(
        num_draws=num_iter, num_chains=4, num_tune=warmup, progressbar=False
    )
    row["time_vanilla"] = round(time.perf_counter() - t0, 2)
    row["tvd_vanilla"]  = round(_compute_tvd_1d(v_result["samples"], marginal_pi_fn, x_range[0]), 4)
    row["ess_vanilla"]  = round(float(ess(v_result["samples"]).mean()), 1)
    row["rhat_vanilla"] = round(float(r_hat(v_result["samples"]).mean()), 3)
    print(
        f"    [Vanilla]  TVD={row['tvd_vanilla']:.4f}  "
        f"ESS={row['ess_vanilla']:.0f}  time={row['time_vanilla']:.1f}s"
    )

    return row


def run_scaling_experiment(rng):
    """Run the dimension-scaling experiment and save results/scaling/scaling_results.csv."""
    os.makedirs("results/scaling", exist_ok=True)
    scenarios = make_scaling_scenarios(SCALING_DIMS)
    rows = []

    print(f"\n{'='*68}")
    print("  DIMENSION SCALING EXPERIMENT")
    print(f"  ({SCALING_ITER} iterations per method)")
    print(f"{'='*68}")

    for scenario in scenarios:
        print(f"\n  d = {scenario['d']}")
        row = run_scaling_scenario(scenario, rng)
        rows.append(row)

    csv_path  = "results/scaling/scaling_results.csv"
    fieldnames = ["d"] + [
        f"{metric}_{method}"
        for metric in ["tvd", "ess", "rhat", "time"]
        for method in ["teleporting", "hybrid", "parallel_tempering", "vanilla"]
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved {csv_path}")

    # Print summary
    print(f"\n  {'d':>4}  {'Teleport':>10}  {'Hybrid':>8}  {'PT':>8}  {'Vanilla':>8}  (TVD)")
    print(f"  {'-'*50}")
    for r in rows:
        print(
            f"  {r['d']:>4}  {r['tvd_teleporting']:>10.4f}"
            f"  {r['tvd_hybrid']:>8.4f}"
            f"  {r['tvd_parallel_tempering']:>8.4f}"
            f"  {r['tvd_vanilla']:>8.4f}"
        )

    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    _setup_dirs()

    rng       = np.random.default_rng(221)
    num_iter  = NUM_ITER
    warmup    = int(num_iter * WARMUP_FRACTION)
    scenarios = make_scenarios(rng)

    print(f"Running {num_iter} iterations with {warmup} warmup iterations.")

    all_rows = []
    for scenario in scenarios:
        print(f"\n{'='*68}")
        print(f"  {scenario['label']}")
        print(f"{'='*68}")
        row = run_scenario(scenario, rng, num_iter)
        all_rows.append(row)
        print(
            f"\n  TVD — Teleporting: {row['tvd_teleporting']:.4f}  |  "
            f"Hybrid: {row['tvd_hybrid']:.4f}  |  "
            f"PT: {row['tvd_parallel_tempering']:.4f}  |  "
            f"Vanilla: {row['tvd_vanilla']:.4f}"
        )

    # ----------------------------------------------------------------
    # Final comparison table + CSV
    # ----------------------------------------------------------------
    col = 42
    header = (f"{'Scenario':<{col}}  {'Teleporting':>12}  {'Hybrid':>8}"
              f"  {'PT':>8}  {'Vanilla':>8}")
    sep    = "-" * len(header)

    print(f"\n{'='*68}")
    print("  FINAL TVD COMPARISON")
    print(f"{'='*68}")
    print(header)
    print(sep)
    for r in all_rows:
        print(
            f"{r['scenario']:<{col}}  {r['tvd_teleporting']:>12.4f}"
            f"  {r['tvd_hybrid']:>8.4f}"
            f"  {r['tvd_parallel_tempering']:>8.4f}  {r['tvd_vanilla']:>8.4f}"
        )

    csv_path = "results/tvd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "tvd_teleporting", "tvd_hybrid",
                "tvd_parallel_tempering", "tvd_vanilla",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved {csv_path}")

    # ----------------------------------------------------------------
    # Metrics comparison table
    # ----------------------------------------------------------------
    print("\n  Generating metrics comparison table...")
    save_metrics_table(all_rows, save_dir="results/comparison")
    print("\n  Outputs saved to results/comparison/")

    # ----------------------------------------------------------------
    # Dimension-scaling experiment
    # ----------------------------------------------------------------
    run_scaling_experiment(rng)


if __name__ == "__main__":
    main()
