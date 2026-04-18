import csv
import os
import numpy as np

from scripts.generate_data import make_scenarios
from samplers.parallel_tempering import (
    ParallelTemperingMCMC,
    grid_search_temperatures,
    optimize_temperatures,
)
from samplers.teleporting_mcmc import TeleportingMCMC, gaussian_q_density, gaussian_q_sample
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


def _setup_dirs():
    os.makedirs("results/comparison", exist_ok=True)


def _fig_path(slug):
    return f"results/comparison/{slug}.png"


# ------------------------------------------------------------------
# Per-scenario runner
# ------------------------------------------------------------------

def run_scenario(scenario, rng, num_iter=2000):
    slug           = scenario["slug"]
    pi_fn          = scenario["pi_fn"]
    d              = scenario["d"]
    x_range        = scenario["x_range"]
    proposal_sigma = scenario["proposal_sigma"]
    warmup         = num_iter // 4
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
    # 2. Parallel Tempering — grid search then adaptive refinement
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
    # 3. Vanilla MCMC (PyMC / NUTS)
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
        "parallel_tempering": cold_chain,
        "vanilla":            v_result["samples"],
    }
    tvds = plot_comparison(
        method_chains=method_chains,
        scenario=scenario,
        save_path=_fig_path(slug),
    )

    for method in ["teleporting", "parallel_tempering", "vanilla"]:
        avg_tvd = float(np.mean(tvds[method]))
        row[f"tvd_{method}"] = round(avg_tvd, 4)

    return row


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    _setup_dirs()

    rng       = np.random.default_rng(221)
    num_iter  = 2000
    scenarios = make_scenarios(rng)

    all_rows = []
    for scenario in scenarios:
        print(f"\n{'='*68}")
        print(f"  {scenario['label']}")
        print(f"{'='*68}")
        row = run_scenario(scenario, rng, num_iter)
        all_rows.append(row)
        print(
            f"\n  TVD — Teleporting: {row['tvd_teleporting']:.4f}  |  "
            f"PT: {row['tvd_parallel_tempering']:.4f}  |  "
            f"Vanilla: {row['tvd_vanilla']:.4f}"
        )

    # ----------------------------------------------------------------
    # Final comparison table + CSV
    # ----------------------------------------------------------------
    col = 42
    header = f"{'Scenario':<{col}}  {'Teleporting':>12}  {'PT':>8}  {'Vanilla':>8}"
    sep    = "-" * len(header)

    print(f"\n{'='*68}")
    print("  FINAL TVD COMPARISON")
    print(f"{'='*68}")
    print(header)
    print(sep)
    for r in all_rows:
        print(
            f"{r['scenario']:<{col}}  {r['tvd_teleporting']:>12.4f}  "
            f"{r['tvd_parallel_tempering']:>8.4f}  {r['tvd_vanilla']:>8.4f}"
        )

    csv_path = "results/tvd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "tvd_teleporting",
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


if __name__ == "__main__":
    main()
