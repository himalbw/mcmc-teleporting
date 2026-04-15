import numpy as np

from scripts.generate_data import make_scenarios
from samplers.parallel_tempering import ParallelTemperingMCMC, grid_search_temperatures
from samplers.teleporting_mcmc import TeleportingMCMC, gaussian_q_density, gaussian_q_sample
from samplers.vanilla_mcmc import VanillaMCMC
from diagnostics import summary, plot_against_truth


# ------------------------------------------------------------------
# Per-scenario runner
# ------------------------------------------------------------------

def run_scenario(scenario, rng, num_iter=2000):
    slug     = scenario["slug"]
    pi_fn    = scenario["pi_fn"]
    mu       = scenario["mu"]
    sigma2   = scenario["sigma2"]
    x_range  = scenario["x_range"]
    warmup   = num_iter // 4
    N_walkers = 8

    # Proposal sigma scaled to component spread
    proposal_sigma = float(np.sqrt(sigma2.max()) * 1.5)

    q_sample_fn  = lambda x, rng: gaussian_q_sample(x, proposal_sigma, rng)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, proposal_sigma)

    # Shared x0: walkers spread across modes
    mode_idx = rng.integers(len(mu), size=N_walkers)
    x0 = rng.normal(loc=mu[mode_idx], scale=np.sqrt(sigma2.mean())).reshape(N_walkers, 1)

    # ---- 1. Teleporting MCMC ------------------------------------
    print("  [Teleporting MCMC]")
    t_sampler = TeleportingMCMC(pi_fn, q_sample_fn, q_density_fn, rng=rng)
    t_result  = t_sampler.run(x0, num_iter)
    t_chains  = t_result["samples"].transpose(1, 0, 2)[:, warmup:, :]

    print(
        f"    acceptance={t_result['acceptance_rate']:.3f}  "
        f"teleport_proposal={t_result['teleport_proposal_rate']:.3f}  "
        f"teleport_accept={t_result['teleport_accept_rate']:.3f}"
    )
    summary(t_chains, param_names=["x"])
    tvd_t = plot_against_truth(
        t_chains, pi_fn, param_name="x", x_range=x_range,
        save_path=f"results/{slug}_teleporting.png",
    )
    print(f"    TVD = {tvd_t:.4f}")

    # ---- 2. Parallel Tempering (grid search then full run) -------
    print("\n  [Parallel Tempering — grid search]")
    best_betas, gs_results = grid_search_temperatures(
        pi_fn=pi_fn,
        x0_single=x0[0],
        num_replicas_grid=[3, 4, 5],
        beta_min_grid=[0.01, 0.05, 0.1, 0.2, 0.5],
        num_iter=400,
        proposal_scale=proposal_sigma,
        rng=rng,
        verbose=True,
    )

    print(f"\n  [Parallel Tempering — full run, best schedule]")
    pt_x0    = np.tile(x0[0], (len(best_betas), 1))
    pt_sampler = ParallelTemperingMCMC(pi_fn, best_betas, proposal_scale=proposal_sigma, rng=rng)
    pt_result  = pt_sampler.run(pt_x0, num_iter=num_iter)
    cold_chain = pt_result["cold_samples"].transpose(1, 0, 2)[:, warmup:, :]

    swap_str = ", ".join(f"{r:.3f}" for r in pt_result["swap_acceptance_rates"])
    print(f"    swap_rates=[{swap_str}]")
    summary(cold_chain, param_names=["x"])
    tvd_pt = plot_against_truth(
        cold_chain, pi_fn, param_name="x", x_range=x_range,
        save_path=f"results/{slug}_parallel_tempering.png",
    )
    print(f"    TVD = {tvd_pt:.4f}")

    # ---- 3. Vanilla MCMC (PyMC / NUTS) --------------------------
    print("\n  [Vanilla MCMC — PyMC NUTS]")
    vanilla   = VanillaMCMC(pi=scenario["pi"], mu=mu, sigma2=sigma2, seed=221)
    v_result  = vanilla.run(num_draws=num_iter, num_chains=4, num_tune=warmup, progressbar=False)
    summary(v_result["samples"], param_names=["x"])
    tvd_v = plot_against_truth(
        v_result["samples"], pi_fn, param_name="x", x_range=x_range,
        save_path=f"results/{slug}_vanilla.png",
    )
    print(f"    TVD = {tvd_v:.4f}")

    return {"teleporting": tvd_t, "parallel_tempering": tvd_pt, "vanilla": tvd_v}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    rng      = np.random.default_rng(221)
    num_iter = 2000
    scenarios = make_scenarios(rng)

    all_tvd = {}
    for scenario in scenarios:
        print(f"\n{'='*64}")
        print(f"SCENARIO: {scenario['label']}")
        print(
            f"  mu={scenario['mu']}  "
            f"sigma2={scenario['sigma2'].round(2)}  "
            f"pi={scenario['pi'].round(2)}"
        )
        print(f"{'='*64}")
        all_tvd[scenario["label"]] = run_scenario(scenario, rng, num_iter)

    # Final comparison table
    print(f"\n{'='*64}")
    print("FINAL TVD COMPARISON")
    print(f"{'='*64}")
    print(f"{'Scenario':<30}  {'Teleporting':>12}  {'PT':>8}  {'Vanilla':>8}")
    print("-" * 64)
    for label, tvds in all_tvd.items():
        print(
            f"{label:<30}  {tvds['teleporting']:>12.4f}  "
            f"{tvds['parallel_tempering']:>8.4f}  {tvds['vanilla']:>8.4f}"
        )


if __name__ == "__main__":
    main()
