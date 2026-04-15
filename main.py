import numpy as np
from functools import partial

from scripts.generate_data import generate_hierarchical_gaussian_mixture
from samplers.teleporting_mcmc import (
    TeleportingMCMC,
    gaussian_q_density,
    gaussian_q_sample,
)
from diagnostics import summary, plot_against_truth


def gmm_density(x, pi, mu, sigma2):
    """Unnormalised GMM density at scalar x."""
    x = np.asarray(x).ravel()
    assert x.size == 1, "This demo target is 1-D"
    val = 0.0
    for pk, mk, sk2 in zip(pi, mu, sigma2):
        val += pk * np.exp(-0.5 * (x[0] - mk) ** 2 / sk2) / np.sqrt(
            2 * np.pi * sk2
        )
    return float(val)

def main():
    rng = np.random.default_rng(221)

    # 1. Generate Data
    print("Generating hierarchical GMM data...")
    data = generate_hierarchical_gaussian_mixture(
        n=500,
        K=3,
        alpha=[1.0, 1.0, 1.0],
        a=3.0,
        b=2.0,
        c=3.0,
        d=4.0,
        m0=0.0,
        s0_sq=25.0,
        rng=rng,
    )
    print(f"  True mu:     {data['mu'].round(3)}")
    print(f"  True pi:     {data['pi'].round(3)}")
    print(f"  True sigma2: {data['sigma2'].round(3)}")
    print()

    # 2. Define target using true params
    pi_fn = partial(
        gmm_density, pi=data["pi"], mu=data["mu"], sigma2=data["sigma2"]
    )

    #3. Gaussian proposal with sigma = 1.0
    proposal_sigma = 1.0
    q_sample_fn  = lambda x, rng: gaussian_q_sample(x, proposal_sigma, rng)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, proposal_sigma)

    # 4. Teleporting MCMC
    print("Running Teleporting MCMC...")
    N_walkers = 8
    num_iter  = 2000

    # Initialise walkers: assign each walker to a random mode
    mode_idx = rng.integers(len(data["mu"]), size=N_walkers)
    x0 = rng.normal(loc=data["mu"][mode_idx], scale=1.0).reshape(N_walkers, 1)

    sampler = TeleportingMCMC(
        pi_fn=pi_fn,
        q_sample_fn=q_sample_fn,
        q_density_fn=q_density_fn,
        rng=rng,
    )
    result = sampler.run(x0, num_iter)

    print(f"  Acceptance rate:        {result['acceptance_rate']:.3f}")
    print(f"  Teleport proposal rate: {result['teleport_proposal_rate']:.3f}")
    print(f"  Teleport accept rate:   {result['teleport_accept_rate']:.3f}")
    print()

    # Diagnostics
    # samples shape: (num_iter+1, N_walkers, 1)
    # Treat walkers as chains → (N_walkers, num_iter+1, 1)
    samples = result["samples"]                       # (T+1, N, 1)
    chains  = samples.transpose(1, 0, 2)             # (N, T+1, 1)
    warmup = num_iter // 4
    chains_post = chains[:, warmup:, :]             # discard burn-in

    print("Diagnostics (post burn-in):")
    summary(chains_post, param_names=["x"])

    # Plot true density vs sampler KDE with TVD
    tvd = plot_against_truth(
        chains_post,
        pi_fn=pi_fn,
        param_name="x",
        save_path="results/density_vs_truth.png",
    )
    print(f"  Estimated TVD: {tvd:.4f}")

    # ---- 5. Add more samplers here --------------------------------

if __name__ == "__main__":
    main()
