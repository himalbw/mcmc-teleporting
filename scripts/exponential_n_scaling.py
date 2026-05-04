"""
exponential_n_scaling.py
========================
Dimension-scaling experiment where the number of walkers N = 2^d grows
exponentially with dimension.

Theory (Skene et al. 2023): teleportation fires when walkers are close
relative to the proposal width, which requires N exponential in d.  When
walkers are too sparse (constant N), the deletion weights collapse to i=j,
the partition functions cancel, and the scheme reduces to N independent
chains — graceful degeneration, but no interaction bonus.  With N = 2^d,
the interaction is maintained and teleportation fires meaningfully across
all dimensions.

Compares: Teleporting MCMC, Hybrid Teleporting-NUTS, Vanilla NUTS (4 chains).
Target:   bimodal Gaussian  ½N(-5,1) + ½N(5,1)  per coordinate.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import norm as _norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor")

from samplers.hybrid_teleporting_nuts import HybridTeleportingNUTS
from samplers.teleporting_mcmc import (
    TeleportingMCMC,
    gaussian_q_density,
    gaussian_q_sample,
)

# ── experiment settings ────────────────────────────────────────────────────────
# N(d) = 2^d walkers for Teleporting / Hybrid; 4 fixed chains for NUTS.
# Cost per step is O(N²), so the series d=1..7 (N=2..128) is tractable.
DIMS            = [1, 2, 3, 4, 5, 6, 7]
NUM_ITER        = 1000
WARMUP_FRACTION = 0.25
PROPOSAL_SIGMA  = 1.5
N_NUTS_CHAINS   = 4
SEED            = 221
RESULT_DIR      = os.path.join(ROOT, "results", "dimension_scaling")


def n_walkers(d):
    return 2 ** d


# ── target ────────────────────────────────────────────────────────────────────

def make_bimodal_target(dim):
    c = (2.0 * np.pi) ** (-dim / 2.0)
    L, R = np.full(dim, -5.0), np.full(dim, 5.0)
    def pi_fn(x):
        x = np.asarray(x, dtype=float).ravel()
        return float(0.5 * c * np.exp(-0.5 * np.sum((x - L) ** 2)) +
                     0.5 * c * np.exp(-0.5 * np.sum((x - R) ** 2)))
    return pi_fn


def true_marginal(grid):
    return 0.5 * _norm.pdf(grid, -5, 1) + 0.5 * _norm.pdf(grid, 5, 1)


def make_initial_walkers(dim, N, rng):
    modes = rng.choice([-5.0, 5.0], size=N)
    return rng.normal(loc=modes[:, np.newaxis], scale=1.0, size=(N, dim))


# ── TVD ───────────────────────────────────────────────────────────────────────

def average_marginal_tvd(chains, n_grid=500):
    from scipy.stats import gaussian_kde
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]
    grid   = np.linspace(-10.0, 10.0, n_grid)
    dx     = grid[1] - grid[0]
    p_true = true_marginal(grid)
    tvds   = []
    for k in range(chains.shape[2]):
        samples = chains[:, :, k].ravel()
        if np.std(samples) < 1e-10:
            tvds.append(0.5)
            continue
        p_kde = gaussian_kde(samples, bw_method="scott")(grid)
        tvds.append(0.5 * np.sum(np.abs(p_true - p_kde)) * dx)
    return float(np.mean(tvds))


# ── samplers ──────────────────────────────────────────────────────────────────

def run_vanilla_nuts(dim, num_iter, warmup):
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except Exception as exc:
        print(f"    skipped: {exc}")
        return None

    with pm.Model():
        def logp(value):
            log_norm  = -0.5 * dim * np.log(2.0 * np.pi)
            log_left  = log_norm - 0.5 * pt.sum((value + 5.0) ** 2)
            log_right = log_norm - 0.5 * pt.sum((value - 5.0) ** 2)
            return pt.logaddexp(np.log(0.5) + log_left, np.log(0.5) + log_right)

        pm.DensityDist("x", logp=logp, shape=dim, initval=np.zeros(dim))
        init_rng = np.random.default_rng(SEED + dim)
        initvals = [{"x": init_rng.normal(0.0, 6.0, size=dim)}
                    for _ in range(N_NUTS_CHAINS)]
        trace = pm.sample(
            draws=num_iter, tune=warmup,
            chains=N_NUTS_CHAINS, cores=1,
            initvals=initvals, random_seed=SEED,
            progressbar=False, compute_convergence_checks=False,
        )

    samples = trace.posterior["x"].values  # (chains, draws, dim)
    if samples.ndim == 2:
        samples = samples[:, :, np.newaxis]
    return samples


def run_dimension(dim, rng):
    N       = n_walkers(dim)
    warmup  = int(NUM_ITER * WARMUP_FRACTION)
    pi_fn   = make_bimodal_target(dim)
    x0      = make_initial_walkers(dim, N, rng)
    q_s     = lambda x, r: gaussian_q_sample(x, PROPOSAL_SIGMA, r)
    q_d     = lambda x, m: gaussian_q_density(x, m, PROPOSAL_SIGMA)

    print(f"\n  d={dim}  N={N} walkers")
    row = {"dimension": dim, "n_walkers": N}

    print("  [Teleporting MCMC]")
    t_sampler = TeleportingMCMC(pi_fn, q_s, q_d, rng=rng)
    t_chains  = (t_sampler.run(x0, NUM_ITER)["samples"]
                 .transpose(1, 0, 2)[:, warmup:, :])
    row["tvd_teleporting"] = round(average_marginal_tvd(t_chains), 4)
    print(f"    TVD = {row['tvd_teleporting']:.4f}")

    print("  [Hybrid Teleporting-NUTS]")
    h_sampler = HybridTeleportingNUTS(
        pi_fn=pi_fn, q_sample_fn=q_s, q_density_fn=q_d,
        init_step_size=PROPOSAL_SIGMA / 5.0, max_tree_depth=5,
        target_accept=0.65, rng=rng,
    )
    h_chains = (h_sampler.run(x0, NUM_ITER, num_warmup=warmup)["samples"]
                .transpose(1, 0, 2)[:, warmup:, :])
    row["tvd_hybrid"] = round(average_marginal_tvd(h_chains), 4)
    print(f"    TVD = {row['tvd_hybrid']:.4f}")

    print(f"  [Vanilla NUTS ({N_NUTS_CHAINS} chains)]")
    v_chains = run_vanilla_nuts(dim, NUM_ITER, warmup)
    if v_chains is None:
        row["tvd_vanilla"] = float("nan")
    else:
        row["tvd_vanilla"] = round(average_marginal_tvd(v_chains), 4)
        print(f"    TVD = {row['tvd_vanilla']:.4f}")

    return row


# ── output ────────────────────────────────────────────────────────────────────

def save_results(rows):
    import matplotlib.pyplot as plt

    os.makedirs(RESULT_DIR, exist_ok=True)

    csv_path = os.path.join(RESULT_DIR, "tvd_exponential_n.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    dims = [r["dimension"] for r in rows]
    ns   = [r["n_walkers"]  for r in rows]

    methods = ["teleporting", "hybrid",   "vanilla"]
    labels  = ["Teleporting (N=2ᵈ)", "Hybrid (N=2ᵈ)", f"NUTS ({N_NUTS_CHAINS} chains)"]
    colors  = ["#1f77b4",  "#ff7f0e",  "#d62728"]
    markers = ["o",        "s",        "D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        tvds = [r[f"tvd_{method}"] for r in rows]
        ax.plot(dims, tvds, label=label, color=color, marker=marker,
                linewidth=2, markersize=7)

    ax.set_xticks(dims)
    ax.set_xticklabels([f"d={d}\nN={n}" for d, n in zip(dims, ns)], fontsize=9)
    ax.set_xlabel("Dimension  (N = 2ᵈ walkers for Teleporting / Hybrid)", fontsize=11)
    ax.set_ylabel("Average marginal TVD", fontsize=11)
    ax.set_title("TVD vs. dimension with exponential walker count  N = 2ᵈ\n"
                 "(lower is better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    png_path = os.path.join(RESULT_DIR, "tvd_exponential_n.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


def main():
    rng = np.random.default_rng(SEED)
    print(f"Dimensions:     {DIMS}")
    print(f"N(d) = 2^d:     {[n_walkers(d) for d in DIMS]}")
    print(f"Iterations:     {NUM_ITER}  (warmup: {int(NUM_ITER * WARMUP_FRACTION)})")
    print(f"NUTS chains:    {N_NUTS_CHAINS}  (fixed)\n")

    rows = [run_dimension(dim, rng) for dim in DIMS]
    save_results(rows)


if __name__ == "__main__":
    main()
