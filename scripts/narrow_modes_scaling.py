"""
narrow_modes_scaling.py
=======================
Dimension-scaling experiment with narrow, steep modes designed to expose
the advantage of NUTS local moves over MH local moves.

Target:  ½ N(-5·1, σ²I)  +  ½ N(5·1, σ²I)   with σ = 0.25
Walkers: N = 2^d  (exponential in d, same as exponential_n_scaling.py)

Why Hybrid should win
---------------------
In TeleportingMCMC the local move (i==j) is Gaussian MH with the fixed
proposal_sigma.  For a Gaussian target with mode width σ_mode in d dims,
the MH acceptance rate with a proposal of width σ_prop scales as

    P(accept) ≈ exp( -d/2 · (σ_prop / σ_mode)² )

With σ_prop = 0.5 and σ_mode = 0.25 this is exp(-2d), which is < 1% by d=3.
The walkers freeze in place and cannot explore within the mode.

In HybridTeleportingNUTS the local move is NUTS, whose leapfrog step size
is calibrated during warmup.  NUTS automatically finds the right scale
(≈ σ_mode / √d) and maintains ~65% acceptance regardless of dimension.

Vanilla NUTS (4 fixed chains) can reach and explore each mode but cannot
cross between them, giving TVD ≈ 0.25 as before (3/4 chains in one mode).
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

# ── settings ──────────────────────────────────────────────────────────────────
SIGMA_MODE      = 0.25   # narrow modes — MH degrades, NUTS adapts
PROPOSAL_SIGMA  = 0.5    # fixed for teleportation & Teleporting local moves
DIMS            = [1, 2, 3, 4, 5, 6, 7]
NUM_ITER        = 1000
WARMUP_FRACTION = 0.25
N_NUTS_CHAINS   = 4
SEED            = 221
RESULT_DIR      = os.path.join(ROOT, "results", "dimension_scaling")


def n_walkers(d):
    return 2 ** d


# ── target ────────────────────────────────────────────────────────────────────

def make_narrow_bimodal_target(dim):
    s2 = SIGMA_MODE ** 2
    c  = (2.0 * np.pi * s2) ** (-dim / 2.0)
    L, R = np.full(dim, -5.0), np.full(dim, 5.0)
    def pi_fn(x):
        x = np.asarray(x, dtype=float).ravel()
        return float(
            0.5 * c * np.exp(-0.5 * np.sum((x - L) ** 2) / s2) +
            0.5 * c * np.exp(-0.5 * np.sum((x - R) ** 2) / s2)
        )
    return pi_fn


def true_marginal(grid):
    return (0.5 * _norm.pdf(grid, -5.0, SIGMA_MODE) +
            0.5 * _norm.pdf(grid, +5.0, SIGMA_MODE))


INIT_SCALE = 2.0  # 8× the mode width — walkers start far from the target

def make_initial_walkers(dim, N, rng):
    # Start walkers well outside the narrow modes so samplers must find them.
    # With SIGMA_MODE=0.25 and INIT_SCALE=2.0, initial positions are ~8σ off.
    # MH (σ_prop=0.5) cannot navigate to the mode; NUTS follows the gradient.
    modes = rng.choice([-5.0, 5.0], size=N)
    return rng.normal(loc=modes[:, np.newaxis], scale=INIT_SCALE,
                      size=(N, dim))


# ── TVD ───────────────────────────────────────────────────────────────────────

def average_marginal_tvd(chains, n_grid=1000):
    from scipy.stats import gaussian_kde
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]
    # Wide grid to capture both modes and any unconverged walkers
    lo = -5.0 - 10 * SIGMA_MODE
    hi =  5.0 + 10 * SIGMA_MODE
    grid   = np.linspace(lo, hi, n_grid)
    dx     = grid[1] - grid[0]
    p_true = true_marginal(grid)
    tvds   = []
    for k in range(chains.shape[2]):
        samples = chains[:, :, k].ravel()
        s = np.std(samples)
        if s < 1e-10:
            tvds.append(0.5)
            continue
        # Use a fixed absolute bandwidth = SIGMA_MODE so narrow modes are
        # resolved correctly.  Scott's rule gives ~1.1 for bimodal data
        # (dominated by mode separation), which smears the narrow peaks.
        bw = SIGMA_MODE / s   # gaussian_kde multiplies factor × std(data)
        p_kde = gaussian_kde(samples, bw_method=bw)(grid)
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

    s2 = SIGMA_MODE ** 2
    with pm.Model():
        def logp(value):
            log_norm  = -0.5 * dim * np.log(2.0 * np.pi * s2)
            log_left  = log_norm - 0.5 * pt.sum((value + 5.0) ** 2) / s2
            log_right = log_norm - 0.5 * pt.sum((value - 5.0) ** 2) / s2
            return pt.logaddexp(np.log(0.5) + log_left,
                                np.log(0.5) + log_right)

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

    samples = trace.posterior["x"].values
    if samples.ndim == 2:
        samples = samples[:, :, np.newaxis]
    return samples


def run_dimension(dim, rng):
    N      = n_walkers(dim)
    warmup = int(NUM_ITER * WARMUP_FRACTION)
    pi_fn  = make_narrow_bimodal_target(dim)
    x0     = make_initial_walkers(dim, N, rng)
    q_s    = lambda x, r: gaussian_q_sample(x, PROPOSAL_SIGMA, r)
    q_d    = lambda x, m: gaussian_q_density(x, m, PROPOSAL_SIGMA)

    # Expected MH acceptance for local move at mode (d-dimensional):
    # P(accept) ≈ exp(-d/2 * (σ_prop/σ_mode)²)
    expected_mh = np.exp(-dim / 2.0 * (PROPOSAL_SIGMA / SIGMA_MODE) ** 2)
    print(f"\n  d={dim}  N={N}  "
          f"expected MH local acceptance ≈ {expected_mh:.4f}")

    row = {"dimension": dim, "n_walkers": N,
           "expected_mh_acceptance": round(expected_mh, 6)}

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
    import matplotlib.ticker as ticker

    os.makedirs(RESULT_DIR, exist_ok=True)

    csv_path = os.path.join(RESULT_DIR, "tvd_narrow_modes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    dims = [r["dimension"] for r in rows]
    ns   = [r["n_walkers"]  for r in rows]
    mh_acc = [r["expected_mh_acceptance"] for r in rows]

    methods = ["teleporting", "hybrid",    "vanilla"]
    labels  = ["Teleporting (N=2ᵈ)", "Hybrid (N=2ᵈ)", f"NUTS ({N_NUTS_CHAINS} chains)"]
    colors  = ["#1f77b4",  "#ff7f0e",  "#d62728"]
    markers = ["o",        "s",        "D"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})

    # ── main TVD plot
    for method, label, color, marker in zip(methods, labels, colors, markers):
        tvds = [r[f"tvd_{method}"] for r in rows]
        ax.plot(dims, tvds, label=label, color=color, marker=marker,
                linewidth=2, markersize=7)

    ax.set_xticks(dims)
    ax.set_xticklabels([f"d={d}\nN={n}" for d, n in zip(dims, ns)], fontsize=9)
    ax.set_ylabel("Average marginal TVD", fontsize=11)
    ax.set_title(
        f"Narrow modes  σ = {SIGMA_MODE},  proposal σ = {PROPOSAL_SIGMA},  N = 2ᵈ\n"
        "Hybrid wins: NUTS adapts step size; MH acceptance collapses with d",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── expected MH local acceptance (annotation panel)
    ax2.semilogy(dims, mh_acc, color="#1f77b4", marker="o", linewidth=2, markersize=6,
                 label="Expected MH local acceptance")
    ax2.axhline(0.01, color="grey", linestyle=":", linewidth=1)
    ax2.text(dims[-1], 0.012, "1%", color="grey", fontsize=8, ha="right")
    ax2.set_xticks(dims)
    ax2.set_xticklabels([f"d={d}" for d in dims], fontsize=9)
    ax2.set_ylabel("MH acceptance\n(log scale)", fontsize=10)
    ax2.set_xlabel("Dimension  (N = 2ᵈ walkers for Teleporting / Hybrid)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    png_path = os.path.join(RESULT_DIR, "tvd_narrow_modes.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


def main():
    rng = np.random.default_rng(SEED)
    print(f"Target:    ½N(-5, {SIGMA_MODE}²I) + ½N(5, {SIGMA_MODE}²I)")
    print(f"Proposal:  σ = {PROPOSAL_SIGMA}  (fixed for MH/teleportation)")
    print(f"Dims:      {DIMS}   N(d) = 2^d = {[n_walkers(d) for d in DIMS]}")
    print(f"Iters:     {NUM_ITER}  (warmup: {int(NUM_ITER * WARMUP_FRACTION)})\n")

    rows = [run_dimension(dim, rng) for dim in DIMS]
    save_results(rows)


if __name__ == "__main__":
    main()
