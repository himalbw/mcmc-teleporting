import csv
import os
import sys

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm as _norm, t as _t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor")

from diagnostics import ess, r_hat
from samplers.hybrid_teleporting_nuts import HybridTeleportingNUTS
from samplers.parallel_tempering import ParallelTemperingMCMC
from samplers.teleporting_mcmc import (
    TeleportingMCMC,
    gaussian_q_density,
    gaussian_q_sample,
)

DIMS            = list(range(1, 21))
NUM_ITER        = 5000
WARMUP_FRACTION = 0.25
N_WALKERS       = 8
BETAS           = np.geomspace(1.0, 0.01, 6)
SEED            = 221
RESULT_DIR      = os.path.join(ROOT, "results", "dimension_scaling")

METHODS = ["teleporting", "hybrid", "parallel_tempering", "vanilla"]
LABELS  = ["Teleporting", "Hybrid", "PT", "NUTS"]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]


# ── pytensor helpers (used inside vanilla_logp_factory closures) ───────────────

def _gauss_logsum_pt(value, loc, dim, pt):
    """Sum_i log N(value_i; loc, 1) — pytensor."""
    return -0.5 * dim * np.log(2.0 * np.pi) - 0.5 * pt.sum((value - loc) ** 2)


def _t_logsum_pt(value, loc, nu, dim, pt):
    """Sum_i log t_nu(value_i; loc) — pytensor."""
    log_const = dim * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi))
    return log_const - (nu + 1) / 2 * pt.sum(pt.log(1 + (value - loc) ** 2 / nu))


# ── target family definitions ──────────────────────────────────────────────────
# Each family dict must have:
#   name, slug, make_target(dim)->pi_fn, true_marginal(grid)->array,
#   vanilla_logp_factory(dim)->logp_fn, proposal_sigma, x_range, modes_1d

def _bimodal_gaussian_family():
    def make_target(dim):
        c = (2.0 * np.pi) ** (-dim / 2.0)
        L, R = np.full(dim, -5.0), np.full(dim, 5.0)
        def pi_fn(x):
            x = np.asarray(x, dtype=float).ravel()
            return float(0.5 * c * np.exp(-0.5 * np.sum((x - L) ** 2)) +
                         0.5 * c * np.exp(-0.5 * np.sum((x - R) ** 2)))
        return pi_fn

    def true_marginal(grid):
        return 0.5 * _norm.pdf(grid, -5, 1) + 0.5 * _norm.pdf(grid, 5, 1)

    def vanilla_logp_factory(dim):
        def logp(value):
            import pytensor.tensor as pt
            return pt.logaddexp(
                np.log(0.5) + _gauss_logsum_pt(value, -5.0, dim, pt),
                np.log(0.5) + _gauss_logsum_pt(value,  5.0, dim, pt),
            )
        return logp

    return dict(
        name="Bimodal Gaussian",    slug="bimodal_gaussian",
        make_target=make_target,    true_marginal=true_marginal,
        vanilla_logp_factory=vanilla_logp_factory,
        proposal_sigma=1.5,         x_range=(-10.0, 10.0),
        modes_1d=[-5.0, 5.0],
    )


def _bimodal_t3_family():
    nu = 3.0

    def make_target(dim):
        lc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
        def _t_log1d(xi, loc):
            return lc - (nu + 1) / 2 * np.log(1 + (xi - loc) ** 2 / nu)
        def pi_fn(x):
            x = np.asarray(x, dtype=float).ravel()
            ll = sum(_t_log1d(xi, -5.0) for xi in x)
            lr = sum(_t_log1d(xi,  5.0) for xi in x)
            return float(np.exp(np.logaddexp(np.log(0.5) + ll, np.log(0.5) + lr)))
        return pi_fn

    def true_marginal(grid):
        return 0.5 * _t.pdf(grid, df=nu, loc=-5.0) + 0.5 * _t.pdf(grid, df=nu, loc=5.0)

    def vanilla_logp_factory(dim):
        def logp(value):
            import pytensor.tensor as pt
            return pt.logaddexp(
                np.log(0.5) + _t_logsum_pt(value, -5.0, nu, dim, pt),
                np.log(0.5) + _t_logsum_pt(value,  5.0, nu, dim, pt),
            )
        return logp

    return dict(
        name="Bimodal t₃",          slug="bimodal_t3",
        make_target=make_target,    true_marginal=true_marginal,
        vanilla_logp_factory=vanilla_logp_factory,
        proposal_sigma=2.0,         x_range=(-14.0, 14.0),
        modes_1d=[-5.0, 5.0],
    )


def _trimodal_gaussian_family():
    modes_1d = [-7.0, 0.0, 7.0]

    def make_target(dim):
        c = (2.0 * np.pi) ** (-dim / 2.0)
        centers = [np.full(dim, m) for m in modes_1d]
        def pi_fn(x):
            x = np.asarray(x, dtype=float).ravel()
            return float(sum(
                c / 3.0 * np.exp(-0.5 * np.sum((x - ctr) ** 2))
                for ctr in centers
            ))
        return pi_fn

    def true_marginal(grid):
        return sum(_norm.pdf(grid, m, 1.0) / 3.0 for m in modes_1d)

    def vanilla_logp_factory(dim):
        def logp(value):
            import pytensor.tensor as pt
            lw = np.log(1.0 / 3.0)
            return pt.logaddexp(
                pt.logaddexp(
                    lw + _gauss_logsum_pt(value, -7.0, dim, pt),
                    lw + _gauss_logsum_pt(value,  0.0, dim, pt),
                ),
                lw + _gauss_logsum_pt(value, 7.0, dim, pt),
            )
        return logp

    return dict(
        name="Three-mode Gaussian",  slug="trimodal_gaussian",
        make_target=make_target,     true_marginal=true_marginal,
        vanilla_logp_factory=vanilla_logp_factory,
        proposal_sigma=1.5,          x_range=(-12.0, 12.0),
        modes_1d=modes_1d,
    )


def _unequal_weight_family():
    def make_target(dim):
        c = (2.0 * np.pi) ** (-dim / 2.0)
        L, R = np.full(dim, -5.0), np.full(dim, 5.0)
        def pi_fn(x):
            x = np.asarray(x, dtype=float).ravel()
            return float(0.8 * c * np.exp(-0.5 * np.sum((x - L) ** 2)) +
                         0.2 * c * np.exp(-0.5 * np.sum((x - R) ** 2)))
        return pi_fn

    def true_marginal(grid):
        return 0.8 * _norm.pdf(grid, -5, 1) + 0.2 * _norm.pdf(grid, 5, 1)

    def vanilla_logp_factory(dim):
        def logp(value):
            import pytensor.tensor as pt
            return pt.logaddexp(
                np.log(0.8) + _gauss_logsum_pt(value, -5.0, dim, pt),
                np.log(0.2) + _gauss_logsum_pt(value,  5.0, dim, pt),
            )
        return logp

    return dict(
        name="Unequal-weight bimodal (0.8/0.2)", slug="unequal_weight",
        make_target=make_target,    true_marginal=true_marginal,
        vanilla_logp_factory=vanilla_logp_factory,
        proposal_sigma=1.5,         x_range=(-10.0, 10.0),
        modes_1d=[-5.0, 5.0],
    )


TARGET_FAMILIES = [
    _bimodal_gaussian_family(),
    _bimodal_t3_family(),
    _trimodal_gaussian_family(),
    _unequal_weight_family(),
]


# ── core sampling utilities ────────────────────────────────────────────────────

def make_initial_walkers(dim, modes_1d, rng):
    chosen = rng.choice(modes_1d, size=N_WALKERS)
    return rng.normal(loc=chosen[:, np.newaxis], scale=1.0, size=(N_WALKERS, dim))


def average_marginal_tvd(chains, true_marginal_fn, x_range, n_grid=500):
    from scipy.stats import gaussian_kde

    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx   = grid[1] - grid[0]
    p_true = true_marginal_fn(grid)
    tvds = []

    for dim_idx in range(chains.shape[2]):
        samples = chains[:, :, dim_idx].ravel()
        if np.std(samples) < 1e-10:
            tvds.append(0.5)   # completely stuck — point mass → TVD ≈ 0.5
            continue
        p_kde = gaussian_kde(samples, bw_method="scott")(grid)
        tvds.append(0.5 * np.sum(np.abs(p_true - p_kde)) * dx)

    return float(np.mean(tvds))


def run_vanilla_nuts(dim, family, num_iter, warmup):
    try:
        import pymc as pm
    except Exception as exc:
        print(f"    skipped: PyMC could not be imported ({exc})")
        return None

    logp_fn = family["vanilla_logp_factory"](dim)
    with pm.Model():
        pm.DensityDist("x", logp=logp_fn, shape=dim, initval=np.zeros(dim))
        init_rng = np.random.default_rng(SEED + dim)
        initvals = [{"x": init_rng.normal(loc=0.0, scale=6.0, size=dim)} for _ in range(4)]
        trace = pm.sample(
            draws=num_iter, tune=warmup, chains=4, cores=1,
            initvals=initvals, random_seed=SEED,
            progressbar=False, compute_convergence_checks=False,
        )

    samples = trace.posterior["x"].values
    if samples.ndim == 2:
        samples = samples[:, :, np.newaxis]
    return samples


def run_dimension(dim, family, rng, num_iter=NUM_ITER):
    warmup       = int(num_iter * WARMUP_FRACTION)
    pi_fn        = family["make_target"](dim)
    x0           = make_initial_walkers(dim, family["modes_1d"], rng)
    sigma        = family["proposal_sigma"]
    x_range      = family["x_range"]
    true_marginal = family["true_marginal"]

    q_sample_fn  = lambda x, r: gaussian_q_sample(x, sigma, r)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, sigma)

    row = {"dimension": dim}

    print("  [Teleporting MCMC]")
    t_sampler = TeleportingMCMC(pi_fn, q_sample_fn, q_density_fn, rng=rng)
    t_chains  = t_sampler.run(x0, num_iter)["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_teleporting"]  = round(average_marginal_tvd(t_chains, true_marginal, x_range), 4)
    row["ess_teleporting"]  = round(float(np.nanmean(ess(t_chains))), 1)
    row["rhat_teleporting"] = round(float(np.nanmean(r_hat(t_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_teleporting']:.4f}")

    print("  [Hybrid Teleporting-NUTS]")
    h_sampler = HybridTeleportingNUTS(
        pi_fn=pi_fn, q_sample_fn=q_sample_fn, q_density_fn=q_density_fn,
        init_step_size=sigma / 5.0, max_tree_depth=5, target_accept=0.65, rng=rng,
    )
    h_chains = h_sampler.run(x0, num_iter, num_warmup=warmup)["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_hybrid"]  = round(average_marginal_tvd(h_chains, true_marginal, x_range), 4)
    row["ess_hybrid"]  = round(float(np.nanmean(ess(h_chains))), 1)
    row["rhat_hybrid"] = round(float(np.nanmean(r_hat(h_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_hybrid']:.4f}")

    print("  [Parallel Tempering]")
    pt_x0     = np.tile(x0[0], (len(BETAS), 1))
    pt_sampler = ParallelTemperingMCMC(
        pi_fn=pi_fn, inverse_temperatures=BETAS, proposal_scale=sigma, rng=rng,
    )
    pt_chains = pt_sampler.run(pt_x0, num_iter=num_iter)["cold_samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_parallel_tempering"]  = round(average_marginal_tvd(pt_chains, true_marginal, x_range), 4)
    row["ess_parallel_tempering"]  = round(float(np.nanmean(ess(pt_chains))), 1)
    row["rhat_parallel_tempering"] = round(float(np.nanmean(r_hat(pt_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_parallel_tempering']:.4f}")

    print("  [Vanilla NUTS]")
    v_chains = run_vanilla_nuts(dim, family, num_iter, warmup)
    if v_chains is None:
        row["tvd_vanilla"] = row["ess_vanilla"] = row["rhat_vanilla"] = float("nan")
    else:
        row["tvd_vanilla"]  = round(average_marginal_tvd(v_chains, true_marginal, x_range), 4)
        row["ess_vanilla"]  = round(float(np.nanmean(ess(v_chains))), 1)
        row["rhat_vanilla"] = round(float(np.nanmean(r_hat(v_chains))), 3)
        print(f"    avg marginal TVD={row['tvd_vanilla']:.4f}")

    return row


# ── output ────────────────────────────────────────────────────────────────────

def save_results(results_by_family):
    import matplotlib.pyplot as plt

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── per-family CSVs + backward-compat files for the first (Gaussian) family
    for family, rows in results_by_family:
        csv_path = os.path.join(RESULT_DIR, f"tvd_{family['slug']}.csv")
        fieldnames = ["dimension"] + [f"tvd_{m}" for m in METHODS]
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerows(rows)
        print(f"Saved {csv_path}")

    # backward-compat: keep tvd_by_dimension.csv / tvd_by_dimension.png
    # pointing at the first (bimodal Gaussian) family
    first_family, first_rows = results_by_family[0]
    compat_csv = os.path.join(RESULT_DIR, "tvd_by_dimension.csv")
    fieldnames = ["dimension"] + [f"tvd_{m}" for m in METHODS]
    with open(compat_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(first_rows)

    metrics_csv = os.path.join(RESULT_DIR, "metrics_by_dimension.csv")
    all_metrics = ["tvd", "ess", "rhat"]
    fieldnames_m = ["dimension"] + [f"{mt}_{m}" for mt in all_metrics for m in METHODS]
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_m, extrasaction="ignore")
        w.writeheader(); w.writerows(first_rows)

    # ── 2×2 robustness plot
    n = len(results_by_family)
    ncols, nrows = 2, (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (family, rows) in enumerate(results_by_family):
        ax   = axes[idx // ncols, idx % ncols]
        dims = [r["dimension"] for r in rows]
        for method, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
            tvds = [r[f"tvd_{method}"] for r in rows]
            ax.plot(dims, tvds, label=label, color=color, marker=marker,
                    linewidth=2, markersize=5)
        ax.set_title(family["name"], fontsize=12)
        ax.set_xlabel("Dimension", fontsize=10)
        ax.set_ylabel("Avg marginal TVD", fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("TVD vs. dimension — robustness across target families (lower is better)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    rob_png = os.path.join(RESULT_DIR, "tvd_robustness.png")
    fig.savefig(rob_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {rob_png}")

    # ── also save the single-family line plot (tvd_by_dimension.png) for the
    #    bimodal Gaussian so existing references still work
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    dims = [r["dimension"] for r in first_rows]
    for method, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
        tvds = [r[f"tvd_{method}"] for r in first_rows]
        ax2.plot(dims, tvds, label=label, color=color, marker=marker,
                 linewidth=2, markersize=6)
    ax2.set_xlabel("Dimension", fontsize=12)
    ax2.set_ylabel("Average marginal TVD", fontsize=12)
    ax2.set_title(f"TVD vs. dimension — {first_family['name']} (lower is better)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xticks(dims)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    single_png = os.path.join(RESULT_DIR, "tvd_by_dimension.png")
    fig2.savefig(single_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {single_png}")


def main():
    rng = np.random.default_rng(SEED)
    print(f"Dimensions: {DIMS}  |  Iterations: {NUM_ITER}  |  Warmup: {int(NUM_ITER * WARMUP_FRACTION)}")
    print(f"Families:   {[f['name'] for f in TARGET_FAMILIES]}\n")

    results_by_family = []
    for family in TARGET_FAMILIES:
        print(f"\n{'#' * 68}")
        print(f"  Family: {family['name']}")
        print(f"{'#' * 68}")
        rows = []
        for dim in DIMS:
            print(f"\n  d={dim}")
            rows.append(run_dimension(dim, family, rng))
        results_by_family.append((family, rows))

    save_results(results_by_family)


if __name__ == "__main__":
    main()
