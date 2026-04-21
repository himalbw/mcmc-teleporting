import csv
import os
import sys

import numpy as np

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


DIMS = [1, 2, 5, 10, 20]
NUM_ITER = 5000
WARMUP_FRACTION = 0.25
N_WALKERS = 8
MODE_SEPARATION = 5.0
PROPOSAL_SIGMA = 1.5
BETAS = np.geomspace(1.0, 0.01, 6)
SEED = 221
RESULT_DIR = os.path.join(ROOT, "results", "dimension_scaling")


def make_bimodal_target(dim, mode_separation=MODE_SEPARATION):
    left = np.full(dim, -mode_separation)
    right = np.full(dim, mode_separation)
    norm_const = (2.0 * np.pi) ** (-dim / 2.0)

    def pi_fn(x):
        x = np.asarray(x, dtype=float).ravel()
        left_density = norm_const * np.exp(-0.5 * np.sum((x - left) ** 2))
        right_density = norm_const * np.exp(-0.5 * np.sum((x - right) ** 2))
        return float(0.5 * left_density + 0.5 * right_density)

    return pi_fn, left, right


def make_initial_walkers(dim, rng):
    modes = rng.choice([-MODE_SEPARATION, MODE_SEPARATION], size=N_WALKERS)
    return rng.normal(loc=modes[:, np.newaxis], scale=1.0, size=(N_WALKERS, dim))


def marginal_mixture_density(grid):
    left = np.exp(-0.5 * (grid + MODE_SEPARATION) ** 2) / np.sqrt(2.0 * np.pi)
    right = np.exp(-0.5 * (grid - MODE_SEPARATION) ** 2) / np.sqrt(2.0 * np.pi)
    return 0.5 * left + 0.5 * right


def average_marginal_tvd(chains, x_range=(-10.0, 10.0), n_grid=500):
    from scipy.stats import gaussian_kde

    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx = grid[1] - grid[0]
    p_true = marginal_mixture_density(grid)
    tvds = []

    for dim_idx in range(chains.shape[2]):
        samples = chains[:, :, dim_idx].ravel()
        p_kde = gaussian_kde(samples, bw_method="scott")(grid)
        tvds.append(0.5 * np.sum(np.abs(p_true - p_kde)) * dx)

    return float(np.mean(tvds))


def run_vanilla_nuts(dim, num_iter, warmup):
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except Exception as exc:
        print(f"    skipped: PyMC could not be imported ({exc})")
        return None

    with pm.Model():

        def logp(value):
            left_diff = value + MODE_SEPARATION
            right_diff = value - MODE_SEPARATION
            log_norm = -0.5 * dim * np.log(2.0 * np.pi)
            log_left = log_norm - 0.5 * pt.sum(left_diff**2)
            log_right = log_norm - 0.5 * pt.sum(right_diff**2)
            return pt.logaddexp(np.log(0.5) + log_left, np.log(0.5) + log_right)

        pm.DensityDist(
            "x",
            logp=logp,
            shape=dim,
            initval=np.full(dim, -MODE_SEPARATION),
        )

        trace = pm.sample(
            draws=num_iter,
            tune=warmup,
            chains=4,
            cores=1,
            random_seed=SEED,
            progressbar=False,
            compute_convergence_checks=False,
        )

    samples = trace.posterior["x"].values
    if samples.ndim == 2:
        samples = samples[:, :, np.newaxis]
    return samples


def run_dimension(dim, rng, num_iter=NUM_ITER):
    warmup = int(num_iter * WARMUP_FRACTION)
    pi_fn, _, _ = make_bimodal_target(dim)
    x0 = make_initial_walkers(dim, rng)
    q_sample_fn = lambda x, r: gaussian_q_sample(x, PROPOSAL_SIGMA, r)
    q_density_fn = lambda x, mean: gaussian_q_density(x, mean, PROPOSAL_SIGMA)

    print(f"\n{'=' * 68}")
    print(f"  Bimodal mixture dimension d={dim}")
    print(f"{'=' * 68}")

    row = {"dimension": dim}

    print("  [Teleporting MCMC]")
    t_sampler = TeleportingMCMC(pi_fn, q_sample_fn, q_density_fn, rng=rng)
    t_result = t_sampler.run(x0, num_iter)
    t_chains = t_result["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_teleporting"] = round(average_marginal_tvd(t_chains), 4)
    row["ess_teleporting"] = round(float(np.nanmean(ess(t_chains))), 1)
    row["rhat_teleporting"] = round(float(np.nanmean(r_hat(t_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_teleporting']:.4f}")

    print("  [Hybrid Teleporting-NUTS]")
    h_sampler = HybridTeleportingNUTS(
        pi_fn=pi_fn,
        q_sample_fn=q_sample_fn,
        q_density_fn=q_density_fn,
        init_step_size=PROPOSAL_SIGMA / 5.0,
        max_tree_depth=5,
        target_accept=0.65,
        rng=rng,
    )
    h_result = h_sampler.run(x0, num_iter, num_warmup=warmup)
    h_chains = h_result["samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_hybrid"] = round(average_marginal_tvd(h_chains), 4)
    row["ess_hybrid"] = round(float(np.nanmean(ess(h_chains))), 1)
    row["rhat_hybrid"] = round(float(np.nanmean(r_hat(h_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_hybrid']:.4f}")

    print("  [Parallel Tempering]")
    pt_x0 = np.tile(x0[0], (len(BETAS), 1))
    pt_sampler = ParallelTemperingMCMC(
        pi_fn=pi_fn,
        inverse_temperatures=BETAS,
        proposal_scale=PROPOSAL_SIGMA,
        rng=rng,
    )
    pt_result = pt_sampler.run(pt_x0, num_iter=num_iter)
    pt_chains = pt_result["cold_samples"].transpose(1, 0, 2)[:, warmup:, :]
    row["tvd_parallel_tempering"] = round(average_marginal_tvd(pt_chains), 4)
    row["ess_parallel_tempering"] = round(float(np.nanmean(ess(pt_chains))), 1)
    row["rhat_parallel_tempering"] = round(float(np.nanmean(r_hat(pt_chains))), 3)
    print(f"    avg marginal TVD={row['tvd_parallel_tempering']:.4f}")

    print("  [Vanilla NUTS]")
    v_chains = run_vanilla_nuts(dim, num_iter, warmup)
    if v_chains is None:
        row["tvd_vanilla"] = float("nan")
        row["ess_vanilla"] = float("nan")
        row["rhat_vanilla"] = float("nan")
    else:
        row["tvd_vanilla"] = round(average_marginal_tvd(v_chains), 4)
        row["ess_vanilla"] = round(float(np.nanmean(ess(v_chains))), 1)
        row["rhat_vanilla"] = round(float(np.nanmean(r_hat(v_chains))), 3)
        print(f"    avg marginal TVD={row['tvd_vanilla']:.4f}")

    return row


def save_dimension_table(rows):
    import matplotlib.pyplot as plt

    os.makedirs(RESULT_DIR, exist_ok=True)
    methods = ["teleporting", "hybrid", "parallel_tempering", "vanilla"]
    labels = ["Teleporting", "Hybrid", "PT", "NUTS"]
    metrics = ["tvd", "ess", "rhat"]

    csv_path = os.path.join(RESULT_DIR, "metrics_by_dimension.csv")
    fieldnames = ["dimension"] + [
        f"{metric}_{method}" for metric in metrics for method in methods
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    tvd_csv_path = os.path.join(RESULT_DIR, "tvd_by_dimension.csv")
    with open(tvd_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dimension"] + [f"tvd_{method}" for method in methods],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    raw = [[r[f"tvd_{m}"] for m in methods] for r in rows]
    cell_text = [
        [f"{v:.4f}" if not np.isnan(v) else "--" for v in row]
        for row in raw
    ]
    row_labels = [f"{r['dimension']}D" for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.55 * len(rows) + 1.6)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for col in range(len(methods)):
        table[0, col].set_facecolor("#cfd8dc")
        table[0, col].set_text_props(fontweight="bold")

    for row_idx, row in enumerate(raw):
        finite = np.array(row, dtype=float)
        if np.all(np.isnan(finite)):
            continue
        best = int(np.nanargmin(finite))
        worst = int(np.nanargmax(finite))
        for col_idx in range(len(methods)):
            cell = table[row_idx + 1, col_idx]
            if col_idx == best:
                cell.set_facecolor("#c8e6c9")
            elif col_idx == worst:
                cell.set_facecolor("#ffcdd2")

    ax.set_title("Average marginal TVD by dimension (lower is better)", pad=16)
    png_path = os.path.join(RESULT_DIR, "tvd_by_dimension.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {csv_path}")
    print(f"Saved {tvd_csv_path}")
    print(f"Saved {png_path}")


def main():
    rng = np.random.default_rng(SEED)
    warmup = int(NUM_ITER * WARMUP_FRACTION)
    print(
        f"Running dimension scaling with {NUM_ITER} iterations, "
        f"{warmup} warmup iterations, dimensions={DIMS}."
    )
    print("Metric: average 1D marginal TVD across coordinates.")

    rows = [run_dimension(dim, rng, NUM_ITER) for dim in DIMS]
    save_dimension_table(rows)


if __name__ == "__main__":
    main()
