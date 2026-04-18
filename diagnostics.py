"""
MCMC diagnostics: R-hat, ESS, 95% CI, summary table, density plot vs truth,
and per-scenario comparison figures.

All chain arrays follow the shape convention (num_chains, num_draws, num_params).
"""

import os
import numpy as np


def _ensure_3d(chains):
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]
    if chains.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {chains.shape}")
    return chains


# ------------------------------------------------------------------
# R-hat  (split-chain Gelman-Rubin, BDA3 §11.4)
# ------------------------------------------------------------------

def r_hat(chains):
    """
    Compute split-chain R-hat for each parameter.

    Parameters
    ----------
    chains : array-like, shape (M, N, P)

    Returns
    -------
    ndarray, shape (P,)
    """
    chains = _ensure_3d(chains)
    M, N, P = chains.shape

    n = N // 2
    split = np.concatenate([chains[:, :n, :], chains[:, n:2*n, :]], axis=0)
    M2 = split.shape[0]

    chain_means = split.mean(axis=1)
    grand_mean  = chain_means.mean(axis=0)

    B = n / (M2 - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(split.var(axis=1, ddof=1), axis=0)

    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / np.where(W > 0, W, np.nan))


# ------------------------------------------------------------------
# ESS  (bulk, via FFT autocorrelation)
# ------------------------------------------------------------------

def _autocorr(x):
    n = len(x)
    x = x - x.mean()
    fft_len = 1 << (2 * n - 1).bit_length()
    f   = np.fft.rfft(x, n=fft_len)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    return acf


def ess(chains):
    """
    Estimate bulk ESS for each parameter (Geyer initial positive-pair estimator).

    Parameters
    ----------
    chains : array-like, shape (M, N, P)

    Returns
    -------
    ndarray, shape (P,)
    """
    chains = _ensure_3d(chains)
    M, N, P = chains.shape
    ess_vals = np.zeros(P)

    for p in range(P):
        rho = np.zeros(N)
        for m in range(M):
            rho += _autocorr(chains[m, :, p])
        rho /= M

        tau = 1.0
        for t in range(1, N // 2):
            pair = rho[2 * t - 1] + rho[2 * t]
            if pair < 0:
                break
            tau += 2 * pair

        ess_vals[p] = M * N / tau

    return ess_vals


# ------------------------------------------------------------------
# 95% credible interval
# ------------------------------------------------------------------

def ci_95(chains):
    chains = _ensure_3d(chains)
    flat   = chains.reshape(-1, chains.shape[-1])
    return np.percentile(flat, 2.5, axis=0), np.percentile(flat, 97.5, axis=0)


# ------------------------------------------------------------------
# Posterior mean and std
# ------------------------------------------------------------------

def posterior_mean(chains):
    chains = _ensure_3d(chains)
    return chains.reshape(-1, chains.shape[-1]).mean(axis=0)


def posterior_std(chains):
    chains = _ensure_3d(chains)
    return chains.reshape(-1, chains.shape[-1]).std(axis=0, ddof=1)


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def summary(chains, param_names=None):
    """Print a diagnostics summary table."""
    chains = _ensure_3d(chains)
    P = chains.shape[2]

    if param_names is None:
        param_names = [f"param[{i}]" for i in range(P)]

    mean   = posterior_mean(chains)
    std    = posterior_std(chains)
    lo, hi = ci_95(chains)
    rhat   = r_hat(chains)
    n_eff  = ess(chains)

    col_w  = max(max(len(n) for n in param_names), 10)
    header = (
        f"{'param':<{col_w}}  {'mean':>10}  {'std':>10}  "
        f"{'2.5%':>10}  {'97.5%':>10}  {'R-hat':>7}  {'ESS':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, name in enumerate(param_names):
        print(
            f"{name:<{col_w}}  {mean[i]:>10.4f}  {std[i]:>10.4f}  "
            f"{lo[i]:>10.4f}  {hi[i]:>10.4f}  {rhat[i]:>7.3f}  {n_eff[i]:>8.1f}"
        )
    print(sep)


# ------------------------------------------------------------------
# Single-panel density plot vs truth + TVD estimate
# ------------------------------------------------------------------

def plot_against_truth(
    chains,
    pi_fn,
    param_idx=0,
    param_name=None,
    x_range=None,
    n_grid=500,
    save_path=None,
    ax=None,
):
    """
    Plot the sampler's marginal density for one parameter against the true
    target and annotate with estimated TVD.

    Parameters
    ----------
    chains     : array-like, shape (M, N, P)
    pi_fn      : callable — receives a length-1 array, returns float
    param_idx  : int — which column to use (default 0)
    param_name : str, optional
    x_range    : (float, float), optional
    n_grid     : int
    save_path  : str, optional — used only when ax is None
    ax         : matplotlib.axes.Axes, optional
        If provided the plot is drawn into this axes and no figure is
        created or closed here.

    Returns
    -------
    tvd : float
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    chains     = _ensure_3d(chains)
    samples_1d = chains[:, :, param_idx].ravel()

    if param_name is None:
        param_name = f"param[{param_idx}]"
    if x_range is None:
        mu, sd = samples_1d.mean(), samples_1d.std()
        x_range = (mu - 4 * sd, mu + 4 * sd)

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx   = grid[1] - grid[0]

    kde   = gaussian_kde(samples_1d, bw_method="scott")
    p_kde = kde(grid)

    p_true_raw = np.array([pi_fn(np.array([v])) for v in grid])
    p_true     = p_true_raw / (p_true_raw.sum() * dx)

    tvd = 0.5 * np.sum(np.abs(p_true - p_kde)) * dx

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(grid, p_true, color="steelblue", lw=2, label="True target")
    ax.plot(grid, p_kde,  color="tomato",    lw=2, label="Sampler KDE")
    ax.fill_between(grid, p_true, p_kde, alpha=0.15, color="gray")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.text(
        0.98, 0.95, f"TVD = {tvd:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
            print(f"    Saved {save_path}")
        else:
            plt.show()
        plt.close(fig)

    return tvd


# ------------------------------------------------------------------
# Per-scenario comparison figure  (all methods side-by-side)
# ------------------------------------------------------------------

def plot_comparison(method_chains, scenario, save_path=None):
    """
    Create one figure per scenario: columns = methods, rows = dimensions.

    Parameters
    ----------
    method_chains : dict
        Keys: "teleporting", "parallel_tempering", "vanilla"
        Values: ndarray (C, D, d)
    scenario : dict
        Must contain: label, d, x_range (list of d tuples).
        For d==1: pi_fn.
        For d >1: marginal_pi_fns (list of d callables).
    save_path : str, optional

    Returns
    -------
    tvds : dict  method -> list of TVD per dimension
    """
    import matplotlib.pyplot as plt

    d       = scenario["d"]
    methods = ["teleporting", "parallel_tempering", "vanilla"]
    labels  = ["Teleporting MCMC", "Parallel Tempering", "Vanilla NUTS"]

    fig, axes = plt.subplots(d, 3, figsize=(18, 4 * d), squeeze=False)

    tvds = {m: [] for m in methods}

    for col, (method, label) in enumerate(zip(methods, labels)):
        chains = method_chains[method]

        for row in range(d):
            ax = axes[row, col]

            if d == 1:
                pi_fn_1d  = scenario["pi_fn"]
                x_range   = scenario["x_range"][0]
                chains_1d = chains                        # (C, D, 1)
                dim_label = "x"
            else:
                pi_fn_1d  = scenario["marginal_pi_fns"][row]
                x_range   = scenario["x_range"][row]
                chains_1d = chains[:, :, row:row + 1]    # (C, D, 1)
                dim_label = f"x[{row}]"

            tvd = plot_against_truth(
                chains_1d, pi_fn_1d,
                x_range=x_range, param_name=dim_label, ax=ax,
            )
            tvds[method].append(tvd)

            title = label if row == 0 else ""
            if title:
                ax.set_title(f"{title}\nTVD = {tvd:.4f}", fontsize=10)
            else:
                ax.set_title(f"TVD = {tvd:.4f}", fontsize=10)

    fig.suptitle(scenario["label"], fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return tvds
