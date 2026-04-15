"""
MCMC diagnostics: R-hat, ESS, 95% CI, summary table, and density plot vs truth.

All functions expect chains of shape (num_chains, num_draws, num_params)
or (num_chains, num_draws) for a single parameter.
"""

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
    chains : array-like, shape (M, N, P) — M chains, N draws, P params

    Returns
    -------
    ndarray, shape (P,)
    """
    chains = _ensure_3d(chains)
    M, N, P = chains.shape

    # Split each chain in half → 2M chains of length N//2
    n = N // 2
    split = np.concatenate([chains[:, :n, :], chains[:, n:2*n, :]], axis=0)
    M2 = split.shape[0]  # 2M

    chain_means = split.mean(axis=1)          # (2M, P)
    grand_mean = chain_means.mean(axis=0)     # (P,)

    B = n / (M2 - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(split.var(axis=1, ddof=1), axis=0)

    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / np.where(W > 0, W, np.nan))


# ------------------------------------------------------------------
# ESS  (bulk, via FFT autocorrelation)
# ------------------------------------------------------------------

def _autocorr(x):
    """Normalised autocorrelation of 1-D array x via FFT."""
    n = len(x)
    x = x - x.mean()
    # Zero-pad to next power of 2 for efficiency
    fft_len = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_len)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    return acf


def ess(chains):
    """
    Estimate bulk effective sample size for each parameter.

    Uses the Geyer initial monotone sequence estimator (truncated sum of
    positive paired autocorrelations).

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
        rho /= M  # average autocorrelation across chains

        # Geyer's initial positive sequence: sum pairs until pair goes negative
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
    """
    2.5th and 97.5th percentiles across all chains and draws.

    Parameters
    ----------
    chains : array-like, shape (M, N, P)

    Returns
    -------
    lower : ndarray (P,)
    upper : ndarray (P,)
    """
    chains = _ensure_3d(chains)
    flat = chains.reshape(-1, chains.shape[-1])
    lower = np.percentile(flat, 2.5, axis=0)
    upper = np.percentile(flat, 97.5, axis=0)
    return lower, upper


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
    """
    Print a diagnostics summary table.

    Parameters
    ----------
    chains : array-like, shape (M, N, P)
    param_names : list of str, optional
    """
    chains = _ensure_3d(chains)
    P = chains.shape[2]

    if param_names is None:
        param_names = [f"param[{i}]" for i in range(P)]

    mean  = posterior_mean(chains)
    std   = posterior_std(chains)
    lo, hi = ci_95(chains)
    rhat  = r_hat(chains)
    n_eff = ess(chains)

    col_w = max(max(len(n) for n in param_names), 10)
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
# Density plot vs truth + TVD estimate
# ------------------------------------------------------------------

def plot_against_truth(
    chains,
    pi_fn,
    param_idx=0,
    param_name=None,
    x_range=None,
    n_grid=500,
    save_path=None,
):
    """
    Plot the sampler's marginal density for one parameter against the true
    target density, and annotate with an estimated Total Variation Distance.

    TVD is estimated numerically on the same grid:
        TVD = 0.5 * integral |p_true(x) - p_kde(x)| dx
            ≈ 0.5 * sum |p - q| * dx

    Parameters
    ----------
    chains : array-like, shape (M, N, P)
    pi_fn : callable (array,) -> float
        Unnormalized 1-D target density. Receives a length-1 array.
    param_idx : int
        Which parameter column to plot (default 0).
    param_name : str, optional
        Label for the x-axis.
    x_range : (float, float), optional
        Grid range. Defaults to [mean ± 4*std] of the samples.
    n_grid : int
        Number of grid points for numerical integration and true density.
    save_path : str or Path, optional
        If given, save the figure there instead of showing it.

    Returns
    -------
    tvd : float
        Estimated Total Variation Distance.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    chains = _ensure_3d(chains)
    samples_1d = chains[:, :, param_idx].ravel()

    if param_name is None:
        param_name = f"param[{param_idx}]"

    if x_range is None:
        mu, sd = samples_1d.mean(), samples_1d.std()
        x_range = (mu - 4 * sd, mu + 4 * sd)

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx = grid[1] - grid[0]

    # KDE of samples
    kde = gaussian_kde(samples_1d, bw_method="scott")
    p_kde = kde(grid)

    # True density on grid, normalised to integrate to 1 over the grid
    p_true_raw = np.array([pi_fn(np.array([v])) for v in grid])
    p_true = p_true_raw / (p_true_raw.sum() * dx)

    # TVD
    tvd = 0.5 * np.sum(np.abs(p_true - p_kde)) * dx

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, p_true, color="steelblue", lw=2, label="True target")
    ax.plot(grid, p_kde,  color="tomato",    lw=2, label="Sampler KDE")
    ax.fill_between(grid, p_true, p_kde, alpha=0.15, color="gray")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Density")
    ax.set_title("True vs sampled density")
    ax.legend()
    ax.text(
        0.98, 0.95, f"TVD = {tvd:.4f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return tvd
