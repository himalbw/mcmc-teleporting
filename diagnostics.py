"""
MCMC diagnostics: R-hat, ESS, 95% CI, and summary table.

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
