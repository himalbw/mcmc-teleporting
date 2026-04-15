import numpy as np

class ParallelTemperingMCMC:
    def __init__(self, pi_fn, inverse_temperatures, proposal_scale=1.0, rng=None):
        self.pi_fn = pi_fn
        self.inverse_temperatures = self._validate_betas(inverse_temperatures)
        self.proposal_scale = float(proposal_scale)
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _validate_betas(inverse_temperatures):
        betas = np.asarray(inverse_temperatures, dtype=float)
        if betas.ndim != 1:
            raise ValueError("inverse_temperatures must be one-dimensional")
        if len(betas) < 2:
            raise ValueError("parallel tempering requires at least two replicas")
        if not np.isclose(betas[0], 1.0):
            raise ValueError("the cold chain must have inverse temperature beta=1")
        if np.any(betas <= 0):
            raise ValueError("inverse_temperatures must be strictly positive")
        if np.any(np.diff(betas) >= 0):
            raise ValueError("inverse_temperatures must be strictly decreasing")
        return betas

    # Converts the density function into a log density
    def _log_pi(self, x):
        density = float(self.pi_fn(x))
        if density <= 0.0:
            return -np.inf
        return np.log(density)

    def _local_update(self, x, beta):
        proposal = x + self.rng.normal(scale=self.proposal_scale, size=x.shape)
        log_pi_current = self._log_pi(x)
        log_pi_proposal = self._log_pi(proposal)
        log_alpha = beta * (log_pi_proposal - log_pi_current)

        if np.log(self.rng.uniform()) < min(0.0, log_alpha):
            return proposal, True
        return x, False

    def _attempt_swap(self, states, log_pis, level):
        beta_i = self.inverse_temperatures[level]
        beta_j = self.inverse_temperatures[level + 1]
        log_alpha = (beta_i - beta_j) * (log_pis[level + 1] - log_pis[level])

        if np.log(self.rng.uniform()) < min(0.0, log_alpha):
            states[level], states[level + 1] = states[level + 1], states[level]
            log_pis[level], log_pis[level + 1] = log_pis[level + 1], log_pis[level]
            return True
        return False

    def run(self, x0, num_iter, swap_interval=1):
        states = np.array(x0, dtype=float, copy=True)
        if states.ndim == 1:
            states = states[:, np.newaxis]

        num_replicas = states.shape[0]
        if num_replicas != len(self.inverse_temperatures):
            raise ValueError(
                "number of initial states must match number of temperatures"
            )
        if num_iter <= 0:
            raise ValueError("num_iter must be positive")
        if swap_interval <= 0:
            raise ValueError("swap_interval must be positive")

        history = [states.copy()]
        local_accepts = np.zeros(num_replicas, dtype=int)
        swap_attempts = np.zeros(num_replicas - 1, dtype=int)
        swap_accepts = np.zeros(num_replicas - 1, dtype=int)

        for iteration in range(1, num_iter + 1):
            for replica_idx, beta in enumerate(self.inverse_temperatures):
                updated_state, accepted = self._local_update(states[replica_idx], beta)
                states[replica_idx] = updated_state
                local_accepts[replica_idx] += int(accepted)

            log_pis = [self._log_pi(state) for state in states]

            if iteration % swap_interval == 0:
                offset = (iteration // swap_interval + 1) % 2
                for level in range(offset, num_replicas - 1, 2):
                    swap_attempts[level] += 1
                    if self._attempt_swap(states, log_pis, level):
                        swap_accepts[level] += 1

            history.append(states.copy())

        samples = np.array(history)
        swap_acceptance_rates = np.zeros_like(swap_attempts, dtype=float)
        attempted = swap_attempts > 0
        swap_acceptance_rates[attempted] = (
            swap_accepts[attempted] / swap_attempts[attempted]
        )

        return {
            "samples": samples,                          # (T+1, L, d)
            "cold_samples": samples[:, [0], :],         # (T+1, 1, d)
            "inverse_temperatures": self.inverse_temperatures.copy(),
            "local_acceptance_rates": local_accepts / num_iter,
            "swap_acceptance_rates": swap_acceptance_rates,
            "num_iter": num_iter,
            "num_replicas": num_replicas,
        }


# ------------------------------------------------------------------
# Temperature schedule grid search
# ------------------------------------------------------------------

def _ess_1d(x):
    """Bulk ESS for a 1-D chain via FFT autocorrelation (Geyer estimator)."""
    n = len(x)
    x = x - x.mean()
    fft_len = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_len)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0:
        return float(n)
    acf /= acf[0]
    tau = 1.0
    for t in range(1, n // 2):
        pair = acf[2 * t - 1] + acf[2 * t]
        if pair < 0:
            break
        tau += 2 * pair
    return n / tau


def grid_search_temperatures(
    pi_fn,
    x0_single,
    num_replicas_grid=(3, 4, 5),
    beta_min_grid=(0.01, 0.05, 0.1, 0.2, 0.5),
    num_iter=400,
    swap_interval=1,
    proposal_scale=1.0,
    rng=None,
    verbose=True,
):
    """
    Grid search over geometric temperature ladders for parallel tempering.

    For each (num_replicas, beta_min) pair a geometric ladder is constructed:
        betas = [1.0, beta_min^(1/(L-1)), ..., beta_min]
    and PT is run for `num_iter` steps. Schedules are ranked by the ESS of
    the cold chain (higher = better mixing).

    Parameters
    ----------
    pi_fn : callable
        Unnormalized target density.
    x0_single : array-like, shape (d,) or (1, d)
        Starting point; all replicas are initialised here.
    num_replicas_grid : sequence of int
        Ladder lengths to try (must be >= 2).
    beta_min_grid : sequence of float
        Minimum inverse temperature (hottest chain) values to try (in (0, 1)).
    num_iter : int
        Iterations per candidate evaluation (keep short for speed).
    swap_interval : int
        Swap attempt frequency passed to PT.run().
    proposal_scale : float
        Local proposal std passed to each PT instance.
    rng : np.random.Generator, optional
    verbose : bool
        Print each candidate result as it finishes.

    Returns
    -------
    best_betas : ndarray
        Inverse temperature ladder with the highest cold-chain ESS.
    results : list of dict
        All candidates sorted best-first, each with keys:
            num_replicas, beta_min, betas, ess, avg_swap_rate,
            local_acceptance_rates.
    """
    rng = rng if rng is not None else np.random.default_rng()
    x0_single = np.atleast_2d(np.asarray(x0_single, dtype=float))  # (1, d)

    if verbose:
        print(f"  {'L':>4}  {'beta_min':>9}  {'ESS':>8}  {'swap_rate':>10}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*10}")

    results = []
    for n_rep in num_replicas_grid:
        for beta_min in beta_min_grid:
            betas = np.geomspace(1.0, beta_min, n_rep)
            x0 = np.tile(x0_single, (n_rep, 1))

            sampler = ParallelTemperingMCMC(
                pi_fn=pi_fn,
                inverse_temperatures=betas,
                proposal_scale=proposal_scale,
                rng=np.random.default_rng(int(rng.integers(2**31))),
            )
            result = sampler.run(x0, num_iter=num_iter, swap_interval=swap_interval)

            # ESS of cold chain  (shape T+1, 1, d → 1-D array)
            cold = result["cold_samples"][:, 0, 0]  # (T+1,)
            ess_val = _ess_1d(cold)
            avg_swap = float(result["swap_acceptance_rates"].mean())

            entry = {
                "num_replicas": n_rep,
                "beta_min": beta_min,
                "betas": betas,
                "ess": ess_val,
                "avg_swap_rate": avg_swap,
                "local_acceptance_rates": result["local_acceptance_rates"],
            }
            results.append(entry)

            if verbose:
                print(
                    f"  {n_rep:>4}  {beta_min:>9.3f}  {ess_val:>8.1f}  {avg_swap:>10.3f}"
                )

    results.sort(key=lambda r: r["ess"], reverse=True)
    best = results[0]

    if verbose:
        print(
            f"\n  Best: L={best['num_replicas']}, beta_min={best['beta_min']:.3f}, "
            f"ESS={best['ess']:.1f}, avg_swap={best['avg_swap_rate']:.3f}"
        )

    return best["betas"], results
