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

    def _log_pi(self, x):
        density = float(self.pi_fn(x))
        if density <= 0.0:
            return -np.inf
        return np.log(density)

    def _local_update(self, x, beta):
        proposal = x + self.rng.normal(scale=self.proposal_scale, size=x.shape)
        log_pi_current  = self._log_pi(x)
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
            states[level],   states[level + 1]   = states[level + 1],   states[level]
            log_pis[level],  log_pis[level + 1]  = log_pis[level + 1],  log_pis[level]
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

        history        = [states.copy()]
        local_accepts  = np.zeros(num_replicas, dtype=int)
        swap_attempts  = np.zeros(num_replicas - 1, dtype=int)
        swap_accepts   = np.zeros(num_replicas - 1, dtype=int)

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
            "samples":               samples,             # (T+1, L, d)
            "cold_samples":          samples[:, [0], :],  # (T+1, 1, d)
            "inverse_temperatures":  self.inverse_temperatures.copy(),
            "local_acceptance_rates": local_accepts / num_iter,
            "swap_acceptance_rates": swap_acceptance_rates,
            "num_iter":              num_iter,
            "num_replicas":          num_replicas,
        }


# ------------------------------------------------------------------
# ESS helper
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


def _schedule_score(ess, swap_rates, target=0.25):
    """
    Multi-criterion score for a temperature schedule.

    ESS is multiplied by a quality factor that rewards average swap rates
    close to `target` (optimal ~ 0.23–0.25 for PT).  The quality factor is a
    Gaussian bump centred on `target` with width 0.18, so schedules far from
    the sweet spot are penalised even if their ESS looks high.
    """
    if ess <= 0 or len(swap_rates) == 0:
        return -np.inf
    avg = float(swap_rates.mean())
    quality = np.exp(-0.5 * ((avg - target) / 0.18) ** 2)
    return ess * quality


# ------------------------------------------------------------------
# Adaptive temperature refinement
# ------------------------------------------------------------------

def optimize_temperatures(
    pi_fn,
    x0_single,
    betas_init,
    num_rounds=4,
    num_iter_per_round=400,
    target_rate=0.25,
    proposal_scale=1.0,
    rng=None,
    verbose=False,
):
    """
    Adaptively refine a temperature ladder to equalise swap acceptance rates
    near `target_rate`.

    Algorithm (per round):
      1. Run a short PT chain with the current β ladder.
      2. Compute the empirical swap rate r_i for each adjacent pair.
      3. In log-β space, rescale each gap by (log(target)/log(r))^0.5 —
         wide gaps (low r) are shrunk; narrow gaps (high r) are grown.
      4. Renormalise so that β_min is preserved; reconstruct the ladder.

    This keeps the number of replicas fixed and drives each swap rate toward
    the target without changing the span [β_min, 1].

    Parameters
    ----------
    pi_fn           : callable — unnormalized target density
    x0_single       : array-like, shape (d,) or (1, d)
    betas_init      : array-like — initial ladder, betas_init[0] must be 1.0
    num_rounds      : int — number of adaptation rounds
    num_iter_per_round : int — PT iterations per round
    target_rate     : float in (0, 1) — desired swap acceptance rate
    proposal_scale  : float
    rng             : np.random.Generator, optional
    verbose         : bool

    Returns
    -------
    betas      : ndarray — refined ladder
    swap_rates : ndarray — swap rates from the last round
    """
    rng       = rng if rng is not None else np.random.default_rng()
    betas     = np.asarray(betas_init, dtype=float).copy()
    x0_single = np.atleast_2d(np.asarray(x0_single, dtype=float))

    swap_rates = np.zeros(len(betas) - 1)

    if verbose:
        print(f"    {'round':>6}  {'betas':>40}  {'swap_rates'}")

    for round_idx in range(num_rounds):
        x0 = np.tile(x0_single, (len(betas), 1))
        sampler = ParallelTemperingMCMC(
            pi_fn, betas, proposal_scale,
            rng=np.random.default_rng(int(rng.integers(2 ** 31))),
        )
        result     = sampler.run(x0, num_iter=num_iter_per_round)
        swap_rates = result["swap_acceptance_rates"]

        if verbose:
            rates_str = " ".join(f"{r:.2f}" for r in swap_rates)
            betas_str = " ".join(f"{b:.3f}" for b in betas)
            print(f"    {round_idx:>6}  {betas_str:>40}  [{rates_str}]")

        # --- rescale gaps in log-β space ---
        log_betas = np.log(np.maximum(betas, 1e-12))
        gaps      = np.diff(log_betas)          # (L-1,) all ≤ 0

        for i, r in enumerate(swap_rates):
            if 1e-4 < r < 1.0 - 1e-4:
                factor = (np.log(target_rate) / np.log(r)) ** 0.5
                factor = np.clip(factor, 0.3, 3.0)
                gaps[i] *= factor

        # Renormalise to keep β_min fixed
        total = gaps.sum()
        if total != 0.0:
            gaps *= log_betas[-1] / total

        # Reconstruct and validate
        new_log_betas = np.concatenate([[0.0], np.cumsum(gaps)])
        candidate     = np.exp(new_log_betas)
        candidate[0]  = 1.0
        candidate     = np.clip(candidate, 1e-8, 1.0)

        if np.all(np.diff(candidate) < 0):
            betas = candidate

    return betas, swap_rates


# ------------------------------------------------------------------
# Grid search over temperature schedules
# ------------------------------------------------------------------

def grid_search_temperatures(
    pi_fn,
    x0_single,
    num_replicas_grid=(3, 4, 5, 6, 7),
    beta_min_grid=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
    num_iter=400,
    swap_interval=1,
    proposal_scale=1.0,
    rng=None,
    verbose=True,
):
    """
    Grid search over geometric temperature ladders for parallel tempering.

    For each (num_replicas, beta_min) pair a geometric ladder is constructed:
        betas = geomspace(1.0, beta_min, num_replicas)
    and PT is run for `num_iter` steps.  Schedules are ranked by a
    multi-criterion score:  ESS × quality(avg_swap_rate), where quality is a
    Gaussian bump centred on 0.25 — the empirically optimal swap-acceptance
    rate for parallel tempering (Predescu et al. 2004).

    Parameters
    ----------
    pi_fn              : callable — unnormalized target density
    x0_single          : array-like, shape (d,) or (1, d)
    num_replicas_grid  : sequence of int (ladder lengths, must be ≥ 2)
    beta_min_grid      : sequence of float (hottest-chain β, in (0, 1))
    num_iter           : int — iterations per candidate (short for speed)
    swap_interval      : int
    proposal_scale     : float
    rng                : np.random.Generator, optional
    verbose            : bool

    Returns
    -------
    best_betas : ndarray — ladder with the highest score
    results    : list of dict, sorted best-first, keys:
                   num_replicas, beta_min, betas, ess, avg_swap_rate,
                   score, local_acceptance_rates
    """
    rng       = rng if rng is not None else np.random.default_rng()
    x0_single = np.atleast_2d(np.asarray(x0_single, dtype=float))

    if verbose:
        print(f"  {'L':>4}  {'beta_min':>9}  {'ESS':>8}  {'swap_avg':>9}  {'score':>10}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*10}")

    results = []
    for n_rep in num_replicas_grid:
        for beta_min in beta_min_grid:
            betas = np.geomspace(1.0, beta_min, n_rep)
            x0    = np.tile(x0_single, (n_rep, 1))

            sampler = ParallelTemperingMCMC(
                pi_fn=pi_fn,
                inverse_temperatures=betas,
                proposal_scale=proposal_scale,
                rng=np.random.default_rng(int(rng.integers(2 ** 31))),
            )
            result   = sampler.run(x0, num_iter=num_iter,
                                   swap_interval=swap_interval)

            cold     = result["cold_samples"][:, 0, 0]
            ess_val  = _ess_1d(cold)
            avg_swap = float(result["swap_acceptance_rates"].mean())
            score    = _schedule_score(ess_val, result["swap_acceptance_rates"])

            entry = dict(
                num_replicas          = n_rep,
                beta_min              = beta_min,
                betas                 = betas,
                ess                   = ess_val,
                avg_swap_rate         = avg_swap,
                score                 = score,
                local_acceptance_rates = result["local_acceptance_rates"],
            )
            results.append(entry)

            if verbose:
                print(
                    f"  {n_rep:>4}  {beta_min:>9.4f}  {ess_val:>8.1f}"
                    f"  {avg_swap:>9.3f}  {score:>10.2f}"
                )

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]

    if verbose:
        print(
            f"\n  Best: L={best['num_replicas']}, beta_min={best['beta_min']:.4f}, "
            f"ESS={best['ess']:.1f}, avg_swap={best['avg_swap_rate']:.3f}, "
            f"score={best['score']:.2f}"
        )

    return best["betas"], results
