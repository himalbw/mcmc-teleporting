import numpy as np


class ParallelTemperingMCMC:
    """
    Parallel tempering sampler for a target density.

    Runs one replica at each inverse temperature beta. Each replica performs a
    local random-walk Metropolis update targeting pi(x)^beta, followed by swap
    proposals between adjacent temperatures. The cold replica (beta=1) targets
    the true posterior and is the chain used for inference.

    Parameters
    ----------
    pi_fn : callable (x,) -> float
        Unnormalized target density.
    inverse_temperatures : array-like
        Sequence of inverse temperatures beta_1 > ... > beta_L with beta_1 = 1.
    proposal_scale : float, optional
        Standard deviation of the Gaussian random-walk proposal.
    rng : np.random.Generator, optional
    """

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
        """
        Run the sampler.

        Parameters
        ----------
        x0 : array-like, shape (L, d)
            Initial states for the L replicas.
        num_iter : int
            Number of MCMC iterations.
        swap_interval : int, optional
            How often to propose adjacent swaps.

        Returns
        -------
        dict with keys:
            samples                : ndarray (num_iter+1, L, d)
            cold_samples           : ndarray (num_iter+1, 1, d)
            inverse_temperatures   : ndarray (L,)
            local_acceptance_rates : ndarray (L,)
            swap_acceptance_rates  : ndarray (L-1,)
            num_iter               : int
            num_replicas           : int
        """
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
