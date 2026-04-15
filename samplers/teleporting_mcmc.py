import numpy as np


class TeleportingMCMC:
    """
    Teleporting MCMC sampler (Skene et al. 2023).

    Maintains N walkers. At each step, a proposal z is drawn near walker j,
    then walker i is selected via importance weights and swapped to z via an
    MH accept/reject step.  When i != j this is a "teleport".

    Parameters
    ----------
    pi_fn : callable (x,) -> float
        Unnormalized target density.
    q_sample_fn : callable (x, rng) -> z
        Draw a proposal near x.
    q_density_fn : callable (x, mean) -> float
        Evaluate proposal density q(x | mean).
    rng : np.random.Generator, optional
    """

    def __init__(self, pi_fn, q_sample_fn, q_density_fn, rng=None):
        self.pi_fn = pi_fn
        self.q_sample_fn = q_sample_fn
        self.q_density_fn = q_density_fn
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_Z(self, x, z):
        """Normalising constant Z(x, z) for the MH ratio."""
        N = len(x)
        total = 0.0
        for l in range(N):
            numerator = self.q_density_fn(x[l], z)
            for k in range(N):
                if k != l:
                    numerator += self.q_density_fn(x[l], x[k])
            total += numerator / self.pi_fn(x[l])
        return total

    def _compute_importance_weights(self, x, z):
        """Importance weights w_i(x, z) and Z(x, z)."""
        N = len(x)
        Z_val = self._compute_Z(x, z)
        weights = np.zeros(N)
        for i in range(N):
            numerator = self.q_density_fn(x[i], z)
            for k in range(N):
                if k != i:
                    numerator += self.q_density_fn(x[i], x[k])
            weights[i] = (numerator / self.pi_fn(x[i])) / Z_val
        weights /= weights.sum()
        return weights, Z_val

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, x0, num_iter):
        """
        Run the sampler.

        Parameters
        ----------
        x0 : array-like, shape (N, d)
            Initial positions of N walkers in d dimensions.
        num_iter : int
            Number of MCMC iterations.

        Returns
        -------
        dict with keys:
            samples              : ndarray (num_iter+1, N, d)
            acceptance_rate      : float
            teleport_proposal_rate : float
            teleport_accept_rate : float
            num_iter             : int
            num_walkers          : int
        """
        x = np.array(x0, dtype=float, copy=True)
        N = x.shape[0]

        history = [x.copy()]
        mh_accepts = 0
        teleports_proposed = 0
        teleports_accepted = 0

        for _ in range(num_iter):
            j = self.rng.integers(N)
            z = self.q_sample_fn(x[j], self.rng)

            weights, Z_forward = self._compute_importance_weights(x, z)
            i = self.rng.choice(N, p=weights)

            if i != j:
                teleports_proposed += 1

            x_prop = x.copy()
            old_xi = x[i].copy()
            x_prop[i] = z

            Z_reverse = self._compute_Z(x_prop, old_xi)
            alpha = min(1.0, Z_forward / Z_reverse)

            if self.rng.uniform() < alpha:
                x = x_prop
                mh_accepts += 1
                if i != j:
                    teleports_accepted += 1

            history.append(x.copy())

        return {
            "samples": np.array(history),          # (num_iter+1, N, d)
            "acceptance_rate": mh_accepts / num_iter,
            "teleport_proposal_rate": teleports_proposed / num_iter,
            "teleport_accept_rate": teleports_accepted / num_iter,
            "num_iter": num_iter,
            "num_walkers": N,
        }


# ------------------------------------------------------------------
# Default Gaussian proposal (convenience)
# ------------------------------------------------------------------

def gaussian_q_density(x, mean, sigma):
    x, mean = np.asarray(x), np.asarray(mean)
    d = x.size
    diff = x - mean
    return (2.0 * np.pi * sigma**2) ** (-d / 2.0) * np.exp(
        -0.5 * np.dot(diff, diff) / sigma**2
    )


def gaussian_q_sample(mean, sigma, rng):
    mean = np.asarray(mean)
    return mean + rng.normal(size=mean.shape) * sigma
