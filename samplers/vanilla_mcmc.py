import numpy as np
import pymc as pm

class VanillaMCMC:
    """
    PyMC-based sampler for a Gaussian mixture target.

    Uses pm.NormalMixture with NUTS (PyMC default). This serves as the
    reference sampler to compare against TeleportingMCMC.

    Parameters
    ----------
    pi : array-like, shape (K,)
        Mixture weights (must sum to 1).
    mu : array-like, shape (K,)
        Component means.
    sigma2 : array-like, shape (K,)
        Component variances.
    seed : int, optional
        Random seed passed to pm.sample.
    """
    def __init__(self, pi, mu, sigma2, seed=None):
        self.pi = np.asarray(pi, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma2 = np.asarray(sigma2, dtype=float)
        self.seed = seed

    def run(self, num_draws=1000, num_chains=4, num_tune=500, progressbar=True):
        with pm.Model():
            if len(self.pi) == 1:
                pm.Normal("x", mu=self.mu[0], sigma=float(np.sqrt(self.sigma2[0])))
            else:
                pm.NormalMixture(
                    "x",
                    w=self.pi,
                    mu=self.mu,
                    sigma=np.sqrt(self.sigma2),
                )
            trace = pm.sample(
                draws=num_draws,
                tune=num_tune,
                chains=num_chains,
                random_seed=self.seed,
                progressbar=progressbar,
            )

        # posterior["x"]: shape (num_chains, num_draws) → add param axis
        samples = trace.posterior["x"].values[:, :, np.newaxis]  # (C, D, 1)

        return {
            "samples": samples,
            "trace": trace,
            "num_draws": num_draws,
            "num_chains": num_chains,
        }
