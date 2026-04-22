import numpy as np
import pymc as pm


class VanillaMCMC:
    """
    PyMC-based sampler for Gaussian mixture and multivariate Gaussian targets.

    Parameters
    ----------
    vanilla_type : str
        "mixture_1d" — 1-D Gaussian mixture (pm.NormalMixture / pm.Normal).
        "mvnormal"   — d-dimensional Gaussian (pm.MvNormal).
    pi, mu, sigma2 : for "mixture_1d"
    mu_vec, cov    : for "mvnormal"
    seed : int, optional
    """

    def __init__(self, vanilla_type, pi=None, mu=None, sigma2=None,
                 mu_vec=None, cov=None, nd_d=None, seed=None):
        self.vanilla_type = vanilla_type
        self.pi     = np.asarray(pi,     dtype=float) if pi     is not None else None
        self.mu     = np.asarray(mu,     dtype=float) if mu     is not None else None
        self.sigma2 = np.asarray(sigma2, dtype=float) if sigma2 is not None else None
        self.mu_vec = np.asarray(mu_vec, dtype=float) if mu_vec is not None else None
        self.cov    = np.asarray(cov,    dtype=float) if cov    is not None else None
        self.nd_d   = nd_d
        self.seed   = seed

    def run(self, num_draws=1000, num_chains=4, num_tune=500, progressbar=True):
        initvals = None
        with pm.Model():
            if self.vanilla_type == "mixture_1d":
                if len(self.pi) == 1:
                    pm.Normal("x", mu=self.mu[0],
                              sigma=float(np.sqrt(self.sigma2[0])))
                else:
                    pm.NormalMixture("x", w=self.pi, mu=self.mu,
                                     sigma=np.sqrt(self.sigma2))
            elif self.vanilla_type == "mvnormal":
                pm.MvNormal("x", mu=self.mu_vec, cov=self.cov,
                            shape=len(self.mu_vec))
            elif self.vanilla_type == "mixture_nd":
                import pytensor.tensor as pt
                d = self.nd_d
                x = pm.Flat("x", shape=d)
                log05 = float(np.log(0.5))
                log_pi = pt.sum([
                    pt.logaddexp(
                        pm.logp(pm.Normal.dist(mu=-5.0, sigma=1.0), x[i]) + log05,
                        pm.logp(pm.Normal.dist(mu=5.0,  sigma=1.0), x[i]) + log05,
                    )
                    for i in range(d)
                ])
                pm.Potential("log_target", log_pi)
                initvals = {"x": np.full(d, 5.0)}
            else:
                raise ValueError(f"Unknown vanilla_type: {self.vanilla_type!r}")

            trace = pm.sample(
                draws=num_draws,
                tune=num_tune,
                chains=num_chains,
                random_seed=self.seed,
                progressbar=progressbar,
                initvals=initvals,
            )

        # posterior["x"]: (chains, draws) for 1-D  →  expand to (C, D, 1)
        #                  (chains, draws, d) for d-D  →  keep as (C, D, d)
        samples = trace.posterior["x"].values
        if samples.ndim == 2:
            samples = samples[:, :, np.newaxis]

        return {
            "samples":   samples,
            "trace":     trace,
            "num_draws": num_draws,
            "num_chains": num_chains,
        }
