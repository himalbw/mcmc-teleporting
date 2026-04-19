import numpy as np
from scipy.stats import multivariate_normal as _mvn_dist, norm as _norm_dist


# ------------------------------------------------------------------
# Density factories
# ------------------------------------------------------------------

def _make_gmm_pi_fn_1d(pi, mu, sigma2):
    """Unnormalised 1-D Gaussian mixture density."""
    pi, mu, sigma2 = (np.asarray(a, dtype=float) for a in (pi, mu, sigma2))

    def pi_fn(x):
        x_val = float(np.asarray(x).ravel()[0])
        return float(sum(
            pk * np.exp(-0.5 * (x_val - mk) ** 2 / sk2) / np.sqrt(2.0 * np.pi * sk2)
            for pk, mk, sk2 in zip(pi, mu, sigma2)
        ))

    return pi_fn


def _make_mvn_pi_fn(mean, cov):
    """d-dimensional multivariate Gaussian density."""
    rv = _mvn_dist(mean=mean, cov=cov)

    def pi_fn(x):
        return float(rv.pdf(np.asarray(x).ravel()))

    return pi_fn


def _make_norm_marginal_pi_fn(mean, var):
    """1-D marginal density for diagnostics of a 2-D scenario."""
    rv = _norm_dist(loc=mean, scale=np.sqrt(var))

    def pi_fn(x):
        return float(rv.pdf(float(np.asarray(x).ravel()[0])))

    return pi_fn


# ------------------------------------------------------------------
# Benchmark scenarios
# ------------------------------------------------------------------

def make_scenarios(rng):
    """
    Six benchmark target distributions matching the project slide:

    1. standard          — N(0, 1)                                    [1-D]
    2. correlated        — N(0, Σ), ρ=0.9, σ₁=3, σ₂=1               [2-D]
    3. bimodal_moderate  — ½N(−5,1) + ½N(5,1)                        [1-D]
    4. bimodal_large     — ½N(−15,1) + ½N(15,1)                      [1-D]
    5. unequal_weight    — 0.9N(−5,1) + 0.1N(5,1)                    [1-D]
    6. different_scale   — ½N(−5, 0.25) + ½N(5, 4)                   [1-D]

    Each entry contains:
      label, slug, d, pi_fn, x_range (list of d tuples), proposal_sigma,
      vanilla_type ("mixture_1d" | "mvnormal"),
      and for "mixture_1d": pi, mu, sigma2
      and for "mvnormal":   mu_vec, cov, marginal_pi_fns
    """
    # Correlated Gaussian covariance: ρ=0.9, σ₁=3, σ₂=1
    cov = np.array([[9.0, 2.7],
                    [2.7, 1.0]])

    return [
        # ---- 1. Standard Gaussian ----
        dict(
            label        = "Standard Gaussian",
            slug         = "standard",
            d            = 1,
            pi_fn        = _make_gmm_pi_fn_1d([1.0], [0.0], [1.0]),
            pi           = np.array([1.0]),
            mu           = np.array([0.0]),
            sigma2       = np.array([1.0]),
            x_range      = [(-4.0, 4.0)],
            proposal_sigma = 1.5,
            vanilla_type = "mixture_1d",
        ),

        # ---- 2. Correlated Gaussian (2-D) ----
        dict(
            label        = "Correlated Gaussian (ρ=0.9)",
            slug         = "correlated",
            d            = 2,
            pi_fn        = _make_mvn_pi_fn(np.zeros(2), cov),
            mu_vec       = np.zeros(2),
            cov          = cov,
            x_range      = [(-9.0, 9.0), (-3.0, 3.0)],
            marginal_pi_fns = [
                _make_norm_marginal_pi_fn(0.0, 9.0),   # x[0] ~ N(0, 9)
                _make_norm_marginal_pi_fn(0.0, 1.0),   # x[1] ~ N(0, 1)
            ],
            proposal_sigma = 4.5,   # 1.5 × σ_max = 1.5 × 3
            vanilla_type   = "mvnormal",
            use_hessian_q  = True,  # isotropic q fails here; use Laplace reference
        ),

        # ---- 3. Bimodal — moderate separation ----
        dict(
            label        = "Bimodal — moderate (μ=±5)",
            slug         = "bimodal_moderate",
            d            = 1,
            pi_fn        = _make_gmm_pi_fn_1d([0.5, 0.5], [-5.0, 5.0], [1.0, 1.0]),
            pi           = np.array([0.5, 0.5]),
            mu           = np.array([-5.0, 5.0]),
            sigma2       = np.array([1.0, 1.0]),
            x_range      = [(-10.0, 10.0)],
            proposal_sigma = 1.5,
            vanilla_type = "mixture_1d",
        ),

        # ---- 4. Bimodal — large separation ----
        dict(
            label        = "Bimodal — large (μ=±15)",
            slug         = "bimodal_large",
            d            = 1,
            pi_fn        = _make_gmm_pi_fn_1d([0.5, 0.5], [-15.0, 15.0], [1.0, 1.0]),
            pi           = np.array([0.5, 0.5]),
            mu           = np.array([-15.0, 15.0]),
            sigma2       = np.array([1.0, 1.0]),
            x_range      = [(-20.0, 20.0)],
            proposal_sigma = 1.5,
            vanilla_type = "mixture_1d",
        ),

        # ---- 5. Unequal-weight bimodal ----
        dict(
            label        = "Unequal-weight bimodal (0.9 / 0.1)",
            slug         = "unequal_weight",
            d            = 1,
            pi_fn        = _make_gmm_pi_fn_1d([0.9, 0.1], [-5.0, 5.0], [1.0, 1.0]),
            pi           = np.array([0.9, 0.1]),
            mu           = np.array([-5.0, 5.0]),
            sigma2       = np.array([1.0, 1.0]),
            x_range      = [(-10.0, 10.0)],
            proposal_sigma = 1.5,
            vanilla_type = "mixture_1d",
        ),

        # ---- 6. Different-scale bimodal ----
        dict(
            label        = "Different-scale bimodal (σ=0.5 vs σ=2)",
            slug         = "different_scale",
            d            = 1,
            pi_fn        = _make_gmm_pi_fn_1d([0.5, 0.5], [-5.0, 5.0], [0.25, 4.0]),
            pi           = np.array([0.5, 0.5]),
            mu           = np.array([-5.0, 5.0]),
            sigma2       = np.array([0.25, 4.0]),
            x_range      = [(-10.0, 10.0)],
            proposal_sigma = 3.0,   # 1.5 × σ_max = 1.5 × 2
            vanilla_type = "mixture_1d",
        ),
    ]
