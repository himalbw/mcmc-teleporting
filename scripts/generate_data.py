import numpy as np
import pandas as pd

def sample_inverse_gamma(shape, scale, rng):
    gamma_draw = rng.gamma(shape=shape, scale=1.0 / scale)
    return 1.0 / gamma_draw

def generate_hierarchical_gaussian_mixture(
    n,
    K,
    alpha,
    a,
    b,
    c,
    d,
    m0,
    s0_sq,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    alpha = np.asarray(alpha, dtype=float)

    m = rng.normal(loc=m0, scale=np.sqrt(s0_sq))
    tau2 = sample_inverse_gamma(c, d, rng)

    pi = rng.dirichlet(alpha)

    mu = rng.normal(loc=m, scale=np.sqrt(tau2), size=K)
    sigma2 = np.array([sample_inverse_gamma(a, b, rng) for _ in range(K)])

    z = rng.choice(K, size=n, p=pi)

    y = np.zeros(n)
    for i in range(n):
        k = z[i]
        y[i] = rng.normal(loc=mu[k], scale=np.sqrt(sigma2[k]))

    return {
        "y": y,
        "z": z,
        "pi": pi,
        "mu": mu,
        "sigma2": sigma2,
        "m": m,
        "tau2": tau2,
    }

def main():
    rng = np.random.default_rng(221)

    out = generate_hierarchical_gaussian_mixture(
        n=5000,
        K=4,
        alpha=[1.0, 1.0, 1.0],
        a=3.0,
        b=2.0,
        c=3.0,
        d=4.0,
        m0=0.0,
        s0_sq=25.0,
        rng=rng
    )

    df = pd.DataFrame({
        "y": out["y"],
        "z": out["z"]
    })

    df.to_csv("data.csv", index=False)

    print("saved data.csv")
    print(df.head())

if __name__ == "__main__":
    main()


# ------------------------------------------------------------------
# Fixed scenarios for benchmarking samplers
# ------------------------------------------------------------------

def _make_gmm_pi_fn(pi, mu, sigma2):
    """Return a callable for the unnormalized 1-D GMM density."""
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    def pi_fn(x):
        x = np.asarray(x).ravel()
        val = 0.0
        for pk, mk, sk2 in zip(pi, mu, sigma2):
            val += pk * np.exp(-0.5 * (x[0] - mk) ** 2 / sk2) / np.sqrt(
                2.0 * np.pi * sk2
            )
        return float(val)

    return pi_fn


def _gmm_scenario(label, slug, pi, mu, sigma2, n, rng):
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    K = len(mu)
    z = rng.choice(K, size=n, p=pi)
    y = np.array([rng.normal(mu[k], np.sqrt(sigma2[k])) for k in z])
    pad = 4.0 * np.sqrt(sigma2.max())
    return {
        "label": label,
        "slug": slug,
        "pi": pi,
        "mu": mu,
        "sigma2": sigma2,
        "y": y,
        "z": z,
        "pi_fn": _make_gmm_pi_fn(pi, mu, sigma2),
        "x_range": (float(mu.min() - pad), float(mu.max() + pad)),
    }


def make_scenarios(rng, n=500):
    """
    Three 1-D GMM scenarios with increasing mode separation.

    close      : modes at -2, 0, 2  — partly overlapping
    separated  : modes at -6, 0, 6  — clearly separated
    far        : modes at -12, 12   — extreme bimodal gap
    """
    return [
        _gmm_scenario(
            label="close modes (sep=2)",
            slug="close",
            pi=[1 / 3, 1 / 3, 1 / 3],
            mu=[-2.0, 0.0, 2.0],
            sigma2=[0.3, 0.3, 0.3],
            n=n,
            rng=rng,
        ),
        _gmm_scenario(
            label="separated modes (sep=6)",
            slug="separated",
            pi=[0.4, 0.2, 0.4],
            mu=[-6.0, 0.0, 6.0],
            sigma2=[0.5, 0.5, 0.5],
            n=n,
            rng=rng,
        ),
        _gmm_scenario(
            label="far modes (sep=12)",
            slug="far",
            pi=[0.5, 0.5],
            mu=[-12.0, 12.0],
            sigma2=[1.0, 1.0],
            n=n,
            rng=rng,
        ),
    ]
