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
