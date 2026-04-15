import numpy as np


def gaussian_q_density(x, mean, sigma):
    x = np.asarray(x)
    mean = np.asarray(mean)
    d = x.size
    diff = x - mean
    norm_const = (2.0 * np.pi * sigma**2) ** (-d / 2.0)
    return norm_const * np.exp(-0.5 * np.dot(diff, diff) / (sigma**2))


def gaussian_q_sample(mean, sigma, rng):
    mean = np.asarray(mean)
    return mean + rng.normal(size=mean.shape) * sigma


def compute_Z(x, z, pi_fn, q_density_fn):
    N = len(x)
    total = 0.0
    for l in range(N):
        numerator = q_density_fn(x[l], z)
        for k in range(N):
            if k != l:
                numerator += q_density_fn(x[l], x[k])
        total += numerator / pi_fn(x[l])
    return total


def compute_importance_weights(x, z, pi_fn, q_density_fn):
    N = len(x)
    Z_val = compute_Z(x, z, pi_fn, q_density_fn)
    weights = np.zeros(N)
    for i in range(N):
        numerator = q_density_fn(x[i], z)
        for k in range(N):
            if k != i:
                numerator += q_density_fn(x[i], x[k])
        weights[i] = (numerator / pi_fn(x[i])) / Z_val
    weights = weights / weights.sum()
    return weights, Z_val


def run_teleporting_mcmc(x0, num_iter, pi_fn, q_sample_fn, q_density_fn, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.array(x0, dtype=float, copy=True)
    N = x.shape[0]  # number of walkers

    history = [x.copy()]
    mh_accepts = 0
    teleports_accepts = 0
    teleports_proposed = 0

    for t in range(num_iter):
        j = rng.integers(N)
        z = q_sample_fn(x[j], pi_fn, q_density_fn)
        weights, Z_forward = compute_importance_weights(x, z, pi_fn, q_density_fn)
        i = rng.choice(N, p=weights)  # select i according to importance weights
        if i != j:
            teleports_proposed += 1
        x_prop = x.copy()
        old_xi = x[i].copy()
        x_prop[i] = z
        Z_reverse = compute_Z(x_prop, old_xi, pi_fn, q_density_fn)
        alpha = min(1.0, Z_forward / Z_reverse)
        if rng.uniform() < alpha:
            x = x_prop
            mh_accepts += 1
            if i != j:
                teleports_accepts += 1
        history.append(x.copy())

    return {
        "samples": np.array(history),  # shape (num_iter + 1, N, d)
        "acceptance_rate": mh_accepts / num_iter,
        "teleport_proposal_rate": teleports_proposed / num_iter,
        "teleport_accept_rate": teleports_accepts / num_iter,
        "num_iter": num_iter,
        "num_walkers": N,
    }