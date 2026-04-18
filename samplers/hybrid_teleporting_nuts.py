"""
Hybrid Teleporting-NUTS sampler.

At each step the standard teleporting importance-weight mechanism decides
whether the move is a teleport or a local update:

  i ≠ j  →  teleport: standard Teleporting-MCMC acceptance (Z_fwd / Z_rev)
  i == j →  local:    one NUTS trajectory from x_j  (replaces Gaussian MH)

The NUTS step size is adapted during the warm-up phase using dual averaging
(Hoffman & Gelman 2014, §3.2 / Algorithm 4), targeting a 65 % acceptance rate.
"""

import numpy as np

_DELTA_MAX = 1000.0   # divergence guard: stop building tree if ΔH > this


# ------------------------------------------------------------------
# Gradient  (central finite differences on log π)
# ------------------------------------------------------------------

def _log_pi(pi_fn, x):
    v = float(pi_fn(x))
    return np.log(v) if v > 0.0 else -np.inf


def _grad_log_pi(pi_fn, x, eps=1e-5):
    """Central-difference gradient of log π at x (shape (d,))."""
    x  = np.asarray(x, dtype=float)
    d  = x.size
    g  = np.zeros(d)
    lp = _log_pi(pi_fn, x)
    if not np.isfinite(lp):
        return g
    for k in range(d):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        lp_p = _log_pi(pi_fn, xp)
        lp_m = _log_pi(pi_fn, xm)
        if np.isfinite(lp_p) and np.isfinite(lp_m):
            g[k] = (lp_p - lp_m) / (2.0 * eps)
        elif np.isfinite(lp_p):
            g[k] = (lp_p - lp) / eps
        elif np.isfinite(lp_m):
            g[k] = (lp - lp_m) / eps
    return g


# ------------------------------------------------------------------
# Leapfrog
# ------------------------------------------------------------------

def _leapfrog(q, p, grad_fn, step_size, n_steps):
    """n_steps leapfrog steps; returns (q_new, p_new)."""
    q, p = q.copy(), p.copy()
    p   += 0.5 * step_size * grad_fn(q)
    for _ in range(n_steps - 1):
        q += step_size * p
        p += step_size * grad_fn(q)
    q   += step_size * p
    p   += 0.5 * step_size * grad_fn(q)
    return q, p


# ------------------------------------------------------------------
# NUTS tree  (Hoffman & Gelman 2014, Algorithm 3)
# ------------------------------------------------------------------

def _hamiltonian(log_pi_fn, q, p):
    lp = log_pi_fn(q)
    return (-lp + 0.5 * float(np.dot(p, p))) if np.isfinite(lp) else np.inf


def _build_tree(q, p, log_u, v, depth, step_size, log_pi_fn, grad_fn, H0, rng):
    """
    Recursively build a NUTS binary subtree.

    Parameters
    ----------
    q, p      : current position / momentum (start of subtree)
    log_u     : log of the slice threshold  (= log U - H0, U ~ Uniform[0,1])
    v         : direction (+1 or -1)
    depth     : tree depth (0 = one leapfrog step)
    H0        : Hamiltonian at the original start of the trajectory (for alpha)

    Returns
    -------
    q_minus, p_minus : leftmost leaf
    q_plus,  p_plus  : rightmost leaf
    q_sample         : candidate next position
    n                : number of valid (in-slice) leaf nodes
    s                : 1 = no U-turn, 0 = stop
    alpha_sum        : sum of min(1, exp(H0 - H_leaf)) over all base cases
    n_alpha          : number of base cases (for averaging)
    """
    if depth == 0:
        # One leapfrog step
        q_new, p_new = _leapfrog(q, p, grad_fn, v * step_size, 1)
        H_new        = _hamiltonian(log_pi_fn, q_new, p_new)

        # Slice condition: accept leaf if exp(-H_new) >= exp(log_u), i.e., -H_new >= log_u
        n     = 1 if (np.isfinite(H_new) and -H_new >= log_u) else 0
        # Early-stop: divergence guard
        s     = 1 if (np.isfinite(H_new) and -H_new > log_u - _DELTA_MAX) else 0
        # Step-size adaptation: acceptance probability for this leap
        alpha = min(1.0, float(np.exp(H0 - H_new))) if np.isfinite(H_new) else 0.0

        return q_new, p_new, q_new, p_new, q_new, n, s, alpha, 1

    # ---- Recursive case ----
    (q_m, p_m, q_p, p_p,
     q_s, n_s, s_s,
     alpha_s, n_alpha_s) = _build_tree(
        q, p, log_u, v, depth - 1, step_size, log_pi_fn, grad_fn, H0, rng
    )

    if s_s:
        if v == -1:
            (q_m, p_m, _, _,
             q_s2, n_s2, s_s2,
             alpha_s2, n_alpha_s2) = _build_tree(
                q_m, p_m, log_u, v, depth - 1, step_size, log_pi_fn, grad_fn, H0, rng
            )
        else:
            (_, _, q_p, p_p,
             q_s2, n_s2, s_s2,
             alpha_s2, n_alpha_s2) = _build_tree(
                q_p, p_p, log_u, v, depth - 1, step_size, log_pi_fn, grad_fn, H0, rng
            )

        # Metropolis update: accept proposal from second subtree
        total = n_s + n_s2
        if total > 0 and rng.uniform() < (n_s2 / total):
            q_s = q_s2

        # U-turn check
        span  = q_p - q_m
        s_s   = (s_s2
                 * int(float(np.dot(span, p_m)) >= 0)
                 * int(float(np.dot(span, p_p)) >= 0))

        n_s          += n_s2
        alpha_s      += alpha_s2
        n_alpha_s    += n_alpha_s2

    return q_m, p_m, q_p, p_p, q_s, n_s, s_s, alpha_s, n_alpha_s


def nuts_step(q0, pi_fn, grad_fn, step_size, max_tree_depth, rng):
    """
    One NUTS transition from q0.

    Returns
    -------
    q_new      : ndarray (d,) — new position
    avg_alpha  : float        — average acceptance prob (for dual averaging)
    """
    d  = q0.size
    p0 = rng.normal(size=d)

    log_pi_fn = lambda x: _log_pi(pi_fn, x)
    H0        = _hamiltonian(log_pi_fn, q0, p0)

    # log-slice threshold: log u = log(U) - H0  so u ~ Uniform[0, exp(-H0)]
    log_u = np.log(rng.uniform() + 1e-300) - H0

    q_m = q_p = q0.copy()
    p_m = p_p = p0.copy()
    q_new     = q0.copy()
    n         = 1
    s         = 1
    alpha_sum = 0.0
    n_alpha   = 0

    depth = 0
    while s and depth < max_tree_depth:
        v = int(rng.choice([-1, 1]))

        if v == -1:
            (q_m, p_m, _, _, q_cand, n_cand, s_cand,
             a, na) = _build_tree(
                q_m, p_m, log_u, v, depth,
                step_size, log_pi_fn, grad_fn, H0, rng
            )
        else:
            (_, _, q_p, p_p, q_cand, n_cand, s_cand,
             a, na) = _build_tree(
                q_p, p_p, log_u, v, depth,
                step_size, log_pi_fn, grad_fn, H0, rng
            )

        if s_cand and n_cand > 0:
            if rng.uniform() < min(1.0, n_cand / n):
                q_new = q_cand

        n          += n_cand
        span        = q_p - q_m
        s           = (s_cand
                       * int(float(np.dot(span, p_m)) >= 0)
                       * int(float(np.dot(span, p_p)) >= 0))
        alpha_sum  += a
        n_alpha    += na
        depth      += 1

    avg_alpha = alpha_sum / n_alpha if n_alpha > 0 else 0.0
    return q_new, float(min(1.0, avg_alpha)), n_alpha


# ------------------------------------------------------------------
# Step-size calibration  (binary search on NUTS acceptance rate)
# ------------------------------------------------------------------

def calibrate_step_size(q0, pi_fn, grad_fn, init_step,
                         target_accept=0.65, n_steps=80,
                         max_rounds=8, rng=None):
    """
    Find a leapfrog step size ε that achieves `target_accept` ± 0.10.

    Doubles or halves ε based on the average NUTS acceptance rate over a
    short pilot run, repeating up to `max_rounds` times.

    Parameters
    ----------
    q0           : ndarray (d,) — starting position
    pi_fn        : callable — unnormalized target density
    grad_fn      : callable (d-array) → (d-array) — gradient of log π
    init_step    : float — initial guess for ε
    target_accept: float — desired average acceptance probability (default 0.65)
    n_steps      : int   — NUTS steps per calibration round
    max_rounds   : int   — maximum doublings / halvings
    rng          : np.random.Generator, optional

    Returns
    -------
    eps : float — calibrated step size
    """
    rng = rng if rng is not None else np.random.default_rng()
    eps = float(init_step)
    q   = q0.copy()

    for _ in range(max_rounds):
        alphas = []
        q_run  = q.copy()
        for _ in range(n_steps):
            q_run, alpha, _ = nuts_step(
                q_run, pi_fn, grad_fn, eps, max_tree_depth=3, rng=rng
            )
            alphas.append(alpha)
        avg = float(np.mean(alphas))

        if avg > target_accept + 0.10:
            eps *= 2.0
        elif avg < target_accept - 0.10:
            eps /= 2.0
        else:
            break

    return float(np.clip(eps, 1e-6, 1e4))


# ------------------------------------------------------------------
# Hybrid Teleporting-NUTS sampler
# ------------------------------------------------------------------

class HybridTeleportingNUTS:
    """
    Hybrid Teleporting-MCMC with NUTS local moves.

    The importance-weight mechanism from Teleporting MCMC (Skene et al. 2023)
    selects whether each step is a teleport or a local update:

      i ≠ j  →  teleport  :  standard Z_fwd / Z_rev acceptance
      i == j →  local NUTS :  one NUTS trajectory from x_j

    NUTS step size is adapted during the first `num_warmup` iterations via
    dual averaging.

    Parameters
    ----------
    pi_fn          : callable (d-array) → float — unnormalized target density
    q_sample_fn    : callable (x, rng) → z      — teleport proposal sampler
    q_density_fn   : callable (x, mean) → float — teleport proposal density
    init_step_size : float  — initial NUTS leapfrog step size
    max_tree_depth : int    — NUTS max tree depth (default 5, cap ≤ 32 leapfrog steps)
    target_accept  : float  — dual-averaging target acceptance rate (default 0.65)
    grad_eps       : float  — finite-difference step for ∇log π (default 1e-5)
    rng            : np.random.Generator, optional
    """

    def __init__(self, pi_fn, q_sample_fn, q_density_fn,
                 init_step_size=0.1, max_tree_depth=5,
                 target_accept=0.65, grad_eps=1e-5, rng=None):
        self.pi_fn          = pi_fn
        self.q_sample_fn    = q_sample_fn
        self.q_density_fn   = q_density_fn
        self.init_step_size = float(init_step_size)
        self.max_tree_depth = int(max_tree_depth)
        self.target_accept  = float(target_accept)
        self.grad_eps       = float(grad_eps)
        self.rng            = rng if rng is not None else np.random.default_rng()

    def _compute_Z(self, x, z):
        N = len(x)
        total = 0.0
        for l in range(N):
            num = self.q_density_fn(x[l], z)
            for k in range(N):
                if k != l:
                    num += self.q_density_fn(x[l], x[k])
            pi_val = float(self.pi_fn(x[l]))
            if pi_val > 0.0:
                total += num / pi_val
        return total

    def _compute_importance_weights(self, x, z):
        N     = len(x)
        Z_val = self._compute_Z(x, z)
        w     = np.zeros(N)
        for i in range(N):
            num = self.q_density_fn(x[i], z)
            for k in range(N):
                if k != i:
                    num += self.q_density_fn(x[i], x[k])
            pi_val = float(self.pi_fn(x[i]))
            if pi_val > 0.0 and Z_val > 0.0:
                w[i] = (num / pi_val) / Z_val
        s = w.sum()
        return (w / s if s > 0.0 else np.ones(N) / N), Z_val

    def _grad_fn(self, x):
        return _grad_log_pi(self.pi_fn, x, eps=self.grad_eps)

    def run(self, x0, num_iter, num_warmup=None):
        """
        Run the sampler.

        The first `num_warmup` iterations are used to calibrate the NUTS step
        size via a binary-search pilot on the initial walkers.  The calibrated
        step size is then fixed for the remaining (post-warmup) iterations.

        Parameters
        ----------
        x0         : array-like (N, d) — initial walker positions
        num_iter   : int — total iterations (including warm-up)
        num_warmup : int — calibration budget; defaults to num_iter // 4

        Returns
        -------
        dict with keys:
          samples                : (num_iter+1, N, d)
          acceptance_rate        : fraction of steps that moved
          teleport_proposal_rate : fraction of steps where i ≠ j was proposed
          teleport_accept_rate   : fraction of steps that were accepted teleports
          local_nuts_rate        : fraction of steps that used NUTS
          calibrated_step_size   : NUTS ε chosen during warm-up
          num_iter, num_walkers
        """
        if num_warmup is None:
            num_warmup = num_iter // 4

        x    = np.array(x0, dtype=float, copy=True)
        N, d = x.shape

        # ---- calibrate step size once before the main loop ----
        calib_n    = max(60, num_warmup // 4)
        step_size  = calibrate_step_size(
            q0            = x[self.rng.integers(N)].copy(),
            pi_fn         = self.pi_fn,
            grad_fn       = self._grad_fn,
            init_step     = self.init_step_size,
            target_accept = self.target_accept,
            n_steps       = calib_n,
            rng           = self.rng,
        )

        history            = [x.copy()]
        teleports_proposed = 0
        teleports_accepted = 0
        local_nuts_steps   = 0
        total_moved        = 0

        for it in range(num_iter):
            j = self.rng.integers(N)
            z = self.q_sample_fn(x[j], self.rng)

            weights, Z_forward = self._compute_importance_weights(x, z)
            i = self.rng.choice(N, p=weights)

            if i != j:
                # ---- teleport ----
                teleports_proposed += 1
                x_prop    = x.copy()
                old_xi    = x[i].copy()
                x_prop[i] = z
                Z_reverse = self._compute_Z(x_prop, old_xi)
                alpha     = min(1.0, Z_forward / (Z_reverse + 1e-300))

                if self.rng.uniform() < alpha:
                    x = x_prop
                    teleports_accepted += 1
                    total_moved        += 1

            else:
                # ---- local NUTS move for walker j ----
                local_nuts_steps += 1
                x_j_new, _, _ = nuts_step(
                    q0             = x[j].copy(),
                    pi_fn          = self.pi_fn,
                    grad_fn        = self._grad_fn,
                    step_size      = step_size,
                    max_tree_depth = self.max_tree_depth,
                    rng            = self.rng,
                )
                x[j]        = x_j_new
                total_moved += 1   # NUTS is internally MH-correct

            history.append(x.copy())

        return {
            "samples":               np.array(history),
            "acceptance_rate":       total_moved / num_iter,
            "teleport_proposal_rate": teleports_proposed / num_iter,
            "teleport_accept_rate":  teleports_accepted / num_iter,
            "local_nuts_rate":       local_nuts_steps / num_iter,
            "calibrated_step_size":  step_size,
            "num_iter":              num_iter,
            "num_walkers":           N,
        }
