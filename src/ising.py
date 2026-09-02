"""
ising.py — Binary Ising-model image denoiser.

Model

Posterior over clean spins x in {-1,+1}^N given noisy observation y:

    P(x | y)  ∝  exp( beta * sum_i y_i x_i  +  J * sum_{(i,j)} x_i x_j )

- The pairwise term is the ferromagnetic Ising prior (smoothness).
  Neighbourhood: 4-connected (von Neumann) for the baseline;
  8-connected with distance-weighted diagonal coupling for the
  next-nearest-neighbour extension.
- The unary term is the binary-symmetric-channel likelihood, i.e. a
  local-evidence / unary bias term (Murphy 2023 §4.3.2: Eq. 4.82 gives the
  binary-Ising energy with a unary term; Eq. 4.86 is the general K-state /
  Potts local-evidence form). The specific BSC weight beta below is from
  Nishimori & Wong 1999.

For a flip probability p, the statistically correct evidence weight is
    beta = 0.5 * ln((1 - p) / p)          (see e.g. Nishimori & Wong 1999)

Inference

- gibbs_denoise      : checkerboard (red-black) Gibbs sampling
                       (Geman & Geman 1984; Murphy 2023 §12.3.3)
- meanfield_denoise  : coordinate-ascent mean-field / CAVI
                       (Murphy 2023 §10.3.2)

Both are fully vectorised with NumPy: the lattice is partitioned into
two interleaved "checkerboard" subsets; sites within a subset are
conditionally independent given the other subset, so each half-sweep
updates ~N/2 pixels in one vector operation.
"""

import numpy as np



# Neighbourhood sums


def neighbour_sum(x, diag_weight=0.0):
    """Sum of neighbouring spins at every site (zero-padded boundary).

    diag_weight = 0.0 -> 4-neighbour baseline.
    diag_weight = w   -> 8-neighbour extension: diagonals contribute w * spin
                         (e.g. w = 0.5 as in Cohen et al. 2015, or
                          w = 1/sqrt(2) for inverse-distance weighting).
    """
    s = np.zeros_like(x, dtype=np.float64)
    # axial neighbours (N, S, W, E)
    s[1:, :] += x[:-1, :]
    s[:-1, :] += x[1:, :]
    s[:, 1:] += x[:, :-1]
    s[:, :-1] += x[:, 1:]
    if diag_weight != 0.0:
        s[1:, 1:] += diag_weight * x[:-1, :-1]
        s[:-1, :-1] += diag_weight * x[1:, 1:]
        s[1:, :-1] += diag_weight * x[:-1, 1:]
        s[:-1, 1:] += diag_weight * x[1:, :-1]
    return s


def checkerboard_masks(shape):
    """Boolean masks for the two interleaved sublattices."""
    r, c = np.indices(shape)
    black = (r + c) % 2 == 0
    return black, ~black


def beta_from_flip_prob(p):
    """Statistically correct local-evidence weight for flip probability p."""
    return 0.5 * np.log((1.0 - p) / p)



# Gibbs sampling


def gibbs_denoise(y, J=1.0, beta=None, p=None, n_sweeps=30, burn_in=10,
                  diag_weight=0.0, seed=None, estimator="mpm"):
    """Denoise a {-1,+1} image with checkerboard Gibbs sampling.

    Parameters

    y           : (H,W) int array of observed spins in {-1,+1}
    J           : coupling strength of the Ising prior
    beta        : local-evidence weight; if None, computed from p
    p           : assumed flip probability (used when beta is None)
    n_sweeps    : total full sweeps (one sweep = both checkerboard halves)
    burn_in     : sweeps discarded before averaging (MPM estimator)
    diag_weight : 0.0 for 4-neighbour baseline; >0 adds diagonals
    estimator   : "mpm" -> sign of posterior mean over post-burn-in samples
                  (marginal posterior mode / TPM; optimal for pixel accuracy,
                   cf. Nishimori & Wong 1999); "last" -> final sample.

    Returns
    
    x_hat : (H,W) int8 array of denoised spins in {-1,+1}
    """
    if beta is None:
        if p is None:
            raise ValueError("provide either beta or p")
        beta = beta_from_flip_prob(p)
    rng = np.random.default_rng(seed)

    x = y.astype(np.int8).copy()          # initialise at the observation
    black, white = checkerboard_masks(y.shape)
    accum = np.zeros(y.shape, dtype=np.float64)
    n_avg = 0

    for sweep in range(n_sweeps):
        for mask in (black, white):
            field = 2.0 * (beta * y + J * neighbour_sum(x, diag_weight))
            prob_up = 1.0 / (1.0 + np.exp(-field))
            u = rng.random(y.shape)
            x[mask] = np.where(u[mask] < prob_up[mask], 1, -1)
        if sweep >= burn_in:
            accum += x
            n_avg += 1

    if estimator == "mpm" and n_avg > 0:
        x_hat = np.where(accum >= 0, 1, -1).astype(np.int8)
    else:
        x_hat = x
    return x_hat



# Mean-field (CAVI)


def meanfield_denoise(y, J=1.0, beta=None, p=None, n_iters=30,
                      diag_weight=0.0, damping=0.5, tol=1e-5):
    """Denoise with coordinate-ascent mean-field variational inference.

    Iterates mu_i <- tanh(beta*y_i + J * sum_j mu_j) with damping, using
    checkerboard ordering. Deterministic and fast; ignores posterior
    correlations (Murphy 2023 §10.3.2).

    Returns
    
    x_hat : (H,W) int8 array of denoised spins (sign of the means)
    mu    : (H,W) float array of variational means in [-1,1]
    """
    if beta is None:
        if p is None:
            raise ValueError("provide either beta or p")
        beta = beta_from_flip_prob(p)

    mu = y.astype(np.float64).copy()
    black, white = checkerboard_masks(y.shape)

    for _ in range(n_iters):
        mu_old = mu.copy()
        for mask in (black, white):
            field = beta * y + J * neighbour_sum(mu, diag_weight)
            new = np.tanh(field)
            mu[mask] = damping * mu[mask] + (1 - damping) * new[mask]
        if np.max(np.abs(mu - mu_old)) < tol:
            break

    x_hat = np.where(mu >= 0, 1, -1).astype(np.int8)
    return x_hat, mu