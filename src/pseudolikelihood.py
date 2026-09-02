"""Pseudo-likelihood estimation of the Potts coupling J (extension 4).

The joint likelihood of a Markov random field is intractable because of its
normalising constant. Besag's pseudo-likelihood (Besag 1975) replaces it with
the product of per-site conditionals, which for the Potts prior is tractable and
concave in J, so J can be estimated by a one-dimensional maximisation.

For a labelling x with q states, the log pseudo-likelihood of the coupling J is

    l(J) = sum_i [ J * n_i(x_i) - log sum_k exp(J * n_i(k)) ]

where n_i(k) is the number of neighbours of pixel i in state k. Its derivative,

    l'(J) = sum_i [ n_i(x_i) - E_{k~p_i}[ n_i(k) ] ],

is monotone decreasing in J (l is concave), so the maximum is found by bisection
on l'(J) = 0. This module provides both the estimator and the log-PL curve, so
the concavity and the location of the optimum can be inspected.

Besag (1986, sec 5.1.2-5.1.3) warns that estimating J by pseudo-likelihood
DURING reconstruction can diverge (his estimates went 1.3, 1.9, infinity over
successive cycles). This module therefore also supports estimating J from a
fixed labelling (e.g. the clean image, or a single denoised estimate), and the
accompanying experiment tests stability explicitly rather than assuming it.
"""
import numpy as np

from potts import _neighbour_counts, quantise


def _pl_gradient(x, q, J):
    """l'(J) for a fixed integer labelling x at coupling J.

    Returns the scalar sum_i [ n_i(x_i) - E_{k}[n_i(k)] ].
    """
    n = _neighbour_counts(x, q)                 # (q, H, W) integer counts
    H, W = x.shape
    # observed neighbour count at each pixel's actual state
    obs = n[x, np.arange(H)[:, None], np.arange(W)[None, :]]   # (H, W)
    # local conditional p_i(k) proportional to exp(J n_i(k))
    logits = J * n                              # (q, H, W)
    logits -= logits.max(axis=0, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=0, keepdims=True)           # (q, H, W)
    expected = (p * n).sum(axis=0)              # E_k[n_i(k)]  (H, W)
    return float((obs - expected).sum())


def log_pseudolikelihood(x, q, J):
    """l(J) up to an additive constant, for plotting the concave curve."""
    n = _neighbour_counts(x, q)
    H, W = x.shape
    obs = n[x, np.arange(H)[:, None], np.arange(W)[None, :]]
    logits = J * n
    m = logits.max(axis=0, keepdims=True)
    logZ = (m[0] + np.log(np.exp(logits - m).sum(axis=0)))     # (H, W)
    return float((J * obs - logZ).sum())


def estimate_J(x, q, bracket=(0.0, 20.0), tol=1e-4, max_iter=200):
    """Maximum-pseudo-likelihood estimate of J for a fixed labelling x.

    x may be a [0,1] image (quantised internally) or an integer state array.
    Uses bisection on the concave gradient l'(J) = 0. Returns (J_hat, info),
    where info records whether the optimum is interior or ran to the bracket
    edge (the divergence Besag warns about shows up as hitting the upper edge).
    """
    if x.dtype.kind == "f":
        x = quantise(x, q)
    lo, hi = bracket
    g_lo, g_hi = _pl_gradient(x, q, lo), _pl_gradient(x, q, hi)

    # l' is decreasing; a root exists only if g_lo > 0 > g_hi.
    if g_lo <= 0:
        return lo, {"status": "boundary_low", "J": lo, "g_lo": g_lo}
    if g_hi >= 0:
        # gradient still positive at the top of the bracket -> optimum is at or
        # beyond hi: the estimate is diverging (Besag's beta_hat -> infinity).
        return hi, {"status": "diverged_high", "J": hi, "g_hi": g_hi}

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g = _pl_gradient(x, q, mid)
        if abs(g) < tol or (hi - lo) < tol:
            return mid, {"status": "interior", "J": mid, "grad": g}
        if g > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), {"status": "interior_maxit", "J": 0.5 * (lo + hi)}
