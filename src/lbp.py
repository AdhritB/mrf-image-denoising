"""
lbp.py — Loopy belief propagation for the binary Ising denoising posterior.

Model (same as ising.py):
    P(x | y) ∝ exp( beta * sum_i y_i x_i + J * sum_{(i,j)} x_i x_j )

Sum-product message passing on the 4-connected pixel grid, in the
log-ratio parameterisation. For binary spins x in {-1,+1}, a message
m_{i->j}(x_j) is fully described by the scalar

    mu_{i->j} = 0.5 * log( m_{i->j}(+1) / m_{i->j}(-1) )

and the standard update for a pairwise potential exp(J x_i x_j) is

    mu_{i->j} = atanh( tanh(J) * tanh( beta*y_i + sum_{k in N(i)\\j} mu_{k->i} ) )

(see Yedidia, Freeman & Weiss 2005; Murphy 2023 §9.3-9.4). On a graph
with cycles this is a heuristic ("loopy" BP): fixed points correspond to
stationary points of the Bethe free energy and convergence is not
guaranteed, so we use damping and an early-stopping tolerance.

The implementation is fully vectorised: messages are stored as four
(H, W) arrays, one per incoming direction, and all updates are
synchronous (flooding schedule) with damping.
"""

import numpy as np

# Direction indices: message arriving AT pixel (r,c) FROM its neighbour...
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3   # ...above, below, to the left, to the right


def lbp_denoise(y, J=1.0, beta=None, p=None, n_iters=60, damping=0.5,
                tol=1e-5, return_info=False):
    """Denoise a {-1,+1} image with loopy belief propagation.

    Parameters
    
    y        : (H,W) array of observed spins in {-1,+1}
    J        : Ising coupling strength
    beta     : local-evidence weight; if None, computed from p
    p        : assumed flip probability (used when beta is None)
    n_iters  : maximum message-passing iterations
    damping  : new_msg = damping*old + (1-damping)*update  (0 = no damping)
    tol      : stop when max |message change| < tol
    return_info : also return (n_iterations_run, converged, beliefs)

    Returns
    
    x_hat : (H,W) int8 array of denoised spins (sign of the beliefs)
    info  : optional dict with convergence diagnostics
    """
    if beta is None:
        if p is None:
            raise ValueError("provide either beta or p")
        beta = 0.5 * np.log((1.0 - p) / p)

    H, W = y.shape
    local = beta * y.astype(np.float64)      # unary log-ratio at each pixel
    tJ = np.tanh(J)

    # msgs[d] = log-ratio message arriving at each pixel from direction d
    msgs = np.zeros((4, H, W), dtype=np.float64)

    converged = False
    it = 0
    for it in range(1, n_iters + 1):
        total_in = local + msgs.sum(axis=0)

        new = np.zeros_like(msgs)
        # Outgoing message from pixel i to a neighbour j excludes the
        # message that arrived at i from j (avoid double counting), then
        # passes through the edge factor: atanh(tanh(J) * tanh(...)).
        # It then *arrives* at j from the opposite direction.

        # message sent downward by each pixel -> arrives at row+1 as "UP"
        out = total_in - msgs[DOWN]
        new[UP][1:, :] = np.arctanh(np.clip(tJ * np.tanh(out[:-1, :]),
                                            -1 + 1e-12, 1 - 1e-12))
        # message sent upward -> arrives at row-1 as "DOWN"
        out = total_in - msgs[UP]
        new[DOWN][:-1, :] = np.arctanh(np.clip(tJ * np.tanh(out[1:, :]),
                                               -1 + 1e-12, 1 - 1e-12))
        # message sent rightward -> arrives at col+1 as "LEFT"
        out = total_in - msgs[RIGHT]
        new[LEFT][:, 1:] = np.arctanh(np.clip(tJ * np.tanh(out[:, :-1]),
                                              -1 + 1e-12, 1 - 1e-12))
        # message sent leftward -> arrives at col-1 as "RIGHT"
        out = total_in - msgs[LEFT]
        new[RIGHT][:, :-1] = np.arctanh(np.clip(tJ * np.tanh(out[:, 1:]),
                                                -1 + 1e-12, 1 - 1e-12))

        updated = damping * msgs + (1.0 - damping) * new
        delta = np.max(np.abs(updated - msgs))
        msgs = updated
        if delta < tol:
            converged = True
            break

    beliefs = local + msgs.sum(axis=0)
    x_hat = np.where(beliefs >= 0, 1, -1).astype(np.int8)

    if return_info:
        return x_hat, {"iterations": it, "converged": converged,
                       "beliefs": beliefs}
    return x_hat
