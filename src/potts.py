"""
potts.py — q-state Potts-model denoiser for grayscale images
(project extension 1: generalising the binary Ising prior).

Model

Each pixel takes one of q discrete states x_i in {0, ..., q-1},
representing intensity levels c_k = (k + 0.5)/q in [0, 1].

Prior (Potts):        P(x)    ∝ exp( J * sum_{(i,j)} δ(x_i, x_j) )
Likelihood (AWGN):    P(y|x)  ∝ prod_i exp( -(y_i - c_{x_i})² / (2σ²) )

Posterior conditional at pixel i (used by Gibbs sampling):

    P(x_i = k | rest, y) ∝ exp( -(y_i - c_k)²/(2σ²) + J * n_i(k) )

where n_i(k) is the number of 4-neighbours of i currently in state k.
The Potts energy with per-node local evidence is Murphy 2023 Eq. 4.86.
This reduces to the Ising denoiser when q = 2: Murphy 2023 Eq. 4.84 states
the reparameterisation J_potts = 2 * J_ising.

Inference: checkerboard Gibbs sampling, exactly as in the binary case,
but each half-sweep samples from a q-way softmax instead of a sigmoid.

Estimators

- "mmse" (default): posterior-mean intensity, averaged over post-burn-in
  samples. Minimises expected squared error, i.e. maximises PSNR.
- "mpm": per-pixel most frequent state (marginal posterior mode).
"""

import numpy as np


def quantise(img, q):
    """[0,1] grayscale image -> integer states {0,...,q-1}."""
    return np.clip((img * q).astype(np.int32), 0, q - 1)


def levels(q):
    """Intensity value of each state (bin centres)."""
    return (np.arange(q) + 0.5) / q


def _neighbour_counts(x, q, Jh=None, Jv=None):
    """(q,H,W) array: for each state k, the (optionally edge-weighted) count of
    4-neighbours in state k.

    Isotropic (Jh is None): plain integer counts, so the caller multiplies by a
    single scalar J. This is the original behaviour, unchanged.

    Anisotropic (Jh, Jv given): each neighbour contributes its EDGE coupling
    rather than 1, so the returned array already carries the per-edge J and the
    caller must NOT multiply by a scalar J again. Couplings live on edges:
        Jh : (H, W-1)  horizontal edge (r,c)--(r,c+1)
        Jv : (H-1, W)  vertical   edge (r,c)--(r+1,c)
    """
    H, W = x.shape
    onehot = np.zeros((q, H, W), dtype=np.float32)
    onehot[x, np.arange(H)[:, None], np.arange(W)[None, :]] = 1.0
    n = np.zeros_like(onehot)
    if Jh is None:
        n[:, 1:, :] += onehot[:, :-1, :]
        n[:, :-1, :] += onehot[:, 1:, :]
        n[:, :, 1:] += onehot[:, :, :-1]
        n[:, :, :-1] += onehot[:, :, 1:]
    else:
        # weight each neighbour's contribution by the coupling on the shared edge
        n[:, 1:, :] += Jv[None, :, :] * onehot[:, :-1, :]   # neighbour above
        n[:, :-1, :] += Jv[None, :, :] * onehot[:, 1:, :]   # neighbour below
        n[:, :, 1:] += Jh[None, :, :] * onehot[:, :, :-1]   # neighbour left
        n[:, :, :-1] += Jh[None, :, :] * onehot[:, :, 1:]   # neighbour right
    return n


def potts_gibbs_denoise(y, sigma, q=32, J=1.0, n_sweeps=30, burn_in=10,
                        seed=None, estimator="mmse", Jh=None, Jv=None):
    """Denoise a [0,1] grayscale image corrupted by Gaussian noise.

    Parameters
    
    y         : (H,W) float array in [0,1], the noisy observation
    sigma     : noise standard deviation (same [0,1] scale)
    q         : number of Potts states (intensity levels)
    J         : Potts coupling strength (scalar; ISOTROPIC prior)
    n_sweeps  : total Gibbs sweeps; burn_in discarded before averaging
    estimator : "mmse" -> posterior-mean intensity (best PSNR)
                "mpm"  -> most probable state per pixel
    Jh, Jv    : optional per-edge coupling arrays for ANISOTROPIC (spatially
                adaptive) coupling. Jh:(H,W-1), Jv:(H-1,W). When given, the
                scalar J is ignored and the edge arrays set the coupling on
                every edge -- e.g. weak across image edges, strong in flat
                regions. Build them with anisotropic.edge_couplings(...) scaled
                so that J0 plays the role of J. When None, the model is the
                original isotropic Potts prior with coupling J.

    Returns
    -------
    x_hat : (H,W) float array in [0,1], the denoised image
    """
    rng = np.random.default_rng(seed)
    H, W = y.shape
    c = levels(q)                                     # (q,)
    aniso = Jh is not None

    # Unary log-potentials: -(y - c_k)^2 / (2 sigma^2), shape (q,H,W)
    unary = -((y[None, :, :] - c[:, None, None]) ** 2) / (2.0 * sigma ** 2)

    # Initialise at the quantised observation (ML estimate)
    x = quantise(y, q)

    r, ccol = np.indices((H, W))
    black = (r + ccol) % 2 == 0
    masks = (black, ~black)

    mean_accum = np.zeros((H, W), dtype=np.float64)
    state_hist = np.zeros((q, H, W), dtype=np.uint16) if estimator == "mpm" else None
    n_avg = 0

    for sweep in range(n_sweeps):
        for mask in masks:
            if aniso:
                # edge arrays already carry the coupling; do NOT multiply by J
                logits = unary + _neighbour_counts(x, q, Jh, Jv)
            else:
                logits = unary + J * _neighbour_counts(x, q)
            # Gumbel-max trick: vectorised categorical sampling
            g = rng.gumbel(size=logits.shape)
            sample = np.argmax(logits + g, axis=0)           # (H,W)
            x[mask] = sample[mask]
        if sweep >= burn_in:
            mean_accum += c[x]
            if state_hist is not None:
                state_hist[x, r, ccol] += 1
            n_avg += 1

    if estimator == "mmse" and n_avg > 0:
        return mean_accum / n_avg
    if estimator == "mpm" and n_avg > 0:
        return c[np.argmax(state_hist, axis=0)]
    return c[x]