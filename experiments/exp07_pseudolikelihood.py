"""Experiment 07 - learning the Potts coupling J by pseudo-likelihood (extension 4).

Instead of fixing J by hand (J=2.0 adopted in exp03) or by sweeping, this
experiment estimates J from data by maximum pseudo-likelihood (Besag 1975). The
question is twofold: does the learned J agree with the empirically-optimal value,
and is the estimator stable - Besag (1986) warns it can diverge when re-estimated
during reconstruction.

Three regimes are tested per image, mirroring Besag's own investigation:
  (a) CLEAN   : J estimated from the ground-truth labelling. This is the coupling
                that best describes real image structure - an upper reference.
  (b) NOISY   : J estimated from the noisy observation. Tests whether noise
                corrupts the estimate.
  (c) ITERATED: J re-estimated from the current denoised labelling on each of
                several ICM-style cycles. This is the regime Besag flags as prone
                to divergence (his beta_hat went 1.3, 1.9, infinity).

The estimator is validated separately (it recovers a known J from a sampled field
to within 0.04). Whatever the outcome here - convergence near J=2.0, or the
predicted divergence is a genuine, citable result.
"""
import sys, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_folder, gaussian_noise, psnr
from potts import potts_gibbs_denoise, quantise, _neighbour_counts, levels
from pseudolikelihood import estimate_J, log_pseudolikelihood

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

Q = 32
J_ADOPTED = 2.0                 # the hand-tuned value from exp03, for comparison
SIGMA255 = 25
SIGMA = SIGMA255 / 255.0
N_SET12 = None
N_CYCLES = 8                    # ICM-style re-estimation cycles


def noise_seed(path):
    import zlib
    return 100 + (zlib.crc32(path.name.encode()) & 0xFFFF)


def denoise_at_J(y, J, seed):
    """One MPM Potts reconstruction at coupling J (integer labelling out)."""
    xh = potts_gibbs_denoise(y, sigma=SIGMA, q=Q, J=J, n_sweeps=25, burn_in=10,
                             seed=seed, estimator="mpm")
    return quantise(xh, Q)


def iterate_estimate(y, seed, n_cycles=N_CYCLES):
    """Alternate: estimate J from current labelling, then re-denoise at that J.
    Returns the list of J values across cycles (the stability trace)."""
    x = quantise(y, Q)                     # start from the noisy labelling
    traj = []
    for _ in range(n_cycles):
        Jhat, info = estimate_J(x, q=Q, bracket=(0.0, 20.0))
        traj.append((Jhat, info["status"]))
        # re-denoise at the freshly estimated J
        xh = potts_gibbs_denoise(y, sigma=SIGMA, q=Q, J=Jhat, n_sweeps=15,
                                 burn_in=6, seed=seed, estimator="mpm")
        x = quantise(xh, Q)
    return traj


imgs, paths = load_folder(ROOT / "data" / "set12", n=N_SET12, return_paths=True)

print(f"Pseudo-likelihood J estimation on Set12, sigma={SIGMA255}, q={Q}. "
      f"Hand-tuned reference J={J_ADOPTED}.\n")
print(f"{'image':<10} | {'J(clean)':>9} | {'J(noisy)':>9} | "
      f"{'J(final iter)':>13} | {'iter status':>12}")
print("-" * 66)

rows = []
clean_Js, noisy_Js, iter_final_Js = [], [], []
example_traj = None
for img, path in zip(imgs, paths):
    y = gaussian_noise(img, sigma=SIGMA, seed=noise_seed(path))
    xclean = quantise(img, Q)

    Jc, ic = estimate_J(xclean, q=Q, bracket=(0.0, 20.0))
    Jn, in_ = estimate_J(y, q=Q, bracket=(0.0, 20.0))
    traj = iterate_estimate(y, seed=noise_seed(path))
    Jfinal, statusfinal = traj[-1]

    clean_Js.append(Jc); noisy_Js.append(Jn); iter_final_Js.append(Jfinal)
    if path.stem in ("09", "9") or example_traj is None:
        example_traj = (path.stem, [t[0] for t in traj])

    rows.append({"image": path.stem, "J_clean": Jc, "J_noisy": Jn,
                 "J_iter_final": Jfinal, "iter_status": statusfinal,
                 "clean_status": ic["status"], "noisy_status": in_["status"]})
    print(f"{path.stem:<10} | {Jc:>9.3f} | {Jn:>9.3f} | {Jfinal:>13.3f} | "
          f"{statusfinal:>12}")

with open(RESULTS / "exp07_pseudolikelihood.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# summary
def summ(name, arr):
    a = np.array(arr)
    print(f"  {name:<16} mean {a.mean():.3f}  std {a.std():.3f}  "
          f"range [{a.min():.3f}, {a.max():.3f}]")

n_div = sum(1 for r in rows if r["iter_status"] == "diverged_high")
print("\nSummary:")
summ("J from clean", clean_Js)
summ("J from noisy", noisy_Js)
summ("J iterated", iter_final_Js)
print(f"  hand-tuned reference: {J_ADOPTED}")
print(f"  iterated estimates that DIVERGED (hit bracket edge): "
      f"{n_div}/{len(rows)}")

# figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# left: learned J vs hand-tuned, per regime
x = np.arange(len(rows)); w = 0.27
ax1.bar(x - w, clean_Js, w, label="J from clean", color="#3f9b52")
ax1.bar(x,      noisy_Js, w, label="J from noisy", color="#3b6fb0")
ax1.bar(x + w,  iter_final_Js, w, label="J iterated (final)", color="#c0392b")
ax1.axhline(J_ADOPTED, ls="--", color="black", label=f"hand-tuned J={J_ADOPTED}")
ax1.set_xticks(x); ax1.set_xticklabels([r["image"] for r in rows], fontsize=7,
                                       rotation=45)
ax1.set_ylabel("estimated J"); ax1.set_title("Learned vs hand-tuned coupling")
ax1.legend(fontsize=8)

# right: the stability trace (Besag's divergence test) for one image
name, traj = example_traj
ax2.plot(range(1, len(traj) + 1), traj, marker="o", color="#c0392b")
ax2.axhline(J_ADOPTED, ls="--", color="black", label=f"hand-tuned J={J_ADOPTED}")
ax2.set_xlabel("re-estimation cycle"); ax2.set_ylabel("estimated J")
ax2.set_title(f"Stability of iterated estimation (image {name})")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Pseudo-likelihood learning of the Potts coupling", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RESULTS / "exp07_pseudolikelihood.png", dpi=130)

print("\nsaved -> results/exp07_pseudolikelihood.csv, exp07_pseudolikelihood.png")
