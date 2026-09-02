"""Experiment 02 — systematic inference comparison (core extension 3).

Compares three inference algorithms for the same Ising denoising posterior
on binarised MNIST test digits:

  1. Checkerboard Gibbs sampling (MPM estimate)   [Murphy 2023 §12.3.3]
  2. Mean-field CAVI                              [Murphy 2023 §10.3.2]
  3. Loopy belief propagation (sum-product)       [Yedidia et al. 2005]

Overall pixel accuracy on binarised MNIST is background-dominated (~85%
of pixels are background), so it stays high even when a method erases the
digit. We therefore report overall, FOREGROUND and BALANCED accuracy.
The foreground column exposes the key inference-comparison finding: at a
given prior strength the three algorithms preserve the thin digit strokes
to very different degrees, because they approximate the same posterior in
different ways.

Prior strength J = 0.5 (see exp01: J = 1.0 over-smooths thin strokes).
"""
import sys, time, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_mnist, binarise, flip_noise, denoising_report
from ising import gibbs_denoise, meanfield_denoise
from lbp import lbp_denoise

DATA = Path(__file__).resolve().parents[1] / "data" / "mnist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_IMAGES = 50
FLIP_PROBS = [0.05, 0.10, 0.20]
J = 0.5

imgs, _ = load_mnist(DATA, split="test", n=N_IMAGES)
clean = [binarise(im) for im in imgs]

METHODS = ["gibbs", "mf", "lbp"]
rows = []
for p in FLIP_PROBS:
    acc = {mth: {"overall": [], "foreground": [], "balanced": []} for mth in METHODS}
    base_noisy = []
    t = {mth: 0.0 for mth in METHODS}
    lbp_converged = 0
    for k, x in enumerate(clean):
        y = flip_noise(x, p=p, seed=1000 + k)
        base_noisy.append(np.mean(x == y))

        t0 = time.perf_counter()
        xg = gibbs_denoise(y, J=J, p=p, n_sweeps=40, burn_in=15, seed=k)
        t["gibbs"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        xm, _ = meanfield_denoise(y, J=J, p=p, n_iters=40)
        t["mf"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        xl, info = lbp_denoise(y, J=J, p=p, n_iters=200, damping=0.5,
                               return_info=True)
        t["lbp"] += time.perf_counter() - t0
        lbp_converged += info["converged"]

        for mth, xh in zip(METHODS, (xg, xm, xl)):
            r = denoising_report(x, xh, y)
            for key in ("overall", "foreground", "balanced"):
                acc[mth][key].append(r[key])

    row = {"p": p, "acc_noisy": np.mean(base_noisy)}
    for mth in METHODS:
        for key in ("overall", "foreground", "balanced"):
            row[f"{mth}_{key}"] = np.mean(acc[mth][key])
        row[f"ms_{mth}"] = 1000 * t[mth] / N_IMAGES
    row["lbp_conv_rate"] = lbp_converged / N_IMAGES
    rows.append(row)

# print
print(f"Inference comparison at J = {J}.  "
      f"FG = foreground accuracy (fraction of digit preserved).\n")
hdr = (f"{'p':>5} | {'noisy':>6} || "
       f"{'Gib ov':>6} {'Gib FG':>6} {'Gib bal':>7} || "
       f"{'MF ov':>6} {'MF FG':>6} {'MF bal':>6} || "
       f"{'LBP ov':>6} {'LBP FG':>6} {'LBP bal':>7}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['p']:>5.2f} | {r['acc_noisy']:>6.3f} || "
          f"{r['gibbs_overall']:>6.3f} {r['gibbs_foreground']:>6.3f} "
          f"{r['gibbs_balanced']:>7.3f} || "
          f"{r['mf_overall']:>6.3f} {r['mf_foreground']:>6.3f} "
          f"{r['mf_balanced']:>6.3f} || "
          f"{r['lbp_overall']:>6.3f} {r['lbp_foreground']:>6.3f} "
          f"{r['lbp_balanced']:>7.3f}")

print(f"\n{'p':>5} | {'Gibbs ms':>8} | {'MF ms':>6} | {'LBP ms':>6} | {'LBP conv':>8}")
print("-" * 44)
for r in rows:
    print(f"{r['p']:>5.2f} | {r['ms_gibbs']:>8.1f} | {r['ms_mf']:>6.1f} | "
          f"{r['ms_lbp']:>6.1f} | {r['lbp_conv_rate']:>8.0%}")

with open(RESULTS / "exp02_inference_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# line figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))
ps = [r["p"] for r in rows]
styles = [("gibbs", "Gibbs (MPM)", "o"), ("mf", "Mean-field", "s"),
          ("lbp", "Loopy BP", "^")]

for key, label, marker in styles:
    ax1.plot(ps, [r[f"{key}_overall"] for r in rows], marker=marker, label=label)
ax1.plot(ps, [r["acc_noisy"] for r in rows], marker="x", color="red",
         label="no denoising")
ax1.set_xlabel("flip probability p"); ax1.set_ylabel("overall accuracy")
ax1.set_title("Overall accuracy\n(all high — background-dominated)")
ax1.legend(); ax1.grid(alpha=0.3)

for key, label, marker in styles:
    ax2.plot(ps, [r[f"{key}_foreground"] for r in rows], marker=marker, label=label)
ax2.set_xlabel("flip probability p"); ax2.set_ylabel("foreground accuracy (digit preserved)")
ax2.set_title("Foreground accuracy\n(Gibbs erases most, MF least)")
ax2.legend(); ax2.grid(alpha=0.3)

for key, label, marker in styles:
    ax3.plot(ps, [r[f"ms_{key}"] for r in rows], marker=marker, label=label)
ax3.set_xlabel("flip probability p"); ax3.set_ylabel("runtime per image (ms)")
ax3.set_title("Runtime")
ax3.legend(); ax3.grid(alpha=0.3)

fig.suptitle(f"Inference comparison on the Ising denoising posterior "
             f"(50 MNIST digits, J={J})", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "exp02_inference_comparison.png", dpi=130)

# digit figure
# Qualitative panel with BOTH accuracies per digit, at the hardest noise level.
# This is the figure that answers "why is accuracy high when strokes vanish?":
# read ov (overall) against fg (foreground) under each restored digit.
p_show = 0.20
n_col = 6
fig, axes = plt.subplots(4, n_col, figsize=(11, 8.0))
for col in range(n_col):
    x = clean[col]
    y = flip_noise(x, p=p_show, seed=1000 + col)
    xg = gibbs_denoise(y, J=J, p=p_show, n_sweeps=40, burn_in=15, seed=col)
    xm, _ = meanfield_denoise(y, J=J, p=p_show, n_iters=40)
    xl = lbp_denoise(y, J=J, p=p_show, n_iters=200, damping=0.5)
    rg, rm, rl = (denoising_report(x, xh, y) for xh in (xg, xm, xl))
    panels = [
        (xg, "Gibbs", rg),
        (xm, "mean-field", rm),
        (xl, "Loopy BP", rl),
    ]
    # top row: clean reference
    ax = axes[0, col]
    ax.imshow(x, cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("clean", fontsize=11)
    for row, (img, label, rep) in enumerate(panels, start=1):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ov {rep['overall']:.2f} / fg {rep['foreground']:.2f}",
                     fontsize=8.5)
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
fig.suptitle(f"Same posterior, three inference methods (MNIST, J={J}, p={p_show:.0%})"
             "\nov = overall accuracy (background-dominated, stays high) ; "
             "fg = foreground accuracy (digit strokes only, tracks what the eye sees)",
             fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "exp02_inference_digits.png", dpi=130)
print("\nsaved -> results/exp02_inference_comparison.{csv,png} and exp02_inference_digits.png")