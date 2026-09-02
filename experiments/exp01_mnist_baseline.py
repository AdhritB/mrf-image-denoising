"""Experiment 01 = validate the baseline Ising denoiser on MNIST.

Sweeps the prior strength J over {0.3, 0.5, 0.7, 1.0} for flip
probabilities p in {5%, 10%, 20%}, and reports, for both Gibbs and
mean-field, overall, foreground and balanced pixel accuracy.

Reason: overall pixel accuracy on binarised MNIST is dominated by
the ~85% background, so it stays high even when the denoiser erases the
digit. Foreground/balanced accuracy exposes this. The J-sweep shows the
tension: a strong isotropic Ising prior (large J) cleans the background
but over-smooths thin strokes, erasing the signal, most starkly under
Gibbs, which infers the posterior faithfully. This motivates the Potts,
anisotropic-coupling and pseudo-likelihood extensions.
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

DATA = Path(__file__).resolve().parents[1] / "data" / "mnist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_IMAGES = 50
FLIP_PROBS = [0.05, 0.10, 0.20]
J_VALUES = [0.3, 0.5, 0.7, 1.0]
J_FIGURE = 0.5           # J used for the qualitative digit figure

imgs, _ = load_mnist(DATA, split="test", n=N_IMAGES)
clean = [binarise(im) for im in imgs]

# sweep
rows = []
for J in J_VALUES:
    for p in FLIP_PROBS:
        g = {"overall": [], "foreground": [], "balanced": []}
        m = {"overall": [], "foreground": [], "balanced": []}
        base_noisy, base_allbg = [], []
        for k, x in enumerate(clean):
            y = flip_noise(x, p=p, seed=1000 + k)
            xg = gibbs_denoise(y, J=J, p=p, n_sweeps=40, burn_in=15, seed=k)
            xm, _ = meanfield_denoise(y, J=J, p=p, n_iters=40)
            rg = denoising_report(x, xg, y)
            rm = denoising_report(x, xm, y)
            for key in g:
                g[key].append(rg[key]); m[key].append(rm[key])
            base_noisy.append(rg["baseline_noisy"])
            base_allbg.append(rg["baseline_allbg"])
        rows.append({
            "J": J, "p": p,
            "gibbs_overall": np.mean(g["overall"]),
            "gibbs_foreground": np.mean(g["foreground"]),
            "gibbs_balanced": np.mean(g["balanced"]),
            "mf_overall": np.mean(m["overall"]),
            "mf_foreground": np.mean(m["foreground"]),
            "mf_balanced": np.mean(m["balanced"]),
            "baseline_noisy": np.mean(base_noisy),
            "baseline_allbg": np.mean(base_allbg),
        })

# print
print("Foreground accuracy = fraction of the DIGIT preserved "
      "(1.0 = intact, 0.0 = erased)\n")
hdr = (f"{'J':>4} | {'p':>5} | {'noisy':>6} {'allbg':>6} || "
       f"{'Gib ov':>6} {'Gib FG':>6} {'Gib bal':>7} || "
       f"{'MF ov':>6} {'MF FG':>6} {'MF bal':>6}")
print(hdr); print("-" * len(hdr))
last_J = None
for r in rows:
    if last_J is not None and r["J"] != last_J:
        print("-" * len(hdr))
    last_J = r["J"]
    print(f"{r['J']:>4.1f} | {r['p']:>5.2f} | "
          f"{r['baseline_noisy']:>6.3f} {r['baseline_allbg']:>6.3f} || "
          f"{r['gibbs_overall']:>6.3f} {r['gibbs_foreground']:>6.3f} "
          f"{r['gibbs_balanced']:>7.3f} || "
          f"{r['mf_overall']:>6.3f} {r['mf_foreground']:>6.3f} "
          f"{r['mf_balanced']:>6.3f}")

with open(RESULTS / "exp01_mnist_baseline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# finding figure (J-sweep)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for p in FLIP_PROBS:
    sub = [r for r in rows if r["p"] == p]
    Js = [r["J"] for r in sub]
    ax1.plot(Js, [r["gibbs_foreground"] for r in sub], marker="o",
             label=f"p={p:.0%}")
    ax2.plot(Js, [r["gibbs_overall"] for r in sub], marker="o",
             label=f"p={p:.0%}")
ax1.set_xlabel("prior strength J"); ax1.set_ylabel("foreground accuracy (digit preserved)")
ax1.set_title("Gibbs: strong J erases the digit"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_xlabel("prior strength J"); ax2.set_ylabel("overall accuracy")
ax2.set_title("...but overall accuracy hides it (background-dominated)")
ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle("Effect of Ising prior strength on thin-stroke preservation "
             "(50 MNIST digits)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "exp01_J_sweep.png", dpi=130)

# qualitative figure (at J_FIGURE)
# Panel titles now carry BOTH accuracies so the figure answers its own question:
# overall stays high even where the digit is erased; foreground tracks what the
# eye sees. ov = overall pixel accuracy, fg = foreground (digit-only) accuracy.
p_show = 0.20
fig, axes = plt.subplots(4, 6, figsize=(11, 8.0))
for col in range(6):
    x = clean[col]
    y = flip_noise(x, p=p_show, seed=1000 + col)
    xg = gibbs_denoise(y, J=J_FIGURE, p=p_show, n_sweeps=40, burn_in=15, seed=col)
    xm, _ = meanfield_denoise(y, J=J_FIGURE, p=p_show, n_iters=40)
    rg = denoising_report(x, xg, y)
    rm = denoising_report(x, xm, y)
    ry = denoising_report(x, y, y)
    panels = [
        (x, "clean", None),
        (y, f"noisy p={p_show:.0%}", ry),
        (xg, "Gibbs", rg),
        (xm, "mean-field", rm),
    ]
    for row, (img, label, rep) in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if rep is not None:
            ax.set_title(f"ov {rep['overall']:.2f} / fg {rep['foreground']:.2f}",
                         fontsize=8.5)
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
fig.suptitle(f"Baseline Ising denoiser on MNIST (4-neighbour, J={J_FIGURE}, "
             r"$\beta=\frac{1}{2}\ln\frac{1-p}{p}$)"
             "\nov = overall accuracy (background-dominated) ; "
             "fg = foreground accuracy (digit strokes only)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "exp01_mnist_baseline.png", dpi=130)
print("\nsaved -> results/exp01_mnist_baseline.{csv,png} and exp01_J_sweep.png")