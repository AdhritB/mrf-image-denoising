"""Experiment 04 — recognisability of denoised digits (task-level evaluation).

Overall pixel accuracy on binarised MNIST is background-dominated: a method can
score ~0.95 while erasing the digit entirely (see exp01/exp02). Foreground
accuracy exposes that the strokes are lost, but it is still a pixel count.

This experiment asks the question that actually matters for the task: after
denoising, is the image still a READABLE digit? We answer it the way an OCR
system would by running a simple nearest-neighbour digit classifier on the
restored image and asking whether it is still recognised as the correct class.

Two artefacts:
  1. exp04_recognisability_sweep.png - as the Ising prior strength J increases,
     OVERALL pixel accuracy and RECOGNISABILITY are plotted together. The
     hypothesis (to be confirmed by the run, not assumed) is that they DIVERGE:
     pixel accuracy stays high while recognisability collapses, because a strong
     prior erases the thin strokes the classifier needs.
  2. exp04_error_maps.png - the "complete picture". Each restored digit is shown
     as a colour-coded error map: white = stroke correctly kept, RED = stroke
     wrongly erased, BLUE = background wrongly lit, black = background kept. The
     predicted label is printed under each, so a blank panel reads as an all-red
     stroke that the classifier no longer recognises.

The classifier is a pure-numpy k-nearest-neighbour reader built from CLEAN
training digits; it is deliberately simple and dependency-free. It is applied
identically to every image, so what matters is the RELATIVE readability of clean
vs noisy vs each restoration.
"""
import sys, csv
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

N_REF   = 3000          # clean training digits used to build the reader
N_TEST  = 50            # test digits to denoise + classify
K_NN    = 3             # neighbours in the reader
J_VALUES = [0.3, 0.5, 0.7, 1.0]
J_FIXED  = 0.5          # for the method comparison + error maps
P_HARD   = 0.20         # the hard noise level where erasure is starkest

# the reader
# k-NN on binarised {-1,+1} digits. For unit-magnitude spin vectors, Euclidean
# nearest-neighbour is equivalent to LARGEST dot product, so one matmul ranks all
# references. Pure numpy, deterministic, no training.
ref_imgs, ref_labels = load_mnist(DATA, split="train", n=N_REF)
ref_flat = np.stack([binarise(im).ravel() for im in ref_imgs]).astype(np.float32)
ref_labels = np.asarray(ref_labels)

def recognise(images_spins):
    """Classify a list of {-1,+1} images; return predicted labels (N,)."""
    X = np.stack([im.ravel() for im in images_spins]).astype(np.float32)
    sims = X @ ref_flat.T                     # (N, N_REF), larger = closer
    topk = np.argpartition(-sims, kth=K_NN - 1, axis=1)[:, :K_NN]
    preds = np.empty(len(X), dtype=int)
    for i in range(len(X)):
        preds[i] = np.bincount(ref_labels[topk[i]], minlength=10).argmax()
    return preds

def accuracy(preds, labels):
    return float(np.mean(preds == np.asarray(labels)))

# test data
test_imgs, test_labels = load_mnist(DATA, split="test", n=N_TEST)
clean = [binarise(im) for im in test_imgs]
test_labels = np.asarray(test_labels)

# sanity: how readable are the CLEAN test digits themselves? (upper bound)
clean_recog = accuracy(recognise(clean), test_labels)
print(f"Reader: {K_NN}-NN over {N_REF} clean training digits.")
print(f"Clean test digits are recognised at {clean_recog:.1%} "
      f"(this is the ceiling for any denoiser).\n")

# J sweep
# Gibbs is the faithful sampler that most reflects the prior, so it shows the
# erasure most clearly. At fixed hard noise, sweep J and record BOTH the overall
# pixel accuracy and the recognisability, to see whether they diverge.
print(f"J-sweep at p={P_HARD:.0%} (Gibbs). Does pixel accuracy track readability?\n")
print(f"{'J':>4} | {'overall px':>10} | {'foreground':>10} | {'recognised':>10}")
print("-" * 46)
sweep = []
for J in J_VALUES:
    outs, ov, fg = [], [], []
    for k, x in enumerate(clean):
        y = flip_noise(x, p=P_HARD, seed=1000 + k)
        xg = gibbs_denoise(y, J=J, p=P_HARD, n_sweeps=40, burn_in=15, seed=k)
        outs.append(xg)
        r = denoising_report(x, xg, y)
        ov.append(r["overall"]); fg.append(r["foreground"])
    rec = accuracy(recognise(outs), test_labels)
    row = {"J": J, "overall": np.mean(ov), "foreground": np.mean(fg),
           "recognised": rec}
    sweep.append(row)
    print(f"{J:>4.1f} | {row['overall']:>10.3f} | {row['foreground']:>10.3f} | "
          f"{row['recognised']:>10.3f}")

with open(RESULTS / "exp04_recognisability.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
    w.writeheader(); w.writerows(sweep)

# method table
# At the adopted J, compare readability of clean / noisy / each method, next to
# the pixel metrics, so all three ways of measuring sit side by side.
print(f"\nMethod comparison at J={J_FIXED}, p={P_HARD:.0%}:")
print(f"{'source':>12} | {'overall px':>10} | {'foreground':>10} | {'recognised':>10}")
print("-" * 52)
noisy_imgs, g_imgs, m_imgs, l_imgs = [], [], [], []
for k, x in enumerate(clean):
    y = flip_noise(x, p=P_HARD, seed=1000 + k)
    noisy_imgs.append(y)
    g_imgs.append(gibbs_denoise(y, J=J_FIXED, p=P_HARD, n_sweeps=40, burn_in=15, seed=k))
    m_imgs.append(meanfield_denoise(y, J=J_FIXED, p=P_HARD, n_iters=40)[0])
    l_imgs.append(lbp_denoise(y, J=J_FIXED, p=P_HARD, n_iters=200, damping=0.5))

def report_block(name, outs):
    ov = np.mean([denoising_report(clean[k], outs[k], noisy_imgs[k])["overall"]
                  for k in range(N_TEST)])
    fg = np.mean([denoising_report(clean[k], outs[k], noisy_imgs[k])["foreground"]
                  for k in range(N_TEST)])
    rec = accuracy(recognise(outs), test_labels)
    print(f"{name:>12} | {ov:>10.3f} | {fg:>10.3f} | {rec:>10.3f}")
    return rec

print(f"{'clean':>12} | {'1.000':>10} | {'1.000':>10} | {clean_recog:>10.3f}")
report_block("noisy", noisy_imgs)
report_block("Gibbs", g_imgs)
report_block("mean-field", m_imgs)
report_block("Loopy BP", l_imgs)

# sweep figure
fig, ax = plt.subplots(figsize=(7.2, 4.6))
Js = [r["J"] for r in sweep]
ax.plot(Js, [r["overall"] for r in sweep], marker="o", color="#3b6fb0",
        label="overall pixel accuracy")
ax.plot(Js, [r["foreground"] for r in sweep], marker="s", color="#e0873a",
        label="foreground accuracy (digit pixels)")
ax.plot(Js, [r["recognised"] for r in sweep], marker="^", color="#3f9b52",
        label="recognisability (still reads as the digit)")
ax.axhline(clean_recog, ls="--", color="#3f9b52", alpha=0.5,
           label=f"clean-digit ceiling ({clean_recog:.0%})")
ax.set_xlabel("Ising prior strength J")
ax.set_ylabel("accuracy")
ax.set_title(f"As the prior strengthens, pixels stay 'correct' but the digit\n"
             f"stops being readable (Gibbs, p={P_HARD:.0%}, {N_TEST} digits)")
ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=0.3)
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(RESULTS / "exp04_recognisability_sweep.png", dpi=130)

# error maps 
# The "complete picture": subtract restored from clean and colour the outcome.
#   white  = stroke correctly kept (true positive)
#   RED    = stroke wrongly erased (false negative)   <- what the eye/OCR loses
#   BLUE   = background wrongly lit (false positive)
#   black  = background correctly kept (true negative)
def error_map(x, xh):
    h, w = x.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    tp = (x == 1) & (xh == 1)
    fn = (x == 1) & (xh == -1)
    fp = (x == -1) & (xh == 1)
    rgb[tp] = (1, 1, 1)
    rgb[fn] = (0.85, 0.12, 0.12)   # red = erased digit
    rgb[fp] = (0.20, 0.35, 0.90)   # blue = spurious ink
    return rgb

n_show = 6
g_preds = recognise([g_imgs[c] for c in range(n_show)])
fig, axes = plt.subplots(3, n_show, figsize=(11, 6.0))
for c in range(n_show):
    x, xh = clean[c], g_imgs[c]
    true_lab = test_labels[c]
    pred_lab = g_preds[c]
    fg = denoising_report(x, xh, noisy_imgs[c])["foreground"]
    rows = [
        (x, None, f"clean: {true_lab}"),
        (xh, None, "Gibbs output"),
        (error_map(x, xh), "rgb",
         f"reads as '{pred_lab}' " + ("\u2713" if pred_lab == true_lab else "\u2717")
         + f"\nfg {fg:.2f}"),
    ]
    for r, (im, mode, title) in enumerate(rows):
        ax = axes[r, c]
        if mode == "rgb":
            ax.imshow(im)
        else:
            ax.imshow(im, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if c == 0:
            ax.set_ylabel(["clean", "restored", "error map"][r], fontsize=11)
fig.suptitle("What the denoiser loses (Gibbs, J=%.1f, p=%d%%).  "
             "Error map: white kept \u00b7 RED erased digit \u00b7 blue spurious ink.\n"
             "A digit that reads high on pixel accuracy but comes back as red is "
             "erased, not denoised." % (J_FIXED, int(P_HARD * 100)), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RESULTS / "exp04_error_maps.png", dpi=130)

print("\nsaved -> results/exp04_recognisability.csv, "
      "exp04_recognisability_sweep.png, exp04_error_maps.png")
