"""Experiment 03 — Potts-model grayscale denoising (core extension 1).

Evaluates the q-state Potts Gibbs denoiser on Set12 and a BSD68 subset under
additive Gaussian noise at sigma in {15, 25} (0-255 scale), sweeping the
coupling strength J and reporting BOTH PSNR and SSIM.

Why both metrics: PSNR rewards flattening noisy regions, so it can favour a
coupling strength that visibly destroys texture. SSIM is structure-sensitive.
Whether the two metrics agree is an empirical question, answered by the sweep
below - the figure titles are generated FROM THE DATA, so they always state
what was actually observed rather than what was expected.

Set SWEEP_J to a single value for a production run once J is chosen.
"""
import sys, time, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_folder, gaussian_noise, psnr
from potts import potts_gibbs_denoise

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

Q = 32
SWEEP_J = [2.0]
SIGMAS = [15 / 255, 25 / 255]
N_SET12 = None      # None -> ALL 12 (includes Barbara, the texture-critical image)
N_BSD = 10          # None -> all 68
J_ADOPTED = 2.0    # the chosen coupling; used for the qualitative figure
J_FIGURE = J_ADOPTED

# Load images together with their file paths so the run is pinned to an
# explicit manifest. Reproducibility depends on selecting the SAME images and
# giving each the SAME noise every run; printing the manifest makes any drift
# (a re-extracted folder, a renamed file) visible immediately instead of
# showing up later as an unexplained 0.03 dB wobble.
set12_imgs, set12_paths = load_folder(ROOT / "data" / "set12", n=N_SET12,
                                      return_paths=True)
bsd_imgs, bsd_paths = load_folder(ROOT / "data" / "bsd68", n=N_BSD,
                                  return_paths=True)
datasets = {"Set12": set12_imgs, "BSD68subset": bsd_imgs}
manifest = {"Set12": set12_paths, "BSD68subset": bsd_paths}

for name, paths in manifest.items():
    names = [p.name for p in paths]
    print(f"{name}: {len(names)} images -> {names[0]} ... {names[-1]}")
# Guard the subset that actually wobbled: if this count is not what we expect,
# the folder changed and the numbers are not comparable to earlier runs.
if N_BSD is not None:
    assert len(bsd_paths) == N_BSD, (
        f"expected {N_BSD} BSD68 images, loaded {len(bsd_paths)} -- "
        "check data/bsd68 contents")
if len(sys.argv) > 1:
    keep = sys.argv[1]
    datasets = {k: v for k, v in datasets.items() if k == keep}
    if not datasets:
        raise SystemExit(f"unknown dataset {keep!r}")

def noise_seed(path):
    """Deterministic noise seed derived from the image FILENAME, not its
    position in the list. Two runs give an image the same noise even if the
    folder's sort order or subset size changes; this is what makes the BSD68
    numbers reproduce exactly, as the Set12 numbers already did.
    """
    import zlib
    return 100 + (zlib.crc32(path.name.encode()) & 0xFFFF)


rows = []
for name, images in datasets.items():
    paths = manifest[name]
    for J in SWEEP_J:
        for sigma in SIGMAS:
            p_noisy, p_den, s_noisy, s_den = [], [], [], []
            t0 = time.time()
            for k, (img, path) in enumerate(zip(images, paths)):
                y = gaussian_noise(img, sigma=sigma, seed=noise_seed(path))
                xh = potts_gibbs_denoise(y, sigma=sigma, q=Q, J=J,
                                         n_sweeps=25, burn_in=10,
                                         seed=noise_seed(path))
                p_noisy.append(psnr(img, y)); p_den.append(psnr(img, xh))
                s_noisy.append(ssim(img, y, data_range=1.0))
                s_den.append(ssim(img, xh, data_range=1.0))
            rows.append({
                "dataset": name, "J": J, "sigma255": round(sigma * 255),
                "psnr_noisy": np.mean(p_noisy), "psnr_potts": np.mean(p_den),
                "psnr_gain": np.mean(p_den) - np.mean(p_noisy),
                "ssim_noisy": np.mean(s_noisy), "ssim_potts": np.mean(s_den),
                "ssim_gain": np.mean(s_den) - np.mean(s_noisy),
                "sec_per_img": (time.time() - t0) / len(images),
            })
            r = rows[-1]
            print(f"{name:12s} J={J:<4} sigma={r['sigma255']:>2}: "
                  f"PSNR {r['psnr_noisy']:.2f} -> {r['psnr_potts']:.2f} "
                  f"({r['psnr_gain']:+.2f})   "
                  f"SSIM {r['ssim_noisy']:.3f} -> {r['ssim_potts']:.3f} "
                  f"({r['ssim_gain']:+.3f})  [{r['sec_per_img']:.1f} s/img]")

# summary
# A "disagreement" between metrics is only meaningful if the difference is larger
# than the noise in the curve. Near the optimum these curves are very flat, so a
# bare argmax can flag a disagreement worth 0.04 dB. Treat any J whose score is
# within TOL of the best as tied with it.
TOL = {"psnr_potts": 0.10, "ssim_potts": 0.005}   # dB, SSIM units


def describe(sub, key):
    """Return (best_J, tied_set, plain-English description of the curve).

    tied_set = all J whose score is within TOL[key] of the maximum, i.e. the
    values that cannot be meaningfully distinguished on this metric.
    """
    sub = sorted(sub, key=lambda r: r["J"])
    vals = [r[key] for r in sub]
    Js = [r["J"] for r in sub]
    best_i = int(np.argmax(vals))
    best_J, best_v = Js[best_i], vals[best_i]
    tol = TOL[key]
    tied = {J for J, v in zip(Js, vals) if best_v - v <= tol}

    if len(tied) == len(Js):
        shape = f"is flat across the whole range (all J within {tol:g} of best)"
    elif len(tied) > 1:
        shape = (f"is maximal at J={best_J}, but J\u2208{sorted(tied)} are tied "
                 f"within {tol:g}")
    elif best_J == Js[-1]:
        shape = f"rises monotonically to J={Js[-1]}"
    elif best_J == Js[0]:
        shape = f"is highest at the smallest J tested (J={Js[0]})"
    else:
        shape = f"peaks at J={best_J} then falls"
    return best_J, tied, shape

summary = {}
if len(SWEEP_J) > 1:
    print("\n=== J sweep: do PSNR and SSIM choose the same J? ===")
    for name in sorted({r["dataset"] for r in rows}):
        for s in sorted({r["sigma255"] for r in rows}):
            sub = [r for r in rows if r["dataset"] == name and r["sigma255"] == s]
            if not sub:
                continue
            bp, tp, sp = describe(sub, "psnr_potts")
            bs, ts, ss = describe(sub, "ssim_potts")
            summary[(name, s)] = (bp, tp, sp, bs, ts, ss)
            overlap = tp & ts
            if bp == bs:
                verdict = "AGREE"
            elif overlap:
                verdict = (f"AGREE within tolerance (both admit J\u2208"
                           f"{sorted(overlap)})")
            else:
                verdict = "DISAGREE (beyond tolerance)"
            print(f"  {name}, sigma={s}:")
            print(f"      PSNR {sp}")
            print(f"      SSIM {ss}")
            print(f"      -> {verdict}")
    # How many conditions does each J satisfy on BOTH metrics simultaneously?
    from collections import Counter
    tally = Counter()
    for v in summary.values():
        for J in (v[1] & v[4]):
            tally[J] += 1
    n_cond = len(summary)
    if tally:
        print(f"\n  Admissible on BOTH metrics (within tolerance), by condition "
              f"[{n_cond} conditions total]:")
        for J in sorted(tally, key=lambda j: (-tally[j], j)):
            print(f"      J={J}: {tally[J]}/{n_cond}")
        best = max(tally.values())
        winners = sorted(J for J in tally if tally[J] == best)
        if best == n_cond:
            print(f"  -> J\u2208{winners} is admissible in every condition tested.")
        else:
            print(f"  -> No single J is admissible everywhere; J\u2208{winners} "
                  f"comes closest ({best}/{n_cond}). This is itself evidence "
                  f"that one global coupling cannot suit all images.")
    print("\nNote: near the optimum these curves are flat, so a bare argmax can")
    print("flag a 'disagreement' worth a few hundredths of a dB. Verdicts above")
    print("use a tolerance of "
          f"{TOL['psnr_potts']:g} dB / {TOL['ssim_potts']:g} SSIM.")
    print("Both metrics are full-reference and can still favour smoothing that")
    print("visibly posterises texture -- inspect the qualitative figure before")
    print("choosing J; do not select on PSNR/SSIM alone.")

# Overwrite (not append): the CSV is the table for THIS run, not a growing log.
# Running datasets separately therefore updates only their rows; to keep both
# datasets in one CSV, run without a CLI argument.
csv_path = RESULTS / f"exp03_potts_grayscale{'_'+sys.argv[1] if len(sys.argv)>1 else ''}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

#PSNR/SSIM vs J (titles from data)
if len(SWEEP_J) > 1 and "Set12" in datasets:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in sorted({r["sigma255"] for r in rows if r["dataset"] == "Set12"}):
        sub = sorted([r for r in rows if r["dataset"] == "Set12"
                      and r["sigma255"] == s], key=lambda r: r["J"])
        Js = [r["J"] for r in sub]
        axes[0].plot(Js, [r["psnr_potts"] for r in sub], marker="o", label=f"sigma={s}")
        axes[1].plot(Js, [r["ssim_potts"] for r in sub], marker="o", label=f"sigma={s}")

    keys = [k for k in summary if k[0] == "Set12"]
    if keys:
        # Summarise EACH metric across BOTH sigma conditions rather than borrowing
        # a single curve's shape (a panel shows two sigma lines that can differ).
        def panel_summary(metric_idx_best, metric_idx_tied):
            bests = {summary[k][metric_idx_best] for k in keys}
            tied = None
            for k in keys:
                t = summary[k][metric_idx_tied]
                tied = t if tied is None else (tied | t)
            if len(bests) == 1:
                return f"best at J={bests.pop()} (both \u03c3)"
            return f"best J varies with \u03c3 ({', '.join(f'J={b}' for b in sorted(bests))})"

        # summary tuple: (bp, tp, sp, bs, ts, ss)
        ptitle = "PSNR: " + panel_summary(0, 1)
        stitle = "SSIM: " + panel_summary(3, 4)
        common = None
        for k in keys:
            adm = summary[k][1] & summary[k][4]
            common = adm if common is None else (common & adm)
        if common:
            sup = (f"Potts denoiser: PSNR and SSIM both admit J\u2208{sorted(common)} "
                   f"within tolerance across \u03c3 (q={Q}, Set12)")
        else:
            sup = ("Potts denoiser: PSNR and SSIM admit no common J within "
                   f"tolerance in every condition (q={Q}, Set12)")
    else:
        ptitle, stitle, sup = "PSNR vs J", "SSIM vs J", f"Potts denoiser (q={Q}, Set12)"

    axes[0].set_xlabel("coupling J"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title(ptitle); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("coupling J"); axes[1].set_ylabel("SSIM")
    axes[1].set_title(stitle); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "exp03_J_sweep.png", dpi=130)

# qualitative figure: J comparison on ONE image
# Only regenerate when actually sweeping: with a single J this figure would be
# a 3-panel strip that overwrites the informative multi-J comparison.
if len(SWEEP_J) > 1 and "Set12" in datasets and len(datasets["Set12"]) > 8:
    sigma = 25 / 255
    img = datasets["Set12"][8]
    seed_b = noise_seed(manifest["Set12"][8])
    y = gaussian_noise(img, sigma=sigma, seed=seed_b)
    panels = [(img, "clean", None, None),
              (y, "noisy", psnr(img, y), ssim(img, y, data_range=1.0))]
    for J in SWEEP_J:
        xh = potts_gibbs_denoise(y, sigma=sigma, q=Q, J=J,
                                 n_sweeps=25, burn_in=10, seed=seed_b)
        panels.append((xh, f"J={J}", psnr(img, xh),
                       ssim(img, xh, data_range=1.0)))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6))
    for ax, (im, lab, p, sv) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab if p is None else f"{lab}\n{p:.1f} dB / {sv:.3f}",
                     fontsize=9)
    fig.suptitle("Set12 #9 (Barbara), sigma=25 — texture loss as J increases "
                 "(titles: PSNR / SSIM)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "exp03_J_visual_barbara.png", dpi=130)

#standard 3-image figure
if "Set12" in datasets:
    sigma = 25 / 255
    imgs = datasets["Set12"]
    picks = [2, 5, 8] if len(imgs) > 8 else list(range(min(3, len(imgs))))
    fig, axes = plt.subplots(3, len(picks), figsize=(3.5 * len(picks), 10.5))
    if len(picks) == 1:
        axes = axes.reshape(3, 1)
    for col, idx in enumerate(picks):
        img = imgs[idx]
        seed_i = noise_seed(manifest["Set12"][idx])
        y = gaussian_noise(img, sigma=sigma, seed=seed_i)
        xh = potts_gibbs_denoise(y, sigma=sigma, q=Q, J=J_FIGURE,
                                 n_sweeps=25, burn_in=10, seed=seed_i)
        rowspec = [
            (img, "clean"),
            (y, f"noisy ({psnr(img,y):.1f} dB / {ssim(img,y,data_range=1.0):.3f})"),
            (xh, f"Potts ({psnr(img,xh):.1f} dB / {ssim(img,xh,data_range=1.0):.3f})"),
        ]
        for row, (im, label) in enumerate(rowspec):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label.split(" (")[0], fontsize=12)
            ax.set_title(label if row else f"Set12 #{idx+1}", fontsize=9)
    fig.suptitle(f"Potts denoiser (q={Q}, J={J_FIGURE}), Gaussian sigma=25 "
                 f"— titles show PSNR / SSIM", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "exp03_potts_grayscale.png", dpi=130)

print("\nsaved -> results/exp03_potts_grayscale.{csv,png}"
      + (", exp03_J_sweep.png" if len(SWEEP_J) > 1 else "")
      + (", exp03_J_visual_barbara.png"
         if (len(SWEEP_J) > 1 and "Set12" in datasets
             and len(datasets["Set12"]) > 8) else ""))