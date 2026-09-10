"""Regenerate the report's plot figures from the result CSVs.

Why this exists: the figures in the submitted report were unreadable at 100% zoom,
and several had overlapping legends, colliding tick labels, or titles that did not
match what the axes plotted. Fixing that is a layout change only. Every number
plotted here is read from the CSV that the corresponding experiment wrote, so
running this script cannot change a single reported value.

Regenerated from CSV (instant):
    exp03_J_sweep.png
    exp04_recognisability_sweep.png
    exp05_anisotropic_sweep_mmse.png
    exp05_anisotropic_sweep_mpm.png
    exp06_bm3d_benchmark.png
    exp07_pseudolikelihood.png   (left panel from CSV; right panel needs the
                                  stability trace, which the CSV does not store,
                                  so it is recomputed for image 09 only)
    exp03_J_visual_barbara.png   (relaid out as two rows of three so each panel
                                  is legible; four reconstructions of one image
                                  are recomputed, with the same seed as exp03)

exp04_error_maps.png is not regenerated here; it comes from exp04 itself.

Run from the project root:
    python experiments/replot_figures.py

Pass --csv-only to skip the two steps that recompute anything and redraw only
the figures that come straight from the CSVs.
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The clean-digit ceiling is printed by exp04 but not stored in its CSV.
CLEAN_RECOG = 0.92
Q = 32
J_ADOPTED = 2.0
TOL = {"psnr_potts": 0.10, "ssim_potts": 0.005}


def load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  skipped, missing: {name}")
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# exp03
def fig_exp03():
    rows = load("exp03_potts_grayscale.csv")
    if rows is None:
        return
    for r in rows:
        r["J"] = float(r["J"]); r["sigma255"] = int(r["sigma255"])
        r["psnr_potts"] = float(r["psnr_potts"]); r["ssim_potts"] = float(r["ssim_potts"])
    Js_all = sorted({r["J"] for r in rows})
    if len(Js_all) < 2:
        print("  exp03 CSV holds a single J; rerun the sweep first")
        return

    def best_and_tied(sub, key):
        sub = sorted(sub, key=lambda r: r["J"])
        vals = [r[key] for r in sub]; Js = [r["J"] for r in sub]
        bi = int(np.argmax(vals)); best = vals[bi]
        return Js[bi], {J for J, v in zip(Js, vals) if best - v <= TOL[key]}

    conds = sorted({(r["dataset"], r["sigma255"]) for r in rows})
    summary = {}
    for c in conds:
        sub = [r for r in rows if (r["dataset"], r["sigma255"]) == c]
        bp, tp = best_and_tied(sub, "psnr_potts")
        bs, ts = best_and_tied(sub, "ssim_potts")
        summary[c] = (bp, tp, bs, ts)

    n_agree = sum(1 for c in conds if summary[c][1] & summary[c][3])
    tally = {}
    for c in conds:
        for J in (summary[c][1] & summary[c][3]):
            tally[J] = tally.get(J, 0) + 1
    top = max(tally, key=lambda j: (tally[j], -j)) if tally else None

    keys = [c for c in conds if c[0] == "Set12"]

    def peaks(idx):
        per = {c[1]: summary[c][idx] for c in sorted(keys, key=lambda k: k[1])}
        if len(set(per.values())) == 1:
            return f"peak at J={next(iter(per.values()))} for both $\\sigma$"
        return "peak at " + ", ".join(f"J={v} ($\\sigma$={s})" for s, v in per.items())

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    for s in sorted({c[1] for c in keys}):
        sub = sorted([r for r in rows if r["dataset"] == "Set12"
                      and r["sigma255"] == s], key=lambda r: r["J"])
        Js = [r["J"] for r in sub]
        axes[0].plot(Js, [r["psnr_potts"] for r in sub], marker="o",
                     label=f"$\\sigma$={s}")
        axes[1].plot(Js, [r["ssim_potts"] for r in sub], marker="o",
                     label=f"$\\sigma$={s}")
    axes[0].set_xlabel("coupling J"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR against coupling J\n" + peaks(0))
    axes[1].set_xlabel("coupling J"); axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM against coupling J\n" + peaks(2))
    for ax in axes:
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # Data-generated suptitle. It states the two things the sweep actually
    # establishes, and neither is asserted in advance.
    sup = (f"Potts coupling sweep, q={Q}: the two metrics agree in "
           f"{n_agree} of {len(conds)} conditions")
    if top is not None:
        sup += f"; J={top} is admissible in {tally[top]} of {len(conds)}"
    fig.suptitle(sup, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    fig.savefig(RESULTS / "exp03_J_sweep.png")
    plt.close(fig)
    print("  wrote exp03_J_sweep.png")


# exp03 Barbara
def fig_exp03_barbara(sweep=(1.0, 1.75, 2.0, 2.5)):
    """Redraw the qualitative Barbara strip as two rows of three.

    Six panels in a single row leaves each image about an inch wide on the page,
    which is why the scarf and tablecloth texture could not be judged without
    zooming. A 2x3 grid roughly doubles each panel. This recomputes four
    reconstructions of one image with the same seed and parameters as exp03, so
    the panel scores are identical to that run; it takes a couple of minutes.
    """
    import zlib
    sys.path.insert(0, str(ROOT / "src"))
    from datasets import load_folder, gaussian_noise, psnr
    from potts import potts_gibbs_denoise
    from skimage.metrics import structural_similarity as ssim

    imgs, paths = load_folder(ROOT / "data" / "set12", n=None, return_paths=True)
    if len(imgs) <= 8:
        print("  skipped Barbara figure: Set12 image 9 not found")
        return
    img = imgs[8]
    seed = 100 + (zlib.crc32(paths[8].name.encode()) & 0xFFFF)
    sigma = 25 / 255.0
    y = gaussian_noise(img, sigma=sigma, seed=seed)

    panels = [(img, "clean", None, None),
              (y, "noisy", psnr(img, y), ssim(img, y, data_range=1.0))]
    for J in sweep:
        xh = potts_gibbs_denoise(y, sigma=sigma, q=Q, J=J, n_sweeps=25,
                                 burn_in=10, seed=seed)
        panels.append((xh, f"J={J}", psnr(img, xh),
                       ssim(img, xh, data_range=1.0)))
        print(f"    J={J}: {panels[-1][2]:.2f} dB / {panels[-1][3]:.3f}")

    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.35 * nrow))
    for ax, (im, lab, pv, sv) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab if pv is None
                     else f"{lab}\nPSNR {pv:.1f} dB, SSIM {sv:.3f}", fontsize=13)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Set12 #9 (Barbara), $\\sigma$=25: texture loss as J increases",
                 fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(RESULTS / "exp03_J_visual_barbara.png")
    plt.close(fig)
    print("  wrote exp03_J_visual_barbara.png")

# exp04
def fig_exp04():
    rows = load("exp04_recognisability.csv")
    if rows is None:
        return
    Js = [float(r["J"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(Js, [float(r["overall"]) for r in rows], marker="o", color="#3b6fb0",
            label="overall pixel accuracy")
    ax.plot(Js, [float(r["foreground"]) for r in rows], marker="s", color="#e0873a",
            label="foreground accuracy (digit pixels)")
    ax.plot(Js, [float(r["recognised"]) for r in rows], marker="^", color="#3f9b52",
            label="recognisability (still reads as the digit)")
    ax.axhline(CLEAN_RECOG, ls="--", color="#3f9b52", alpha=0.7,
               label=f"clean-digit ceiling: {CLEAN_RECOG:.0%}")
    ax.set_xlabel("Ising prior strength J")
    ax.set_ylabel("accuracy")
    ax.set_title("As the prior strengthens, pixels stay 'correct' but the digit\n"
                 "stops being readable (Gibbs, p=20%, 50 digits)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    # Legend below the axes: inside the axes it sat on top of the crossing curves.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp04_recognisability_sweep.png")
    plt.close(fig)
    print("  wrote exp04_recognisability_sweep.png")


# exp05
def fig_exp05(estimator):
    rows = load(f"exp05_anisotropic_{estimator}.csv")
    if rows is None:
        return
    for r in rows:
        r["psnr"] = float(r["psnr"]); r["ssim"] = float(r["ssim"])
        r["kappa"] = float(r["kappa"]) if r["kappa"] not in ("", "nan") else float("nan")
    ks = sorted({r["kappa"] for r in rows if r["mode"] == "perona_malik"})
    iso = [r for r in rows if r["mode"] == "isotropic"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, metric, ylab, title in [
            (ax1, "psnr", "PSNR (dB)", "Set12 aggregate PSNR"),
            (ax2, "ssim", "SSIM", "Set12 aggregate SSIM")]:
        pm = [next(r[metric] for r in rows
                   if r["mode"] == "perona_malik" and r["kappa"] == k) for k in ks]
        co = [next(r[metric] for r in rows
                   if r["mode"] == "cohen" and r["kappa"] == k) for k in ks]
        ax.axhline(iso[metric], ls="--", color="black", label="isotropic (baseline)")
        ax.plot(ks, pm, marker="o", color="#3f9b52",
                label="Perona-Malik (edge-preserving)")
        ax.plot(ks, co, marker="s", color="#c0392b",
                label="Cohen inverted (ablation)")
        ax.set_xlabel("$\\kappa$ (gradient scale)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.grid(alpha=0.3)
        # lower right is the empty corner in both panels; the default sat on the
        # leftmost data point.
        ax.legend(loc="lower right", fontsize=11)
    fig.suptitle(f"Adaptive vs global Potts coupling, {estimator.upper()} estimator "
                 f"(Set12, $\\sigma$=25, $J_0$={J_ADOPTED})", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.savefig(RESULTS / f"exp05_anisotropic_sweep_{estimator}.png")
    plt.close(fig)
    print(f"  wrote exp05_anisotropic_sweep_{estimator}.png")


# exp06
def fig_exp06():
    rows = load("exp06_bm3d_benchmark.csv")
    if rows is None:
        return
    # "BSD68subset" was long enough to collide with its neighbour's label.
    labels = [f"{r['dataset'].replace('BSD68subset', 'BSD68')}\n"
              f"$\\sigma$={r['sigma255']}" for r in rows]
    x = np.arange(len(rows)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.6))
    for ax, kp, kb, ylab, title in [
            (ax1, "psnr_potts", "psnr_bm3d", "PSNR (dB)", "PSNR: Potts vs BM3D"),
            (ax2, "ssim_potts", "ssim_bm3d", "SSIM", "SSIM: Potts vs BM3D")]:
        ax.bar(x - w/2, [float(r[kp]) for r in rows], w, color="#3b6fb0",
               label="Potts (this work)")
        ax.bar(x + w/2, [float(r[kb]) for r in rows], w, color="#e0873a",
               label="BM3D")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    # Bars fill the axes from zero, so there is no interior space for a legend.
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Interpretable Potts MRF vs BM3D on identical noisy images",
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.22)
    fig.savefig(RESULTS / "exp06_bm3d_benchmark.png")
    plt.close(fig)
    print("  wrote exp06_bm3d_benchmark.png")


# exp07
def fig_exp07(trace=None):
    rows = load("exp07_pseudolikelihood.csv")
    if rows is None:
        return
    names = [r["image"] for r in rows]
    clean = [float(r["J_clean"]) for r in rows]
    noisy = [float(r["J_noisy"]) for r in rows]
    itr = [float(r["J_iter_final"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = np.arange(len(rows)); w = 0.27
    ax1.bar(x - w, clean, w, label="fitted to clean image", color="#3f9b52")
    ax1.bar(x,     noisy, w, label="fitted to noisy image", color="#3b6fb0")
    ax1.bar(x + w, itr,   w, label="fitted iteratively (final)", color="#c0392b")
    ax1.axhline(J_ADOPTED, ls="--", color="black",
                label=f"hand-tuned input J={J_ADOPTED}")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11, rotation=45)
    ax1.set_xlabel("Set12 image")
    # Short label: the long form collided with the neighbouring panel's title.
    ax1.set_ylabel("fitted J")
    ax1.set_title("Fitted vs hand-tuned coupling")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_ylim(0, max(itr) * 1.45)
    ax1.grid(axis="y", alpha=0.3)

    if trace:
        name, traj = trace
        ax2.plot(range(1, len(traj) + 1), traj, marker="o", color="#c0392b")
        ax2.axhline(J_ADOPTED, ls="--", color="black",
                    label=f"hand-tuned J={J_ADOPTED}")
        ax2.set_xlabel("re-estimation cycle"); ax2.set_ylabel("fitted J")
        ax2.set_title(f"Stability of iterated fitting (image {name})")
        ax2.legend(fontsize=11); ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "stability trace not recomputed", ha="center")

    fig.suptitle("Pseudo-likelihood learning of the Potts coupling\n"
                 "(the vertical axis is a coupling fitted to data, "
                 "not one supplied to the denoiser)", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.76)
    fig.savefig(RESULTS / "exp07_pseudolikelihood.png")
    plt.close(fig)
    print("  wrote exp07_pseudolikelihood.png")


def exp07_trace():
    """Recompute the 8-cycle stability trace for image 09 only.

    Identical code path, seed and parameters to exp07, so the values match its
    run exactly; only one image is processed, which takes a couple of minutes
    rather than rerunning the whole experiment.
    """
    import zlib
    sys.path.insert(0, str(ROOT / "src"))
    from datasets import load_folder, gaussian_noise
    from potts import potts_gibbs_denoise, quantise
    from pseudolikelihood import estimate_J

    imgs, paths = load_folder(ROOT / "data" / "set12", n=None, return_paths=True)
    pick = [i for i, p in enumerate(paths) if p.stem in ("09", "9")]
    if not pick:
        return None
    i = pick[0]
    path = paths[i]
    seed = 100 + (zlib.crc32(path.name.encode()) & 0xFFFF)
    y = gaussian_noise(imgs[i], sigma=25 / 255.0, seed=seed)
    x = quantise(y, Q)
    traj = []
    for c in range(8):
        Jhat, _ = estimate_J(x, q=Q, bracket=(0.0, 20.0))
        traj.append(Jhat)
        xh = potts_gibbs_denoise(y, sigma=25 / 255.0, q=Q, J=Jhat, n_sweeps=15,
                                 burn_in=6, seed=seed, estimator="mpm")
        x = quantise(xh, Q)
        print(f"    cycle {c+1}: J = {Jhat:.3f}")
    return (path.stem, traj)


if __name__ == "__main__":
    print("Regenerating figures from results/*.csv")
    fig_exp03()
    fig_exp04()
    fig_exp05("mmse")
    fig_exp05("mpm")
    fig_exp06()
    if "--csv-only" in sys.argv:
        fig_exp07(None)
    else:
        print("  recomputing exp07 stability trace for image 09 "
              "(a couple of minutes)")
        fig_exp07(exp07_trace())
        print("  recomputing the Barbara panels at 2x3 (a couple of minutes)")
        fig_exp03_barbara()
    print("done")