"""Experiment 05 — anisotropic (spatially adaptive) Potts coupling (extension 2).

Motivation (from exp03): a single global coupling J cannot serve both smooth
regions (which want strong coupling to remove noise) and textured regions (which
want weak coupling to preserve detail). Barbara loses its scarf/tablecloth
texture as J rises even while the Set12 aggregate still improves. Anisotropic
coupling makes J depend on LOCAL gradient: strong in flat regions, weak across
edges/texture.

This experiment tests rather than assumes whether that helps on REAL
grayscale images. The binary-Ising version of this idea produced a NEGATIVE
result (see anisotropic.py), argued to be structural: the binary prior has no
gradual-ramp failure mode to fix. The Potts/grayscale setting is where a benefit
is plausible, so it is tested here directly.

Three couplings, all at matched overall strength J0 = J_ISO:
    isotropic     J_ij = J0                          (the exp03 baseline)
    perona_malik  J_ij = J0 / (1 + (g_ij/kappa)^2)   (edge-preserving)
    cohen         J_ij = J0 * (1 + (g_ij/kappa)^2)   (published inverted form,
                                                      kept as an ablation)
kappa (the gradient scale at which PM coupling halves) is swept, because the
right value is an empirical question and the wrong one collapses PM to either
isotropic (kappa -> inf) or near-zero coupling (kappa -> 0).

Metrics: PSNR and SSIM, as in exp03. Per-image results are printed so the
texture-critical images (Barbara) can be read separately from the aggregate -
the exp03 finding was that these disagree, and that is the whole point.
"""
import sys, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_folder, gaussian_noise, psnr
from potts import potts_gibbs_denoise
from anisotropic import edge_couplings

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

Q = 32
J_ISO = 2.0                    # matched overall strength (exp03 adopted value)
SIGMA = 25 / 255               # the harder noise level, where texture loss bites
KAPPAS = [0.05, 0.10, 0.15, 0.25, 0.40]   # gradient scale for perona_malik
N_SWEEPS, BURN_IN = 25, 10
N_SET12 = None                 # all 12 (includes Barbara at index 8)

# Estimator under test. This is the point of the follow-up run:
#   "mmse" (posterior mean) already reconstructs smooth intensity ramps, so the
#          over-smoothing that anisotropic coupling targets is largely absent,
#          hence the earlier tie with isotropic.
#   "mpm"  (per-pixel marginal mode) gives the piecewise-constant estimate whose
#          quadratic smoothing DOES blur ramps. If the edge-preservation account
#          is right, anisotropic coupling should help HERE and not under mmse.
# Run both (CLI arg "mmse" | "mpm"; default mmse) and compare the verdicts.
ESTIMATOR = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("mmse", "mpm") else "mmse"

set12, paths = load_folder(ROOT / "data" / "set12", n=N_SET12, return_paths=True)
names = [p.stem for p in paths]

def noise_seed(path):
    import zlib
    return 100 + (zlib.crc32(path.name.encode()) & 0xFFFF)

# Pre-generate the noisy images once (same noise for every coupling scheme, so
# differences are due to the prior, not the noise).
noisy = []
for img, p in zip(set12, paths):
    noisy.append(gaussian_noise(img, sigma=SIGMA, seed=noise_seed(p)))

def run(clean, y, seed, mode, kappa):
    if mode == "isotropic":
        xh = potts_gibbs_denoise(y, sigma=SIGMA, q=Q, J=J_ISO,
                                 n_sweeps=N_SWEEPS, burn_in=BURN_IN, seed=seed,
                                 estimator=ESTIMATOR)
    else:
        Jh, Jv = edge_couplings(y, J0=J_ISO, kappa=kappa, mode=mode,
                                sigma=1.0, use_median=False)  # gaussian noise
        xh = potts_gibbs_denoise(y, sigma=SIGMA, q=Q, n_sweeps=N_SWEEPS,
                                 burn_in=BURN_IN, seed=seed, Jh=Jh, Jv=Jv,
                                 estimator=ESTIMATOR)
    return psnr(clean, xh), ssim(clean, xh, data_range=1.0)

# sweep
# isotropic baseline (kappa-independent)
iso = [run(set12[k], noisy[k], k, "isotropic", None) for k in range(len(set12))]
iso_p = np.mean([v[0] for v in iso]); iso_s = np.mean([v[1] for v in iso])

rows = [{"mode": "isotropic", "kappa": float("nan"),
         "psnr": iso_p, "ssim": iso_s,
         "psnr_barbara": iso[8][0], "ssim_barbara": iso[8][1]}]

print(f"Anisotropic Potts on Set12, sigma={round(SIGMA*255)}, J0={J_ISO}, q={Q}, "
      f"estimator={ESTIMATOR.upper()}.")
print(f"{'mode':<14}{'kappa':>6} | {'PSNR':>6} {'SSIM':>6} | "
      f"{'Barbara PSNR':>12} {'Barbara SSIM':>12}")
print("-" * 66)
print(f"{'isotropic':<14}{'--':>6} | {iso_p:>6.2f} {iso_s:>6.3f} | "
      f"{iso[8][0]:>12.2f} {iso[8][1]:>12.3f}")

for mode in ("perona_malik", "cohen"):
    for kappa in KAPPAS:
        res = [run(set12[k], noisy[k], k, mode, kappa) for k in range(len(set12))]
        mp = np.mean([v[0] for v in res]); msim = np.mean([v[1] for v in res])
        rows.append({"mode": mode, "kappa": kappa, "psnr": mp, "ssim": msim,
                     "psnr_barbara": res[8][0], "ssim_barbara": res[8][1]})
        print(f"{mode:<14}{kappa:>6.2f} | {mp:>6.2f} {msim:>6.3f} | "
              f"{res[8][0]:>12.2f} {res[8][1]:>12.3f}")

with open(RESULTS / f"exp05_anisotropic_{ESTIMATOR}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# verdict
pm = [r for r in rows if r["mode"] == "perona_malik"]
best_pm_agg = max(pm, key=lambda r: r["psnr"])
best_pm_bar = max(pm, key=lambda r: r["psnr_barbara"])
best_pm_bar_ssim = max(pm, key=lambda r: r["ssim_barbara"])
print(f"\nVerdict (data-driven, estimator={ESTIMATOR.upper()}):")
def cmp(tag, aniso, iso, unit="dB"):
    d = aniso - iso
    verdict = "BEATS" if d > 0.02 else ("ties" if d > -0.02 else "loses to")
    print(f"  {tag}: best PM {aniso:.3f} {verdict} isotropic {iso:.3f} "
          f"({d:+.3f} {unit})")
cmp("Set12 aggregate PSNR", best_pm_agg["psnr"], iso_p)
cmp("Barbara PSNR (texture)", best_pm_bar["psnr_barbara"], iso[8][0])
cmp("Barbara SSIM (texture)", best_pm_bar_ssim["ssim_barbara"], iso[8][1], unit="")
print(f"  best kappa: aggregate PSNR={best_pm_agg['kappa']}, "
      f"Barbara SSIM={best_pm_bar_ssim['kappa']}")
if ESTIMATOR == "mmse":
    print("  MMSE reconstructs smooth ramps already, so the over-smoothing that")
    print("  adaptive coupling targets should be largely absent -> expect a tie.")
    print("  Compare against the MPM run: python exp05_anisotropic.py mpm")
else:
    print("  MPM is the piecewise-constant estimate whose quadratic smoothing")
    print("  DOES blur ramps. If edge-preservation is the real mechanism,")
    print("  anisotropic should help HERE where it did not under MMSE.")
    print("  If it still ties, the negative result holds across both estimators.")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ks = KAPPAS
for ax, metric, blabel in [(ax1, "psnr", "PSNR (dB)"), (ax2, "ssim", "SSIM")]:
    pm_y = [next(r[metric] for r in rows if r["mode"] == "perona_malik"
                 and r["kappa"] == k) for k in ks]
    co_y = [next(r[metric] for r in rows if r["mode"] == "cohen"
                 and r["kappa"] == k) for k in ks]
    iso_y = rows[0][metric]
    ax.axhline(iso_y, ls="--", color="black", label="isotropic (baseline)")
    ax.plot(ks, pm_y, marker="o", color="#3f9b52", label="Perona-Malik (edge-preserving)")
    ax.plot(ks, co_y, marker="s", color="#c0392b", label="Cohen inverted (ablation)")
    ax.set_xlabel("kappa (gradient scale)"); ax.set_ylabel(blabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax1.set_title("Set12 aggregate: does adaptive coupling beat a global J?")
ax2.set_title("Set12 aggregate SSIM")
fig.suptitle(f"Anisotropic vs isotropic Potts coupling (Set12, \u03c3={round(SIGMA*255)}, "
             f"J0={J_ISO}, estimator={ESTIMATOR.upper()}) \u2014 verdict is data-driven",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RESULTS / f"exp05_anisotropic_sweep_{ESTIMATOR}.png", dpi=130)

print(f"\nsaved -> results/exp05_anisotropic_{ESTIMATOR}.csv, "
      f"exp05_anisotropic_sweep_{ESTIMATOR}.png")