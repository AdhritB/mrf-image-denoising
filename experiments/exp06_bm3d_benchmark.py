"""Experiment 06 - benchmark the Potts MRF denoiser against BM3D.

Supervisor feedback (report meeting): the interpretable MRF/Potts model should be
compared directly against a strong classical baseline, BM3D (Block-Matching and
3D filtering, Dabov et al. 2007), on the same images and noise so the gap is
quantified rather than asserted. This is not expected to beat BM3D; the point is
to show what the physics-inspired prior does and does not recover, and to give
the report a concrete, fair reference point.

Fairness is enforced by running BOTH denoisers on the IDENTICAL noisy image:
the same filename-derived seed (noise_seed) and the same sigma as exp03, so any
difference is due to the method, not the noise. Metrics (PSNR, SSIM) and the
adopted coupling J are exactly those of exp03, so the Potts column here matches
the exp03 production numbers.

Requires the `bm3d` package (pip install bm3d) - the reference Python port of
the Dabov et al. algorithm.
"""
import sys, csv, zlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

try:
    import bm3d
except ImportError:
    sys.exit("bm3d not installed. Run:  pip install bm3d")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_folder, gaussian_noise, psnr
from potts import potts_gibbs_denoise

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

Q = 32
J_ADOPTED = 2.0                 # identical to exp03
SIGMAS = [15, 25]               # on the 0-255 scale
N_SET12 = None                  # all 12
N_BSD = 10                      # same subset as exp03
N_SWEEPS, BURN_IN = 25, 10


def noise_seed(path):
    """Identical to exp03: filename-derived, reproducible."""
    return 100 + (zlib.crc32(path.name.encode()) & 0xFFFF)


def run_dataset(name, folder, n):
    imgs, paths = load_folder(folder, n=n, return_paths=True)
    rows = []
    for sigma255 in SIGMAS:
        sigma = sigma255 / 255.0
        pot_p, pot_s, bm_p, bm_s, noi_p = [], [], [], [], []
        for img, path in zip(imgs, paths):
            y = gaussian_noise(img, sigma=sigma, seed=noise_seed(path))
            # Potts (identical call to exp03)
            xp = potts_gibbs_denoise(y, sigma=sigma, q=Q, J=J_ADOPTED,
                                     n_sweeps=N_SWEEPS, burn_in=BURN_IN,
                                     seed=noise_seed(path))
            # BM3D on the SAME noisy image, told the true sigma
            xb = np.clip(bm3d.bm3d(y, sigma_psd=sigma), 0.0, 1.0)

            noi_p.append(psnr(img, y))
            pot_p.append(psnr(img, xp)); pot_s.append(ssim(img, xp, data_range=1.0))
            bm_p.append(psnr(img, xb));  bm_s.append(ssim(img, xb, data_range=1.0))

        row = {
            "dataset": name, "sigma255": sigma255,
            "psnr_noisy": np.mean(noi_p),
            "psnr_potts": np.mean(pot_p), "ssim_potts": np.mean(pot_s),
            "psnr_bm3d":  np.mean(bm_p),  "ssim_bm3d":  np.mean(bm_s),
            "psnr_gap_bm3d_minus_potts": np.mean(bm_p) - np.mean(pot_p),
            "ssim_gap_bm3d_minus_potts": np.mean(bm_s) - np.mean(pot_s),
        }
        rows.append(row)
        print(f"{name:<12} sigma={sigma255:>2} | "
              f"noisy {row['psnr_noisy']:.2f} | "
              f"Potts {row['psnr_potts']:.2f}/{row['ssim_potts']:.3f} | "
              f"BM3D {row['psnr_bm3d']:.2f}/{row['ssim_bm3d']:.3f} | "
              f"gap {row['psnr_gap_bm3d_minus_potts']:+.2f} dB / "
              f"{row['ssim_gap_bm3d_minus_potts']:+.3f} SSIM")
    return rows


print(f"BM3D vs Potts benchmark. J={J_ADOPTED}, q={Q}. "
      f"Same noisy images (filename-seeded), same metrics as exp03.\n")
print(f"{'dataset':<12} {'sigma':>5} | {'noisy':>5} | "
      f"{'Potts PSNR/SSIM':>16} | {'BM3D PSNR/SSIM':>16} | {'gap (BM3D - Potts)':>18}")
print("-" * 92)

all_rows = []
all_rows += run_dataset("Set12", ROOT / "data" / "set12", N_SET12)
all_rows += run_dataset("BSD68subset", ROOT / "data" / "bsd68", N_BSD)

with open(RESULTS / "exp06_bm3d_benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
labels = [f"{r['dataset']}\n$\\sigma$={r['sigma255']}" for r in all_rows]
x = np.arange(len(all_rows)); w = 0.35
for ax, key_p, key_b, ylab, title in [
        (ax1, "psnr_potts", "psnr_bm3d", "PSNR (dB)", "PSNR: Potts vs BM3D"),
        (ax2, "ssim_potts", "ssim_bm3d", "SSIM", "SSIM: Potts vs BM3D")]:
    ax.bar(x - w/2, [r[key_p] for r in all_rows], w, label="Potts (this work)",
           color="#3b6fb0")
    ax.bar(x + w/2, [r[key_b] for r in all_rows], w, label="BM3D",
           color="#e0873a")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Interpretable Potts MRF vs BM3D on identical noisy images",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RESULTS / "exp06_bm3d_benchmark.png", dpi=130)

# verdict
print("\nVerdict:")
for r in all_rows:
    print(f"  {r['dataset']} sigma={r['sigma255']}: BM3D leads Potts by "
          f"{r['psnr_gap_bm3d_minus_potts']:.2f} dB PSNR, "
          f"{r['ssim_gap_bm3d_minus_potts']:+.3f} SSIM.")
print("  (BM3D is expected to lead; the value is a quantified, fair gap for the "
      "report, not a claim that the MRF competes.)")
print("\nsaved -> results/exp06_bm3d_benchmark.csv, exp06_bm3d_benchmark.png")
