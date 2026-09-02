# Image Denoising with Probabilistic Models

Markov random field (MRF) priors for image denoising, developed for the MSc
dissertation *Image Denoising with Probabilistic Models: Markov Random Field
Priors and Inference* (University of Manchester).

The project starts from the binary Ising model, generalises it to the
multi-state Potts model for grayscale images, compares three approximate
inference algorithms on the same posterior, and tests two extensions: a
spatially adaptive coupling and a coupling learned from data by
pseudo-likelihood.

## Model

A clean image is treated as a lattice of spins/states with a smoothness prior
and a likelihood term. For the binary case, the posterior over spins
`x ∈ {-1,+1}` given a noisy observation `y` is

    P(x | y)  ∝  exp( β Σᵢ yᵢxᵢ  +  J Σ₍ᵢⱼ₎ xᵢxⱼ )

where the pairwise term is the ferromagnetic Ising prior (smoothness over the
4-nearest-neighbour lattice) and the unary term is the binary-symmetric-channel
likelihood. For flip probability `p`, `β = ½ ln((1-p)/p)`. The Potts
generalisation replaces the two spins with `q` ordered intensity states and a
Gaussian likelihood.

## Repository layout

```
src/
  ising.py            binary Ising model: neighbour sums, Gibbs sampler, mean-field
  potts.py            q-state Potts model for grayscale (isotropic + per-edge coupling)
  lbp.py              loopy belief propagation (sum-product) for the Ising posterior
  anisotropic.py      gradient-dependent edge couplings (Perona–Malik and Cohen forms)
  pseudolikelihood.py maximum-pseudo-likelihood estimation of the Potts coupling J
  datasets.py         data loading (MNIST/BSD68/Set12/Kodak), noise models, metrics
experiments/
  exp01_mnist_baseline.py       baseline Ising denoiser on MNIST; J-sweep
  exp02_inference_comparison.py Gibbs vs mean-field vs loopy BP
  exp03_potts_grayscale.py      Potts coupling tuning on Set12/BSD68
  exp04_recognizability.py      task-level (OCR-style) evaluation + error maps
  exp05_anisotropic.py          anisotropic coupling (MMSE and MPM estimators)
  exp06_bm3d_benchmark.py       Potts vs BM3D on identical noisy images
  exp07_pseudolikelihood.py     learning the coupling J from data
results/                        generated figures and CSVs (created on run)
data/                           datasets (not tracked; see below)
```

## Setup

```bash
pip install -r requirements.txt
```

`bm3d` (used only by `exp06`) is the reference Python port of the BM3D
algorithm. The other experiments need only NumPy, Pillow, Matplotlib and
scikit-image.

### Datasets

The datasets are not included in the repository. Place them under `data/`:

- **MNIST** — `data/mnist/` (the standard `idx` files):
  <http://yann.lecun.com/exdb/mnist/>
- **Set12** and **BSD68** — `data/set12/` and `data/bsd68/`, the test splits
  distributed with DnCNN: <https://github.com/cszn/DnCNN>
- **Kodak** (optional, colour) — `data/kodak/`:
  <https://r0k.us/graphics/kodak/>

## Running the experiments

Each experiment is self-contained and writes its figures and a CSV to
`results/`:

```bash
python experiments/exp01_mnist_baseline.py
python experiments/exp03_potts_grayscale.py
python experiments/exp06_bm3d_benchmark.py
# ...etc
```

Noise is seeded from each image's filename, so results are reproducible across
runs and independent of the order in which files are loaded.

## Selected results

**Metric choice matters (binary MNIST, exp01/exp04).** Overall pixel accuracy is
misleading on sparse images: at `J=0.3`, `p=0.2` the denoiser scores `0.888`
overall while a blank image scores `0.886`. Foreground accuracy and a task-level
recognisability metric expose the over-smoothing that overall accuracy hides — as
the coupling rises, recognisability collapses (`0.88 → 0.68 → 0.48 → 0.38`) while
overall accuracy stays near `0.9`.

**Potts denoiser (exp03), adopted `J = 2.0`, `q = 32`:**

| Dataset | σ  | PSNR gain | SSIM gain |
|---------|----|-----------|-----------|
| Set12   | 15 | +3.86 dB  | +0.220    |
| Set12   | 25 | +4.73 dB  | +0.235    |
| BSD68   | 15 | +3.30 dB  | +0.194    |
| BSD68   | 25 | +4.45 dB  | +0.234    |

**Benchmark vs BM3D (exp06).** BM3D leads by 2.9–4.9 dB, and the gap widens with
noise; the interpretable Potts model recovers roughly half of the achievable gain
with a single coupling parameter. The BM3D baseline reproduces its published
BSD68 figures to within 0.3 dB.

**Extensions.** Anisotropic coupling (exp05) does not improve on a well-tuned
global coupling, for a reason the experiment isolates (the posterior-mean
estimator already reconstructs smooth gradients). Learning the coupling by
pseudo-likelihood (exp07) returns a different estimate depending on the labelling
it is fitted to (1.73 from clean images, 0.92 from noisy, 3.20 iterated), a
concrete illustration of the limitation Besag identified.

## References

Geman & Geman (1984) *IEEE TPAMI* 6, 721–741 ·
Besag (1974) *JRSS-B* 36, 192–236 ·
Besag (1975) *The Statistician* 24, 179–195 ·
Besag (1986) *JRSS-B* 48, 259–302 ·
Murphy (2023) *Probabilistic Machine Learning: Advanced Topics*, MIT Press ·
Cohen et al. (2015) *Signal Processing: Image Communication* 34, 14–21 ·
Yedidia, Freeman & Weiss (2005) *IEEE Trans. Information Theory* 51, 2282–2312 ·
Dabov et al. (2007) *IEEE Trans. Image Processing* 16, 2080–2095 ·
Roth & Black (2009) *IJCV* 82, 205–229 ·
Zhang et al. (2017) *IEEE Trans. Image Processing* 26, 3142–3155.
