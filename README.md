# Image Denoising with Probabilistic Models

Markov random field priors for image denoising, written for the MSc dissertation
*Image Denoising with Probabilistic Models: Markov Random Field Priors and
Inference* (University of Manchester, 2026).

The code starts from the binary Ising model on MNIST, generalises it to a
q-state Potts model for grayscale images, compares three approximate inference
algorithms on the same posterior, and tests two extensions: a spatially adaptive
coupling, and a coupling learned from data by pseudo-likelihood. Two of the
results are negative. Both are reported here with the mechanism that produces
them, because a tested extension that fails for an identifiable reason is still
a result.

## The model

A clean image is a lattice of spins or states with a smoothness prior and a
likelihood term. In the binary case the posterior over spins `x ∈ {-1,+1}` given
a noisy observation `y` is

    P(x | y)  ∝  exp( β Σᵢ yᵢxᵢ  +  J Σ₍ᵢⱼ₎ xᵢxⱼ )

The pairwise term is the ferromagnetic Ising prior over the 4-nearest-neighbour
lattice. The unary term is the binary-symmetric-channel likelihood, and for flip
probability `p` the statistically correct weight is `β = ½ ln((1-p)/p)`.

The Potts generalisation replaces two spins with `q` ordered intensity states
`c_k = (k + 0.5)/q` and swaps the channel likelihood for a Gaussian one, so the
Gibbs conditional at a pixel becomes a q-way softmax over
`-(yᵢ - c_k)²/(2σ²) + J n_i(k)`, where `n_i(k)` counts the neighbours currently
in state `k`. All experiments use `q = 32`.

## Repository layout

```
src/
  ising.py            binary Ising model: neighbour sums, Gibbs sampler, mean-field
  potts.py            q-state Potts model for grayscale (isotropic + per-edge coupling)
  lbp.py              loopy belief propagation (sum-product) for the Ising posterior
  anisotropic.py      gradient-dependent edge couplings (Perona-Malik and Cohen forms)
  pseudolikelihood.py maximum-pseudo-likelihood estimation of the Potts coupling J
  datasets.py         data loading (MNIST/BSD68/Set12/Kodak), noise models, metrics
experiments/
  exp01_mnist_baseline.py       baseline Ising denoiser on MNIST; J-sweep
  exp02_inference_comparison.py Gibbs vs mean-field vs loopy BP
  exp03_potts_grayscale.py      Potts coupling tuning on Set12 and a BSD68 subset
  exp04_recognizability.py      task-level (OCR-style) evaluation + error maps
  exp05_anisotropic.py          anisotropic coupling (MMSE and MPM estimators)
  exp06_bm3d_benchmark.py       Potts vs BM3D on identical noisy images
  exp07_pseudolikelihood.py     learning the coupling J from data
  replot_figures.py             redraws the report figures from the result CSVs
results/                        figures and CSVs written by the experiments
data/                           datasets, not tracked (see below)
```

`replot_figures.py` reads the CSVs and redraws the plots for legibility, so it
cannot change a reported number. Two figures need code rather than a CSV (the
Barbara panel and the stability trace in exp07), and it recomputes those with
the same seeds. Pass `--csv-only` to skip that step.

## Setup

```bash
pip install -r requirements.txt
```

SciPy is used by `anisotropic.py` for the median and Gaussian pre-smoothing that
the gradient estimate depends on. The file falls back to an unsmoothed gradient
if SciPy is missing, which silently changes the exp05 numbers, so install it.
`bm3d` is only needed by exp06.

### Datasets

Datasets are not in the repository. Put them under `data/`:

- MNIST, `data/mnist/` (the four `idx` files):
  <http://yann.lecun.com/exdb/mnist/>
- Set12 and BSD68, `data/set12/` and `data/bsd68/`, the test splits distributed
  with DnCNN: <https://github.com/cszn/DnCNN>
- Kodak, `data/kodak/`. The loader supports it, no experiment uses it.

The MNIST experiments use the first 50 test digits, and exp04 builds its reader
from 3000 clean training digits. exp03 and exp06 use all 12 Set12 images and the
first 10 BSD68 images by natural filename order. That subset is asserted at load
time, so a re-extracted folder fails loudly instead of quietly shifting the
numbers.

## Running the experiments

Each script is self-contained and writes its figures and a CSV to `results/`:

```bash
python experiments/exp01_mnist_baseline.py
python experiments/exp03_potts_grayscale.py          # or: exp03_potts_grayscale.py Set12
python experiments/exp05_anisotropic.py mmse         # estimator: mmse (default) or mpm
python experiments/exp06_bm3d_benchmark.py
python experiments/replot_figures.py
```

Noise is seeded from each image's filename rather than its position in the list,
so a run reproduces byte-for-byte and does not depend on the order files load
in. Set12 at four coupling values and two noise levels takes roughly 28 s per
image per condition on a laptop CPU, which is the slowest thing here.

## Results

Overall pixel accuracy is misleading on sparse binary images. At `J=0.3`,
`p=0.2` the Ising denoiser scores 0.888 overall while a blank image scores
0.886, because 89% of the pixels in a binarised MNIST digit are background. Foreground accuracy
tells the real story: it falls 0.791, 0.620, 0.363, 0.241 as `J` goes 0.3 to
1.0. A task-level check agrees. Feeding the restorations to a 3-NN digit reader,
recognisability drops 0.88, 0.68, 0.48, 0.38 against a clean-digit ceiling of
0.92, while overall accuracy stays near 0.9 throughout. The prior is erasing the
strokes, not denoising them.

The three inference algorithms cannot be separated on overall accuracy (0.942 to
0.951 at `p=0.20`) and separate clearly on foreground accuracy: mean-field
0.671, loopy BP 0.626, Gibbs 0.620. Gibbs erases most because it samples the
posterior faithfully, and that posterior is the one being asked for. Mean-field
is not a better approximation, its error just happens to leave strokes standing.
Runtimes per image are 4.0 ms for Gibbs, 3.8 ms for mean-field and 10.3 ms for
loopy BP, which converged on every image.

Potts denoising on grayscale, at the adopted `J = 2.0` and `q = 32`:

| Dataset | σ  | PSNR gain | SSIM gain |
|---------|----|-----------|-----------|
| Set12   | 15 | +3.86 dB  | +0.220    |
| Set12   | 25 | +4.73 dB  | +0.235    |
| BSD68   | 15 | +3.30 dB  | +0.194    |
| BSD68   | 25 | +4.45 dB  | +0.234    |

PSNR and SSIM peak at the same coupling in all four conditions, which was not
the expected outcome. No single `J` is admissible everywhere under the tie
tolerance used (0.10 dB, 0.005 SSIM): the BSD68 subset carries more fine texture
and peaks lower, at 1.75 to 2.0, while Set12 peaks at 2.0 to 2.5. `J = 2.0` is
admissible in three of the four conditions and is adopted on that basis. Barbara
shows the cost directly, losing its scarf and tablecloth texture as `J` rises
while the aggregate still improves.

Against BM3D on identical noisy images, BM3D leads by 3.88 and 4.89 dB on Set12
and by 2.93 and 3.35 dB on the BSD68 subset at σ=15 and 25, so the gap widens
with noise. The Potts model recovers about half the available gain with one
coupling parameter. As a check on the baseline, this BM3D run reproduces its
published BSD68 numbers (31.05 and 28.30 dB here, 31.07 and 28.57 reported).

Anisotropic coupling does not beat a well-tuned global `J`. Best Perona-Malik
against isotropic is 25.07 vs 25.09 dB under the MMSE estimator and 24.12 vs
24.10 dB under MPM, both inside the tie band. The first run suggested the
posterior mean was already reconstructing smooth ramps, leaving nothing for
edge-aware coupling to fix, so the whole sweep was repeated under MPM, the
piecewise-constant estimate that does blur ramps. It ties there too. What does
show up under MPM is the sign correction: the edge-preserving form beats the
published inverted one on Barbara SSIM at every κ tested, though by margins near
the noise floor.

Cohen et al. (2015), Eq. 7 sets the coupling to `J = 1 + ||∇N||²`, which
strengthens smoothing where an edge should be preserved. `anisotropic.py`
implements the reciprocal, `J₀ / (1 + (||∇N||/κ)²)`, and keeps the published
form as `mode="cohen"` so the difference can be measured instead of asserted.

Learning the coupling by pseudo-likelihood returns a different answer depending
on what it is fitted to: 1.73 from clean labellings, 0.92 from noisy ones, and
3.20 when re-estimated from the current reconstruction over eight cycles. None
of the 12 estimates hit the bracket edge, so this is not the outright divergence
Besag reported in 1986, but the feedback loop still drifts upward and away from
the hand-tuned value. His conclusion holds: pseudo-likelihood is a poor fit for
estimating a coupling during reconstruction.

## Reproducibility

Noise seeds come from filenames, image subsets are asserted at load time, and
every number quoted above sits in a CSV under `results/`. Metric values can
still shift in the third decimal across NumPy and scikit-image versions. The
exp04 reader in particular moved from 0.88 to 0.86 recognisability on a
different machine, so figures in the dissertation come from a single environment
rather than a mix.

## References

Geman & Geman (1984) *IEEE TPAMI* 6, 721-741 ·
Besag (1974) *JRSS-B* 36, 192-236 ·
Besag (1975) *The Statistician* 24, 179-195 ·
Besag (1986) *JRSS-B* 48, 259-302 ·
Murphy (2023) *Probabilistic Machine Learning: Advanced Topics*, MIT Press ·
Cohen et al. (2015) *Signal Processing: Image Communication* 34, 14-21 ·
Yedidia, Freeman & Weiss (2005) *IEEE Trans. Information Theory* 51, 2282-2312 ·
Dabov et al. (2007) *IEEE Trans. Image Processing* 16, 2080-2095 ·
Roth & Black (2009) *IJCV* 82, 205-229 ·
Zhang et al. (2017) *IEEE Trans. Image Processing* 26, 3142-3155.
