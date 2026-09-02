"""
datasets.py — Unified data loading and noise injection for the
"Image Denoising with Probabilistic Models" project.

Datasets

- MNIST  : 28x28 binary digits (after binarisation)   -> Ising model
- BSD68  : 68 grayscale natural images (321x481)      -> Potts model
- Set12  : 12 classic grayscale test images (256/512) -> Potts model
- Kodak  : 24 true-colour images (768x512)            -> Potts (per channel)

Noise models

- Binary channel noise (pixel flips, prob p)          -> for binary Ising
- Salt & pepper impulse noise (prob p)                -> as in Cohen et al. (2015)
- Additive Gaussian noise (sigma)                     -> standard benchmark setting

Usage

    from datasets import load_mnist, load_folder, binarise
    from datasets import flip_noise, salt_pepper, gaussian_noise

    imgs, labels = load_mnist("data/mnist", split="test", n=100)
    x = binarise(imgs[0])            # {-1,+1} spins
    y = flip_noise(x, p=0.1, seed=0) # noisy observation
"""

import gzip
import struct
from pathlib import Path

import numpy as np
from PIL import Image


# Loading

def load_mnist(root, split="test", n=None):
    """Load MNIST digits as float arrays in [0,1].

    Parameters
    
    root  : path to folder containing the four idx .gz files
    split : "train" (60k) or "test" (10k)
    n     : optionally limit to first n images

    Returns
    
    images : (N, 28, 28) float32 array in [0, 1]
    labels : (N,) int array
    """
    root = Path(root)
    prefix = "train" if split == "train" else "t10k"
    with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if n is not None:
        images, labels = images[:n], labels[:n]
    return images.astype(np.float32) / 255.0, labels.astype(int)


def _natural_key(path):
    """Sort key that orders test2 before test10 (numeric-aware).

    Plain sorted() is lexicographic, so 'test10.png' sorts before 'test2.png'.
    For a subset like the first N images this silently changes WHICH images are
    selected, and because each image is later paired with a position-based noise
    seed, it also changes the noise. Natural order makes the selection depend on
    the intended numbering rather than on string collation.
    """
    import re
    return [int(t) if t.isdigit() else t
            for t in re.split(r"(\d+)", str(path.name))]


def load_folder(root, grayscale=True, n=None, return_paths=False):
    """Load all PNG images from a folder (BSD68 / Set12 / Kodak).

    Returns a list of float32 arrays in [0,1]; grayscale -> (H,W),
    colour -> (H,W,3). A list is used because image sizes differ.

    Files are ordered by a natural (numeric-aware) sort so that a subset
    (n < total) always selects the same images regardless of zero-padding or
    string collation. Pass return_paths=True to also get the ordered file list,
    which lets an experiment assert exactly which images it ran on.
    """
    root = Path(root)
    paths = sorted(root.glob("*.png"), key=_natural_key)
    if n is not None:
        paths = paths[:n]
    out = []
    for p in paths:
        img = Image.open(p)
        img = img.convert("L") if grayscale else img.convert("RGB")
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return (out, paths) if return_paths else out



# Preprocessing


def binarise(img, threshold=0.5):
    """Map a [0,1] grayscale image to {-1,+1} spins (Ising convention)."""
    return np.where(img > threshold, 1, -1).astype(np.int8)


def quantise(img, q=16):
    """Quantise a [0,1] grayscale image to q levels {0,...,q-1} (Potts states)."""
    return np.clip((img * q).astype(np.int32), 0, q - 1)



# Noise models


def flip_noise(spins, p=0.1, seed=None):
    """Binary symmetric channel: flip each {-1,+1} spin with probability p."""
    rng = np.random.default_rng(seed)
    flips = rng.random(spins.shape) < p
    return np.where(flips, -spins, spins).astype(spins.dtype)


def salt_pepper(img, p=0.1, seed=None):
    """Impulse noise on a [0,1] image: each pixel independently set to
    0 or 1 with total probability p (as in Cohen et al. 2015)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < p / 2] = 0.0
    out[(r >= p / 2) & (r < p)] = 1.0
    return out


def gaussian_noise(img, sigma=25 / 255, seed=None, clip=True):
    """Additive white Gaussian noise on a [0,1] image.
    sigma=25/255 matches the standard BSD68/Set12 benchmark setting."""
    rng = np.random.default_rng(seed)
    out = img + rng.normal(0.0, sigma, img.shape).astype(img.dtype)
    return np.clip(out, 0.0, 1.0) if clip else out



# Metrics


def psnr(clean, noisy, data_range=1.0):
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((np.asarray(clean, np.float64) - np.asarray(noisy, np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def pixel_accuracy(clean_spins, restored_spins):
    """Fraction of correctly recovered {-1,+1} pixels."""
    return float(np.mean(clean_spins == restored_spins))

def denoising_report(clean, restored, noisy):
    clean, restored = np.asarray(clean), np.asarray(restored)
    fg, bg = clean == 1, clean == -1
    fg_acc = float(np.mean(restored[fg] == 1)) if fg.any() else float("nan")
    bg_acc = float(np.mean(restored[bg] == -1)) if bg.any() else float("nan")
    return {
        "overall":     float(np.mean(clean == restored)),
        "foreground":  fg_acc,                       # the honest number
        "balanced":    0.5 * (fg_acc + bg_acc),
        "baseline_noisy":  float(np.mean(clean == np.asarray(noisy))),
        "baseline_allbg":  float(np.mean(clean == -1)),
    }


if __name__ == "__main__":
    # smoke test
    base = Path(__file__).parent
    imgs, labels = load_mnist(base / "mnist", split="test", n=5)
    x = binarise(imgs[0])
    y = flip_noise(x, p=0.1, seed=0)
    print("MNIST ok:", imgs.shape, "| flip acc vs clean:", pixel_accuracy(x, y))
    bsd = load_folder(base / "bsd68", n=2)
    print("BSD68 ok:", bsd[0].shape)
    s12 = load_folder(base / "set12", n=2)
    print("Set12 ok:", s12[0].shape)
    kod = load_folder(base / "kodak", grayscale=False, n=2)
    print("Kodak ok:", kod[0].shape)
    g = gaussian_noise(bsd[0], sigma=25 / 255, seed=0)
    print("Gaussian noise PSNR:", round(psnr(bsd[0], g), 2), "dB")