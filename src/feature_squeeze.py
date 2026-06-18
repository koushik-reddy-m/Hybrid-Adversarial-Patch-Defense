import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import (
    DEVICE, FS_BIT_DEPTH, FS_MEDIAN_K, FS_THRESH_SWEEP,
    FS_MAX_FPR, FS_VAL_N, THRESHOLD_PATH, PLOTS_DIR,
    IMAGENET_MEAN, IMAGENET_STD
)


def _normalised_to_float01(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=image.device)[:, None, None]
    return (image * std + mean).clamp(0, 1)


def _float01_to_normalised(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=image.device)[:, None, None]
    return (image - mean) / std


def squeeze_bit_depth(image: torch.Tensor) -> torch.Tensor:
    img01    = _normalised_to_float01(image)
    levels   = 2 ** FS_BIT_DEPTH          #  4 levels for 2-bit
    quantised = torch.floor(img01 * levels) / levels
    quantised = quantised.clamp(0, 1)
    return _float01_to_normalised(quantised)


def squeeze_median(image: torch.Tensor) -> torch.Tensor:
    img01  = _normalised_to_float01(image.cpu())   # (C,H,W)
    pil_img = to_pil_image(img01)
    smoothed = pil_img.filter(ImageFilter.MedianFilter(size=FS_MEDIAN_K))
    smoothed_t = to_tensor(smoothed).to(image.device)
    return _float01_to_normalised(smoothed_t)


def squeeze_batch(images: torch.Tensor, squeeze_fn) -> torch.Tensor:
    """Apply squeeze_fn to every image in a (B,C,H,W) batch."""
    return torch.stack([squeeze_fn(images[i]) for i in range(images.size(0))])


# Score computation

@torch.no_grad()
def compute_fs_score(
    model,
    images: torch.Tensor,
) -> torch.Tensor:
    images = images.to(DEVICE)
    logits_orig  = model(images)
    probs_orig   = F.softmax(logits_orig, dim=1)

    # Squeezer 1: bit-depth
    sq1     = squeeze_batch(images, squeeze_bit_depth).to(DEVICE)
    probs1  = F.softmax(model(sq1), dim=1)

    # Squeezer 2: median
    sq2     = squeeze_batch(images, squeeze_median).to(DEVICE)
    probs2  = F.softmax(model(sq2), dim=1)

    # L1 distance (sum over classes, normalised by 2)
    l1_1 = (probs_orig - probs1).abs().sum(dim=1)
    l1_2 = (probs_orig - probs2).abs().sum(dim=1)

    scores = torch.max(l1_1, l1_2)   # (B,)
    return scores


def tune_threshold(
    model,
    images_clean:   torch.Tensor,
    images_patched: torch.Tensor,
    save: bool = True,
) -> float:
    print(f"Computing scores on {images_clean.size(0)} "
          f"clean + {images_patched.size(0)} patched images …")

    scores_clean   = compute_fs_score(model, images_clean).cpu().numpy()
    scores_patched = compute_fs_score(model, images_patched).cpu().numpy()

    best_thresh   = None
    best_det_rate = 0.0

    sweep_results = []
    for thresh in FS_THRESH_SWEEP:
        tp  = (scores_patched >= thresh).sum()
        fn  = (scores_patched  < thresh).sum()
        fp  = (scores_clean   >= thresh).sum()
        tn  = (scores_clean    < thresh).sum()

        det_rate = tp / max(tp + fn, 1)
        fpr      = fp / max(fp + tn, 1)

        sweep_results.append({
            "threshold":    thresh,
            "detection_rate": float(det_rate),
            "fpr":            float(fpr),
        })

        if fpr <= FS_MAX_FPR and det_rate > best_det_rate:
            best_det_rate = det_rate
            best_thresh   = thresh

    if best_thresh is None:
        # Relax FPR constraint if nothing passes
        best_thresh = FS_THRESH_SWEEP[len(FS_THRESH_SWEEP) // 2]
        print("no threshold met FPR constraint :using median.")

    print(f"Best threshold: {best_thresh:.3f}  "
          f"Detection: {100*best_det_rate:.1f}%")

    if save:
        torch.save({"threshold": best_thresh, "sweep": sweep_results}, THRESHOLD_PATH)
        print(f" Threshold saved {THRESHOLD_PATH}")
        _plot_threshold_sweep(sweep_results, best_thresh)
        _plot_score_histograms(scores_clean, scores_patched, best_thresh)

    return best_thresh


def load_threshold() -> float:
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"No threshold at {THRESHOLD_PATH}"
        )
    data = torch.load(THRESHOLD_PATH, map_location="cpu")
    t    = data["threshold"]
    print(f"Threshold loaded: {t:.3f}")
    return t


# Single-image detection 

@torch.no_grad()
def detect_fs(
    model,
    image:     torch.Tensor,
    threshold: float,
) -> tuple[bool, float]:
    
    score = compute_fs_score(model, image.unsqueeze(0)).item()
    return score >= threshold, score


@torch.no_grad()
def detect_fs_batch(
    model,
    images:    torch.Tensor,
    threshold: float,
) -> tuple[list[bool], torch.Tensor]:
    
    scores = compute_fs_score(model, images)
    flags  = (scores >= threshold).tolist()
    return flags, scores


def evaluate_fs(
    model,
    images_clean:   torch.Tensor,
    images_patched: torch.Tensor,
    threshold:      float,
) -> dict:
    flags_clean,   sc = detect_fs_batch(model, images_clean,   threshold)
    flags_patched, sp = detect_fs_batch(model, images_patched, threshold)

    tp = sum(flags_patched)
    fn = len(flags_patched) - tp
    fp = sum(flags_clean)
    tn = len(flags_clean) - fp

    det_rate  = tp / max(tp + fn, 1)
    fpr       = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    result = {
        "threshold":      threshold,
        "detection_rate": round(100 * det_rate,  2),
        "fpr":            round(100 * fpr,        2),
        "precision":      round(100 * precision,  2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    print(f"Threshold {threshold:.3f}  "
          f"Detection: {result['detection_rate']:.1f}%  "
          f"FPR: {result['fpr']:.1f}%  "
          f"Precision: {result['precision']:.1f}%")
    return result



def _plot_threshold_sweep(sweep: list[dict], best_thresh: float):
    thresholds = [s["threshold"]    for s in sweep]
    det_rates  = [s["detection_rate"] for s in sweep]
    fprs       = [s["fpr"]            for s in sweep]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, det_rates, "o-", color="#1D9E75", label="Detection rate %")
    ax.plot(thresholds, fprs,      "s--", color="#D85A30", label="False positive rate %")
    ax.axvline(best_thresh, color="#534AB7", linestyle=":", linewidth=1.5,
               label=f"Chosen threshold = {best_thresh:.3f}")
    ax.axhline(100 * FS_MAX_FPR, color="#D85A30", linestyle=":", alpha=0.5,
               label=f"Max FPR = {100*FS_MAX_FPR:.0f}%")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Percentage")
    ax.set_title("Feature squeezing :threshold sweep")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fs_threshold_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f" Threshold sweep plot saved {path}")


def _plot_score_histograms(
    scores_clean:   np.ndarray,
    scores_patched: np.ndarray,
    threshold:      float,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(scores_clean.max(), scores_patched.max()) + 0.01, 40)
    ax.hist(scores_clean,   bins=bins, alpha=0.6, color="#378ADD", label="Clean images")
    ax.hist(scores_patched, bins=bins, alpha=0.6, color="#D85A30", label="Patched images")
    ax.axvline(threshold, color="#534AB7", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Adversarial score (max L1)")
    ax.set_ylabel("Count")
    ax.set_title("Feature squeezing score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fs_score_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f" Score histogram saved  {path}")
