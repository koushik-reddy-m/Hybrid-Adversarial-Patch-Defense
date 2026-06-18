
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.config import (
    IMAGE_SIZE, GRID_CELLS, KP_SIGMA, KP_THRESH_PCTILE,
    KP_MAX_AREA_FRAC, CLASSICAL_TEST_N, DEVICE,
    PATCH_SIZE_PX, PLOTS_DIR, CHECKPOINT_DIR
)


# Keypoint detectors 

def _get_detector(method: str):
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create(nfeatures=500)
    # elif method == "surf":
    #     # SURF is patented and not included in standard OpenCV
    #     return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    elif method == "akaze":
        return cv2.AKAZE_create()
    elif method == "orb":
        return cv2.ORB_create(nfeatures=500)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: sift, surf, akaze, orb, fast, brisk")


def _tensor_to_cv(image: torch.Tensor) -> np.ndarray:
    from src.data_loader import denormalise
    img = denormalise(image.cpu())                     # (C,H,W) in [0,1]
    img_np = img.permute(1, 2, 0).numpy()              # (H,W,C) RGB float
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


def _build_density_map(keypoints: list, h: int, w: int) -> np.ndarray:
    grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)

    cell_h = h / GRID_CELLS
    cell_w = w / GRID_CELLS

    for kp in keypoints:
        r = int(kp.pt[1] / cell_h)
        c = int(kp.pt[0] / cell_w)
        r = min(r, GRID_CELLS - 1)
        c = min(c, GRID_CELLS - 1)
        grid[r, c] += 1

    # Gaussian smooth
    grid = gaussian_filter(grid, sigma=KP_SIGMA)

    # Normalise
    if grid.max() > 0:
        grid /= grid.max()

    return grid




def _density_to_bbox(density: np.ndarray) -> tuple[int, int, int, int] | None:
    if density.max() == 0:
        return None
    
    threshold = np.percentile(density, 90)  
    hot = density >= threshold
    
    rows, cols = np.where(hot)
    if len(rows) < 5:
        return None
    
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    
    bbox_cells = (r2 - r1 + 1) * (c2 - c1 + 1)
    fill_ratio = len(rows) / bbox_cells
    
    if fill_ratio < 0.4:
        return None
    
    hot_area_frac = hot.mean()
    if hot_area_frac < 0.05 or hot_area_frac > 0.30:
        return None
    
    cell_h = IMAGE_SIZE / GRID_CELLS
    cell_w = IMAGE_SIZE / GRID_CELLS
    
    r1_px = int(r1 * cell_h)
    c1_px = int(c1 * cell_w)
    r2_px = int((r2 + 1) * cell_h)
    c2_px = int((c2 + 1) * cell_w)
    
    return (r1_px, c1_px, r2_px, c2_px)



def detect_patch(
    image: torch.Tensor,
    method: str = "akaze",
) -> tuple[bool, tuple[int, int, int, int] | None, dict]:

    detector = _get_detector(method)
    img_bgr  = _tensor_to_cv(image)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    keypoints, _ = detector.detectAndCompute(gray, None)

    density = _build_density_map(keypoints, IMAGE_SIZE, IMAGE_SIZE)
    bbox    = _density_to_bbox(density)

    is_adversarial = bbox is not None

    return is_adversarial, bbox, {
        "n_keypoints": len(keypoints),
        "density_map": density,
        "method":      method,
    }


def detect_patch_batch(
    images: torch.Tensor,
    method: str = "akaze",
) -> tuple[list[bool], list[tuple | None], list[dict]]:

    flags, bboxes, infos = [], [], []
    for i in range(images.size(0)):
        f, b, info = detect_patch(images[i], method=method)
        flags.append(f)
        bboxes.append(b)
        infos.append(info)
    return flags, bboxes, infos



def benchmark_detectors(
    images_clean:  torch.Tensor,
    images_patched: torch.Tensor,
) -> dict:
    methods = ["sift", "akaze", "orb",]  
    results  = {}
    N        = images_clean.size(0)

    for method in methods:
        print(f"Benchmarking {method.upper()} on {N} clean + {N} patched images …")
        tp = fp = tn = fn = 0
        for i in tqdm(range(N), desc=f"  {method} patched", leave=False):
            flag, _, _ = detect_patch(images_patched[i], method=method)
            if flag:
                tp += 1
            else:
                fn += 1
        for i in tqdm(range(N), desc=f"  {method} clean", leave=False):
            flag, _, _ = detect_patch(images_clean[i], method=method)
            if flag:
                fp += 1
            else:
                tn += 1

        detection_rate = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision= tp / max(tp + fp, 1)

        results[method] = {
            "detection_rate": round(100 * detection_rate, 2),
            "fpr":            round(100 * fpr, 2),
            "precision":      round(100 * precision, 2),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        print(f"  {method.upper():5s}  Detection: {100*detection_rate:.1f}%  "
              f"FPR: {100*fpr:.1f}%  Precision: {100*precision:.1f}%")

    path = os.path.join(CHECKPOINT_DIR, "classical_results.pt")
    torch.save(results, path)
    print(f"Results saved {path}")

    _plot_detector_comparison(results)
    return results



def visualise_detection(
    image: torch.Tensor,
    patch: torch.Tensor,
    patch_pos: tuple[int, int],
    method: str = "akaze",
    save_path: str | None = None,
):
    from src.data_loader import denormalise
    from src.attack import apply_patch
    patched, _ = apply_patch(image.unsqueeze(0), patch,
                              positions=[patch_pos], augment=False)
    patched = patched.squeeze(0)

    flag, pred_bbox, info = detect_patch(patched, method=method)

    img_np  = denormalise(patched).cpu().permute(1, 2, 0).numpy()
    img_bgr = _tensor_to_cv(patched)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detector   = _get_detector(method)
    keypoints, _ = detector.detectAndCompute(gray, None)
    kp_img     = cv2.drawKeypoints(
        img_bgr, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    kp_img = cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    r, c = patch_pos
    P = PATCH_SIZE_PX
    rect = plt.Rectangle((c, r), P, P, edgecolor="red", linewidth=2, fill=False)
    axes[0].add_patch(rect)
    axes[0].set_title("Patched image (red = true patch)")
    axes[0].axis("off")
    axes[1].imshow(kp_img)
    if pred_bbox is not None:
        r1, c1, r2, c2 = pred_bbox
        rect2 = plt.Rectangle((c1, r1), c2-c1, r2-r1,
                               edgecolor="#1D9E75", linewidth=2, fill=False)
        axes[1].add_patch(rect2)
    axes[1].set_title(f"{method.upper()} keypoints (green = predicted bbox)\n"
                      f"n_kp={info['n_keypoints']}  detected={flag}")
    axes[1].axis("off")

    axes[2].imshow(info["density_map"], cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Keypoint density map")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, f"classical_{method}_example.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualisation saved {save_path}")


def _plot_detector_comparison(results: dict):
    methods = list(results.keys())
    det     = [results[m]["detection_rate"] for m in methods]
    fpr     = [results[m]["fpr"]            for m in methods]

    x   = np.arange(len(methods))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, det, w, label="Detection rate %", color="#1D9E75")
    ax.bar(x + w/2, fpr, w, label="False positive rate %", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage")
    ax.set_title("Classical detector comparison :Stage 1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "classical_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Comparison plot saved {path}")
