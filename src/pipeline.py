import os
import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import (
    DEVICE, JPEG_QUALITY, RESULTS_DIR, IMAGENET_MEAN, IMAGENET_STD,
    EVAL_TEST_N
)
from src.classical      import detect_patch
from src.feature_squeeze import detect_fs, compute_fs_score
from src.gradcam_defense import (
    GradCAM, mask_and_reclassify, mask_image, get_gradcam_bbox
)


def jpeg_compress(image: torch.Tensor, quality: int = JPEG_QUALITY) -> torch.Tensor:
    from src.data_loader import denormalise
    from torchvision.transforms.functional import to_tensor

    img01 = denormalise(image.cpu())
    pil = Image.fromarray((img01.permute(1,2,0).numpy() * 255).astype(np.uint8))

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    img_t = to_tensor(compressed).to(image.device)

    mean = torch.tensor(IMAGENET_MEAN, device=image.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=image.device)[:, None, None]
    return (img_t - mean) / std


class HybridDefense:

    def __init__(self, model, gradcam: GradCAM, fs_threshold: float,
                 kp_method: str = "akaze"):
        self.model = model
        self.gradcam = gradcam
        self.fs_threshold = fs_threshold
        self.kp_method = kp_method

    @torch.no_grad()
    def _classify(self, image: torch.Tensor) -> int:
        logits = self.model(image.unsqueeze(0).to(DEVICE))
        return logits.argmax(dim=1).item()

    def defend(self, image: torch.Tensor, true_label: int | None = None) -> dict:
        image = image.to(DEVICE)

        kp_flag, kp_bbox, _ = detect_patch(image, method=self.kp_method)

        if not kp_flag:
            pred = self._classify(image)
            return {
                "prediction": pred,
                "correct": pred == true_label if true_label is not None else None,
                "stage_reached": 1,
                "kp_flag": False,
                "fs_flag": False,
                "kp_bbox": None,
                "mask_bbox": None,
            }

        fs_flag, fs_score = detect_fs(self.model, image, self.fs_threshold)

        if not fs_flag:
            pred = self._classify(image)
            return {
                "prediction": pred,
                "correct": pred == true_label if true_label is not None else None,
                "stage_reached": 2,
                "kp_flag": True,
                "fs_flag": False,
                "kp_bbox": kp_bbox,
                "mask_bbox": None,
            }

        result = mask_and_reclassify(
            self.model, self.gradcam, image,
            true_label=true_label if true_label is not None else -1,
            kp_bbox=kp_bbox,
        )

        return {
            "prediction": result["predicted_after"],
            "correct": result["correct_after"],
            "stage_reached": 3,
            "kp_flag": True,
            "fs_flag": True,
            "kp_bbox": kp_bbox,
            "mask_bbox": result["mask_bbox"],
        }


def run_full_evaluation(
    model,
    gradcam: GradCAM,
    images_clean: torch.Tensor,
    images_patched: torch.Tensor,
    labels: torch.Tensor,
    fs_threshold: float,
    kp_method: str = "akaze",
) -> dict:
    model.eval()
    defense = HybridDefense(model, gradcam, fs_threshold, kp_method=kp_method)
    N = labels.size(0)
    
    labels_cpu = labels.cpu()
    labels_dev = labels.to(DEVICE)

    results = {}

    print("\nEvaluating: no defense")
    with torch.no_grad():
        clean_preds = model(images_clean.to(DEVICE)).argmax(1).cpu()
        patched_preds = model(images_patched.to(DEVICE)).argmax(1).cpu()

    clean_successes = (clean_preds == labels_cpu).sum().item()
    patched_successes = (patched_preds == labels_cpu).sum().item()
    
    from src.evaluate import bootstrap_ci_metric_pair
    ci_metrics = bootstrap_ci_metric_pair(
        clean_successes=clean_successes,
        patched_successes=patched_successes,
        fp_successes=0,  # No FPR for no defense
        total=N
    )
    
    results["no_defense"] = {
        "clean_acc": ci_metrics["clean_acc"]["value"],
        "clean_ci_low": ci_metrics["clean_acc"]["ci_low"],
        "clean_ci_high": ci_metrics["clean_acc"]["ci_high"],
        "attack_sr": ci_metrics["attack_sr"]["value"],
        "attack_ci_low": ci_metrics["attack_sr"]["ci_low"],
        "attack_ci_high": ci_metrics["attack_sr"]["ci_high"],
        "recovered_acc": ci_metrics["recovered_acc"]["value"],
        "rec_ci_low": ci_metrics["recovered_acc"]["ci_low"],
        "rec_ci_high": ci_metrics["recovered_acc"]["ci_high"],
        "fpr": 0.0,
        "fpr_ci_low": 0.0,
        "fpr_ci_high": 0.0,
    }
    print(f"  Clean acc: {ci_metrics['clean_acc']['value']:.1f}% "
          f"(95% CI: {ci_metrics['clean_acc']['ci_low']:.1f}-{ci_metrics['clean_acc']['ci_high']:.1f})")

    print("Evaluating: JPEG compression")
    jpeg_clean_successes = 0
    jpeg_patched_successes = 0
    
    for i in tqdm(range(N), desc="  JPEG evaluation", leave=False):
        img_j_patched = jpeg_compress(images_patched[i].cpu())
        with torch.no_grad():
            pred_patched = model(img_j_patched.unsqueeze(0).to(DEVICE)).argmax(1).cpu().item()
        if pred_patched == labels_cpu[i].item():
            jpeg_patched_successes += 1
    
        img_j_clean = jpeg_compress(images_clean[i].cpu())
        with torch.no_grad():
            pred_clean = model(img_j_clean.unsqueeze(0).to(DEVICE)).argmax(1).cpu().item()
        if pred_clean == labels_cpu[i].item():
            jpeg_clean_successes += 1

    jpeg_fp = N - jpeg_clean_successes
    
    ci_metrics = bootstrap_ci_metric_pair(
        clean_successes=jpeg_clean_successes,
        patched_successes=jpeg_patched_successes,
        fp_successes=jpeg_fp,
        total=N
    )
    
    results["jpeg"] = {
        "clean_acc": ci_metrics["clean_acc"]["value"],
        "clean_ci_low": ci_metrics["clean_acc"]["ci_low"],
        "clean_ci_high": ci_metrics["clean_acc"]["ci_high"],
        "attack_sr": ci_metrics["attack_sr"]["value"],
        "attack_ci_low": ci_metrics["attack_sr"]["ci_low"],
        "attack_ci_high": ci_metrics["attack_sr"]["ci_high"],
        "recovered_acc": ci_metrics["recovered_acc"]["value"],
        "rec_ci_low": ci_metrics["recovered_acc"]["ci_low"],
        "rec_ci_high": ci_metrics["recovered_acc"]["ci_high"],
        "fpr": ci_metrics["fpr"]["value"],
        "fpr_ci_low": ci_metrics["fpr"]["ci_low"],
        "fpr_ci_high": ci_metrics["fpr"]["ci_high"],
    }
    print(f"  Clean acc: {ci_metrics['clean_acc']['value']:.1f}%  "
          f"Recovered: {ci_metrics['recovered_acc']['value']:.1f}%")

    # 3. Feature squeezing only 
    print("Evaluating: feature squeezing only")
    from src.feature_squeeze import detect_fs_batch

    fs_flags_clean, _ = detect_fs_batch(model, images_clean.to(DEVICE), fs_threshold)
    fs_flags_patched, _ = detect_fs_batch(model, images_patched.to(DEVICE), fs_threshold)

    fs_fp = sum(fs_flags_clean)
    fs_tp = sum(fs_flags_patched)
    fs_clean_successes = N - fs_fp
    fs_patched_successes = fs_tp  # For FS, "recovered" = detected
    
    ci_metrics = bootstrap_ci_metric_pair(
        clean_successes=fs_clean_successes,
        patched_successes=fs_patched_successes,
        fp_successes=fs_fp,
        total=N
    )
    
    results["feature_squeeze"] = {
        "clean_acc": ci_metrics["clean_acc"]["value"],
        "clean_ci_low": ci_metrics["clean_acc"]["ci_low"],
        "clean_ci_high": ci_metrics["clean_acc"]["ci_high"],
        "attack_sr": ci_metrics["attack_sr"]["value"],
        "attack_ci_low": ci_metrics["attack_sr"]["ci_low"],
        "attack_ci_high": ci_metrics["attack_sr"]["ci_high"],
        "recovered_acc": ci_metrics["recovered_acc"]["value"],
        "rec_ci_low": ci_metrics["recovered_acc"]["ci_low"],
        "rec_ci_high": ci_metrics["recovered_acc"]["ci_high"],
        "fpr": ci_metrics["fpr"]["value"],
        "fpr_ci_low": ci_metrics["fpr"]["ci_low"],
        "fpr_ci_high": ci_metrics["fpr"]["ci_high"],
    }
    print(f"  Detection rate: {ci_metrics['recovered_acc']['value']:.1f}%  "
          f"FPR: {ci_metrics['fpr']['value']:.1f}%")

    print("Evaluating: GradCAM only")
    gc_patched_successes = 0
    gc_clean_successes = 0
    
    for i in tqdm(range(N), desc="  GradCAM evaluation", leave=False):
        image_patched = images_patched[i].to(DEVICE)
        image_clean = images_clean[i].to(DEVICE)
        label = labels_cpu[i].item()
        
        with torch.no_grad():
            pred_before = model(image_patched.unsqueeze(0)).argmax(1).cpu().item()
        try:
            heatmap, gc_bbox = get_gradcam_bbox(gradcam, image_patched, pred_before)
            if gc_bbox is not None:
                masked = mask_image(image_patched, gc_bbox)
                with torch.no_grad():
                    pred_after = model(masked.unsqueeze(0)).argmax(1).cpu().item()
            else:
                pred_after = pred_before
        except:
            pred_after = pred_before
        if pred_after == label:
            gc_patched_successes += 1
        
        with torch.no_grad():
            pred_before_clean = model(image_clean.unsqueeze(0)).argmax(1).cpu().item()
        try:
            heatmap, gc_bbox = get_gradcam_bbox(gradcam, image_clean, pred_before_clean)
            if gc_bbox is not None:
                masked = mask_image(image_clean, gc_bbox)
                with torch.no_grad():
                    pred_after_clean = model(masked.unsqueeze(0)).argmax(1).cpu().item()
            else:
                pred_after_clean = pred_before_clean
        except:
            pred_after_clean = pred_before_clean
        if pred_after_clean == label:
            gc_clean_successes += 1

    gc_fp = N - gc_clean_successes
    
    ci_metrics = bootstrap_ci_metric_pair(
        clean_successes=gc_clean_successes,
        patched_successes=gc_patched_successes,
        fp_successes=gc_fp,
        total=N
    )
    
    results["gradcam_only"] = {
        "clean_acc": ci_metrics["clean_acc"]["value"],
        "clean_ci_low": ci_metrics["clean_acc"]["ci_low"],
        "clean_ci_high": ci_metrics["clean_acc"]["ci_high"],
        "attack_sr": ci_metrics["attack_sr"]["value"],
        "attack_ci_low": ci_metrics["attack_sr"]["ci_low"],
        "attack_ci_high": ci_metrics["attack_sr"]["ci_high"],
        "recovered_acc": ci_metrics["recovered_acc"]["value"],
        "rec_ci_low": ci_metrics["recovered_acc"]["ci_low"],
        "rec_ci_high": ci_metrics["recovered_acc"]["ci_high"],
        "fpr": ci_metrics["fpr"]["value"],
        "fpr_ci_low": ci_metrics["fpr"]["ci_low"],
        "fpr_ci_high": ci_metrics["fpr"]["ci_high"],
    }
    print(f"  Clean acc: {ci_metrics['clean_acc']['value']:.1f}%  "
          f"Recovered: {ci_metrics['recovered_acc']['value']:.1f}%")

    print(" Evaluating: hybrid pipeline (all 3 stages)")
    hybrid_patched_successes = 0
    hybrid_fp = 0

    for i in tqdm(range(N), desc="  Hybrid evaluation", leave=False):
        res_patched = defense.defend(images_patched[i], true_label=labels_cpu[i].item())
        if res_patched["correct"]:
            hybrid_patched_successes += 1
        
        res_clean = defense.defend(images_clean[i], true_label=labels_cpu[i].item())
        if not res_clean["correct"]:
            hybrid_fp += 1

    hybrid_clean_successes = N - hybrid_fp
    
    ci_metrics = bootstrap_ci_metric_pair(
        clean_successes=hybrid_clean_successes,
        patched_successes=hybrid_patched_successes,
        fp_successes=hybrid_fp,
        total=N
    )
    
    results["hybrid"] = {
        "clean_acc": ci_metrics["clean_acc"]["value"],
        "clean_ci_low": ci_metrics["clean_acc"]["ci_low"],
        "clean_ci_high": ci_metrics["clean_acc"]["ci_high"],
        "attack_sr": ci_metrics["attack_sr"]["value"],
        "attack_ci_low": ci_metrics["attack_sr"]["ci_low"],
        "attack_ci_high": ci_metrics["attack_sr"]["ci_high"],
        "recovered_acc": ci_metrics["recovered_acc"]["value"],
        "rec_ci_low": ci_metrics["recovered_acc"]["ci_low"],
        "rec_ci_high": ci_metrics["recovered_acc"]["ci_high"],
        "fpr": ci_metrics["fpr"]["value"],
        "fpr_ci_low": ci_metrics["fpr"]["ci_low"],
        "fpr_ci_high": ci_metrics["fpr"]["ci_high"],
    }
    print(f"  Recovered: {ci_metrics['recovered_acc']['value']:.1f}%  "
          f"FPR: {ci_metrics['fpr']['value']:.1f}%")
    path = os.path.join(RESULTS_DIR, "results_with_ci.pt")
    torch.save(results, path)
    print(f"\n Results with CIs saved {path}")

    return results

def _print_table(results: dict):
    print("\n" + "=" * 72)
    print(f"{'Method':<22} {'Clean acc':>10} {'Attack SR':>10} "
          f"{'Recovered':>10} {'FPR':>8}")
    print("-" * 72)
    order = ["no_defense", "jpeg", "feature_squeeze", "gradcam_only", "hybrid"]
    for method in order:
        if method not in results:
            continue
        r = results[method]
        marker = " ★" if method == "hybrid" else ""
        print(f"{method + marker:<22} {r['clean_acc']:>9.1f}% "
              f"{r['attack_sr']:>9.1f}% "
              f"{r['recovered_acc']:>9.1f}% "
              f"{r['fpr']:>7.1f}%")
    print("-" * 72)