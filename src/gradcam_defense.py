import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.config import (
    DEVICE, GRADCAM_LAYER, GRADCAM_THRESH,
    GRADCAM_DIR, IMAGE_SIZE,
    GRADCAM_EXAMPLES
)


class GradCAM:
    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.handles = []
        
        base = model.base if hasattr(model, "base") else model
        target_layer = None
        
        for name, module in base.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found")
        
        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
    
    def __call__(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        
        original_mode = self.model.training
        self.model.eval()
        if image.dim() == 3:
            x = image.unsqueeze(0).to(DEVICE)
        else:
            x = image.to(DEVICE)
        
        x = x.clone().detach().requires_grad_(True)
        
        # Clear gradients
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        
        # Forward pass
        logits = self.model(x)
        score = logits[0, target_class]
        
        # Backward pass 
        score.backward(retain_graph=True)
        
        # Check gradients
        if self.gradients is None:
            try:
                grads = torch.autograd.grad(score, self.model.parameters(), 
                                           allow_unused=True, retain_graph=True)
                print("Warning: Using fallback gradient method")
            except:
                raise RuntimeError("Could not compute gradients. Make sure model is trainable.")
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )
    
        cam = cam.squeeze().cpu().detach().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        self.model.train(original_mode)
        
        return cam


def _heatmap_to_bbox(heatmap: np.ndarray, thresh: float = GRADCAM_THRESH):
    threshold_val = np.percentile(heatmap, 100 * (1 - thresh))
    hot = heatmap >= threshold_val
    
    rows, cols = np.where(hot)
    if len(rows) == 0:
        return None
    
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _intersect_bboxes(bbox1, bbox2):
    if bbox1 is None and bbox2 is None:
        return None
    if bbox1 is None:
        return bbox2
    if bbox2 is None:
        return bbox1
    
    r1 = max(bbox1[0], bbox2[0])
    c1 = max(bbox1[1], bbox2[1])
    r2 = min(bbox1[2], bbox2[2])
    c2 = min(bbox1[3], bbox2[3])
    
    if r2 > r1 and c2 > c1:
        return r1, c1, r2, c2
    return bbox1


def get_gradcam_bbox(gradcam, image, target_class):
    heatmap = gradcam(image, target_class)
    bbox = _heatmap_to_bbox(heatmap)
    return heatmap, bbox


def mask_image(image, bbox):
    masked = image.clone()
    r1, c1, r2, c2 = bbox
    # Clamp to valid range
    r1, c1 = max(0, r1), max(0, c1)
    r2, c2 = min(IMAGE_SIZE-1, r2), min(IMAGE_SIZE-1, c2)
    if r2 > r1 and c2 > c1:
        masked[:, r1:r2+1, c1:c2+1] = 0.0
    return masked


def mask_and_reclassify(model, gradcam, image, true_label, kp_bbox=None):
    model.eval()
    
    with torch.no_grad():
        logits_before = model(image.unsqueeze(0).to(DEVICE))
        pred_before = logits_before.argmax(dim=1).item()
    
    try:
        heatmap, gradcam_bbox = get_gradcam_bbox(gradcam, image, pred_before)
    except Exception as e:
        print(f"GradCAM failed: {e}, skipping")
        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        gradcam_bbox = None
    
    mask_bbox = _intersect_bboxes(gradcam_bbox, kp_bbox)
    
    if mask_bbox is None:
        return {
            "predicted_before": pred_before,
            "predicted_after": pred_before,
            "correct_after": pred_before == true_label,
            "heatmap": heatmap,
            "mask_bbox": None,
            "gradcam_bbox": gradcam_bbox,
        }
    
    masked = mask_image(image.to(DEVICE), mask_bbox)
    
    with torch.no_grad():
        logits_after = model(masked.unsqueeze(0))
        pred_after = logits_after.argmax(dim=1).item()
    
    return {
        "predicted_before": pred_before,
        "predicted_after": pred_after,
        "correct_after": pred_after == true_label,
        "heatmap": heatmap,
        "mask_bbox": mask_bbox,
        "gradcam_bbox": gradcam_bbox,
    }

def _save_example(image, result, label, idx):
    import matplotlib.pyplot as plt
    import os
    from src.config import GRADCAM_DIR
    from src.data_loader import denormalise
    
    img_np = denormalise(image.cpu()).permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original (pred={result['predicted_before']}, true={label})")
    axes[0].axis("off")
    
    axes[1].imshow(img_np)
    axes[1].imshow(result["heatmap"], cmap="jet", alpha=0.5)
    if result["gradcam_bbox"]:
        r1, c1, r2, c2 = result["gradcam_bbox"]
        rect = plt.Rectangle((c1, r1), c2-c1, r2-r1, fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title("GradCAM heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(img_np)
    if result["mask_bbox"]:
        r1, c1, r2, c2 = result["mask_bbox"]
        rect = plt.Rectangle((c1, r1), c2-c1, r2-r1, fill=True, facecolor='gray', alpha=0.5)
        axes[2].add_patch(rect)
    axes[2].set_title(f"Masked {result['predicted_after']} ({'done' if result['correct_after'] else 'failed'})")
    axes[2].axis("off")
    
    plt.tight_layout()
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    plt.savefig(os.path.join(GRADCAM_DIR, f"example_{idx:02d}.png"), dpi=150)
    plt.close()

def reclassify_batch(model, gradcam, images_patched, labels, kp_bboxes=None, save_examples=False):
  
    N = images_patched.size(0)
    correct_after = 0
    correct_before = 0
    saved = 0 
    
    print(f"Processing {N} images...")
    
    for i in tqdm(range(N), desc="Reclassifying"):
        image = images_patched[i].to(DEVICE)
        label = labels[i].item()
        kp_bbox = kp_bboxes[i] if kp_bboxes else None
        
        result = mask_and_reclassify(model, gradcam, image, label, kp_bbox=kp_bbox)
        
        if result["predicted_before"] == label:
            correct_before += 1
        if result["correct_after"]:
            correct_after += 1
        if save_examples and saved < GRADCAM_EXAMPLES and result["mask_bbox"] is not None:
            _save_example(images_patched[i], result, label, idx=saved)
            saved += 1
        
    
    recovered_acc = 100 * correct_after / N
    patched_acc = 100 * correct_before / N
    
    print(f"Patched acc (before masking): {patched_acc:.1f}%")
    print(f"Recovered acc (after masking): {recovered_acc:.1f}%")
    
    return {
        "n_images": N, 
        "patched_acc": patched_acc, 
        "recovered_acc": recovered_acc
    }


def gradcam_only_defense(model, gradcam, images_patched, labels):
    return reclassify_batch(
        model, gradcam, images_patched, labels,
        kp_bboxes=None, save_examples=False
    )