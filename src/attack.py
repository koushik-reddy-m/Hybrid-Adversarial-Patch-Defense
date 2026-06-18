import os
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, PATCH_SIZE_PX, TARGET_CLASS, PATCH_STEPS, PATCH_LR,
    PATCH_BATCH, PATCH_ROTATION, PATCH_PATH, PLOTS_DIR,
    ATTACK_TEST_N, IMAGENET_MEAN, IMAGENET_STD,
)

IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
}


TARGET_IMAGENET = IMAGENETTE_TO_IMAGENET[TARGET_CLASS]

def random_transform_patch(patch: torch.Tensor, max_angle: float = PATCH_ROTATION) -> torch.Tensor:
    
    angle = (torch.rand(1).item() * 2 - 1) * max_angle
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    theta = torch.tensor([[cos_a, -sin_a, 0],
                           [sin_a,  cos_a, 0]], dtype=torch.float32)
    theta = theta.unsqueeze(0).to(patch.device)
    grid  = F.affine_grid(theta, patch.unsqueeze(0).size(), align_corners=False)
    rotated = F.grid_sample(patch.unsqueeze(0), grid,
                             align_corners=False, padding_mode="border").squeeze(0)

    factor = 0.7 + torch.rand(1).item() * 0.6   # [0.7, 1.3]
    return (rotated * factor).clamp(0, 1)


def apply_patch(
    images: torch.Tensor,
    patch:  torch.Tensor,
    positions: list[tuple[int, int]] | None = None,
    augment: bool = False,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:

    B, C, H, W = images.shape
    P = patch.shape[-1]

    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=DEVICE)[:, None, None]

    result    = images.clone()
    used_pos  = []

    for i in range(B):
        p = random_transform_patch(patch) if augment else patch.clone()
        p_norm = (p.to(DEVICE) - mean) / std   # put patch in normalised space

        if positions is not None:
            row, col = positions[i]
        else:
            row = torch.randint(0, H - P, (1,)).item()
            col = torch.randint(0, W - P, (1,)).item()
        row = min(row, H - P)
        col = min(col, W - P)

        result[i, :, row:row+P, col:col+P] = p_norm
        used_pos.append((row, col))

    return result, used_pos

def train_patch(
    model,
    data_loader,
    steps: int = PATCH_STEPS,
    save: bool = True,
) -> torch.Tensor:

    print(f"Training patch: {steps} steps, patch size {PATCH_SIZE_PX}×{PATCH_SIZE_PX}")

    patch = torch.rand(3, PATCH_SIZE_PX, PATCH_SIZE_PX,
                       requires_grad=True, device=DEVICE)

    optimizer = torch.optim.Adam([patch], lr=PATCH_LR)
    model.eval()

    from src.model import get_base_model
    base_model = get_base_model()

    data_iter   = iter(data_loader)
    losses      = []
    target_t    = torch.tensor([TARGET_IMAGENET] * PATCH_BATCH, device=DEVICE)

    for step in tqdm(range(steps), desc="Patch training"):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, _ = next(data_iter)

        images = images[:PATCH_BATCH].to(DEVICE)

        optimizer.zero_grad()

        patched, _ = apply_patch(images, patch.clamp(0, 1), augment=True)
        logits = base_model(patched)
        loss = F.cross_entropy(logits, target_t[:images.size(0)])
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            print(f"  step {step+1:4d}/{steps}  loss: {avg:.4f}")

    final_patch = patch.detach().clamp(0, 1)

    if save:
        torch.save(final_patch, PATCH_PATH)
        print(f"Patch saved {PATCH_PATH}")
        _save_patch_image(final_patch)
        _plot_loss(losses)

    return final_patch


def load_patch() -> torch.Tensor:
    if not os.path.exists(PATCH_PATH):
        raise FileNotFoundError(
            f"No patch found at {PATCH_PATH}. Run train_patch() first."
        )
    patch = torch.load(PATCH_PATH, map_location=DEVICE)
    print(f"Patch loaded from {PATCH_PATH}  shape: {tuple(patch.shape)}")
    return patch



def evaluate_attack(
    model,
    patch: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    model.eval()
    B = images.size(0)

    with torch.no_grad():
        clean_preds  = model(images).argmax(dim=1)
        clean_acc    = (clean_preds == labels).float().mean().item()

        patched, _ = apply_patch(images, patch, augment=False)
        patch_preds = model(patched).argmax(dim=1)

    correctly_classified = (clean_preds == labels)
    now_wrong = (patch_preds != labels)
    attack_sr = (correctly_classified & now_wrong).float().sum().item() / max(correctly_classified.sum().item(), 1)

    target_rate = (patch_preds == TARGET_CLASS).float().mean().item()

    result = {
        "clean_acc":   round(100 * clean_acc,   2),
        "attack_sr":   round(100 * attack_sr,   2),
        "target_rate": round(100 * target_rate, 2),
        "n_images":    B,
    }
    print(f"Clean acc: {result['clean_acc']:.1f}%  "
          f"Attack SR: {result['attack_sr']:.1f}%  "
          f"Target rate: {result['target_rate']:.1f}%")
    return result



def _save_patch_image(patch: torch.Tensor):
    p_np = patch.cpu().permute(1, 2, 0).numpy()
    path = os.path.join(PLOTS_DIR, "trained_patch.png")
    plt.figure(figsize=(3, 3))
    plt.imshow(p_np)
    plt.axis("off")
    plt.title("Trained adversarial patch")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Patch image saved  {path}")


def _plot_loss(losses: list[float]):
    path = os.path.join(PLOTS_DIR, "patch_training_loss.png")
    plt.figure(figsize=(8, 3))
    plt.plot(losses, linewidth=0.8, color="#D85A30")
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Adversarial patch training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved  {path}")
