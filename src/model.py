import torch
import torch.nn as nn
from torchvision import models

from src.config import DEVICE

# ImageNette label (0-9) to ImageNet class index
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}

IMAGENET_TO_IMAGENETTE = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}
IMAGENETTE_INDICES = list(IMAGENETTE_TO_IMAGENET.values())  # the 10 ImageNet cols we care about


class ImageNetteResNet(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.indices = torch.tensor(IMAGENET_INDICES_LIST, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_1000 = self.base(x)                          # (B, 1000)
        return logits_1000[:, self.indices]                  # (B, 10)

    @property
    def layer4(self):
        return self.base.layer4

_model_cache: nn.Module | None = None
IMAGENET_INDICES_LIST = IMAGENET_TO_IMAGENETTE  
IMAGENET_INDICES_LIST = IMAGENETTE_INDICES       # correct assignment


def get_model(eval_mode: bool = True) -> ImageNetteResNet:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    print("Loading pretrained ResNet-50 …")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    base    = models.resnet50(weights=weights)
    base    = base.to(DEVICE)

    model = ImageNetteResNet(base).to(DEVICE)

    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    _model_cache = model
    print("Ready")
    return model


def get_base_model() -> nn.Module:
    m = get_model()
    return m.base
