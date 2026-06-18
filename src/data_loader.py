import os
import tarfile
import urllib.request
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from src.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, DEVICE
)

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _download_imagenette():
    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    if os.path.isdir(imagenette_dir):
        print("ImageNette already downloaded.")
        return imagenette_dir

    os.makedirs(DATA_DIR, exist_ok=True)
    url      = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tgz_path = os.path.join(DATA_DIR, "imagenette2-320.tgz")

    print(" Downloading ImageNette-320 (~1.5 GB) …")

    def _reporthook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // max(total_size, 1))
        if count % 300 == 0:
            print(f"  {pct}%", end="\r", flush=True)

    urllib.request.urlretrieve(url, tgz_path, reporthook=_reporthook)
    print("\nExtracting …")

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    os.remove(tgz_path)
    print(" Done.")
    return imagenette_dir


def get_loaders(download: bool = True):
    imagenette_dir = _download_imagenette() if download else os.path.join(DATA_DIR, "imagenette2-320")

    train_root = os.path.join(imagenette_dir, "train")
    val_root   = os.path.join(imagenette_dir, "val")

    full_train = datasets.ImageFolder(train_root, transform=TRAIN_TRANSFORM)
    test_ds    = datasets.ImageFolder(val_root,   transform=EVAL_TRANSFORM)

    n       = len(full_train)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    split    = int(0.8 * n)
    train_ds = Subset(full_train, indices[:split])

    val_full = datasets.ImageFolder(train_root, transform=EVAL_TRANSFORM)
    val_ds   = Subset(val_full, indices[split:])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def get_test_subset(n: int = 200, seed: int = 0) -> tuple:
    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    val_root = os.path.join(imagenette_dir, "val")
    test_ds  = datasets.ImageFolder(val_root, transform=EVAL_TRANSFORM)

    np.random.seed(seed)
    indices = np.random.choice(len(test_ds), size=min(n, len(test_ds)), replace=False)
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=n, shuffle=False, num_workers=NUM_WORKERS)

    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device)
    if tensor.dim() == 3:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    else:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    return (tensor * std + mean).clamp(0, 1)


def verify_clean_accuracy(model, loader, max_batches: int = 20) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= max_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100 * correct / total
    print(f" Clean accuracy: {acc:.1f}%  ({correct}/{total})")
    return acc
