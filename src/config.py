import torch
import os

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[config] Using device: {DEVICE}")

# Paths 
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RESULTS_DIR     = os.path.join(ROOT_DIR, "results")
PATCH_DIR       = os.path.join(RESULTS_DIR, "patches")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
GRADCAM_DIR     = os.path.join(RESULTS_DIR, "gradcam_examples")
CHECKPOINT_DIR  = os.path.join(RESULTS_DIR, "checkpoints")

for d in [DATA_DIR, PATCH_DIR, PLOTS_DIR, GRADCAM_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

PATCH_PATH      = os.path.join(PATCH_DIR, "patch.pt")
THRESHOLD_PATH  = os.path.join(CHECKPOINT_DIR, "fs_threshold.pt")
BBOX_PATH       = os.path.join(CHECKPOINT_DIR, "keypoint_bboxes.pt")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.pt")

# Data 
IMAGE_SIZE      = 224
BATCH_SIZE      = 32        
NUM_WORKERS     = 4
# ImageNet normalisation constants
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# Model 
MODEL_NAME      = "resnet50"

# Attack 
PATCH_RATIO     = 0.40          # patch covers 16 % of image to 89*89 px at 224
PATCH_SIZE_PX   = int(IMAGE_SIZE * PATCH_RATIO)   # = 89 px
TARGET_CLASS    = 3             # ImageNette class 3 = "chain saw" (ImageNet class 491)
PATCH_STEPS     = 1000           # 500 for dev; bump to 1000 for final run
PATCH_LR        = 0.1
PATCH_BATCH     = 16           # images per patch-training step
PATCH_ROTATION  = 22.5          # degrees of random rotation during training
ATTACK_TEST_N   = 200           # images used to measure attack success rate

# Stage 1 :Classical detection 
GRID_CELLS      = 32          # divide image into 8×8 density grid
KP_SIGMA        = 1.2           # Gaussian smoothing sigma for density map
KP_THRESH_PCTILE= 92         # top-N % of grid cells counted as "peak"
KP_MAX_AREA_FRAC= 0.12         # cluster > 30 % of image ie.probably texture, not patch
CLASSICAL_TEST_N= 500           # 250 clean + 250 patched

# Stage 2 :Feature squeezing 
FS_BIT_DEPTH    = 2             # quantise to 2-bit (4 levels)
FS_MEDIAN_K     = 3             # 3×3 median filter kernel
FS_THRESH_SWEEP = [i/20 for i in range(1, 21)]   # 0.05 to1.00
FS_MAX_FPR      = 0.10          # reject thresholds with FPR > 10 %
FS_VAL_N        = 200           # 100 clean + 100 patched for threshold search

# Stage 3 :GradCAM masking 
GRADCAM_LAYER   = "layer4"      # last conv block of ResNet-50
GRADCAM_THRESH  = 0.15          # top 15 % of activation = "hot"
MASK_FILL       = IMAGENET_MEAN # fill masked region with dataset mean colour
GRADCAM_EXAMPLES= 10            # how many visual examples to save

# Evaluation 
EVAL_TEST_N     = 1000          # images for final comparison table
BASELINES       = ["no_defense", "jpeg", "feature_squeeze",
                   "gradcam_only", "hybrid"]
JPEG_QUALITY    = 75            # JPEG compression quality for baseline

# ImageNette label (0-9) -ImageNet class index
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