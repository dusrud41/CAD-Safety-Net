# src/config/config.py

import torch


# =========================================
# Device
# =========================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================
# Dataset Paths
# =========================================
TRAIN_DIR = "../../data/train"
VAL_DIR = "../../data/val"
TEST_DIR = "../../data/test"

FAILURE_CASE_DIR = "../../data/failure_cases"


# =========================================
# Processed Dataset Paths
# =========================================
BLUR_DIR = (
    "../../data/processed/blur"
)

NOISE_DIR = (
    "../../data/processed/noise"
)

LOW_QUALITY_DIR = (
    "../../data/processed/low_quality"
)

CONTRAST_MISMATCH_DIR = (
    "../../data/processed/contrast_mismatch"
)


# =========================================
# Output Paths
# =========================================
CHECKPOINT_DIR = (
    "../../outputs/checkpoints"
)

FIGURE_DIR = (
    "../../outputs/figures"
)

REPORT_DIR = (
    "../../outputs/reports"
)

PREDICTION_DIR = (
    "../../outputs/predictions"
)


# =========================================
# Model Settings
# =========================================
MODEL_NAME = "resnet18"

NUM_CLASSES = 3

CLASS_NAMES = [
    "benign",
    "malignant",
    "normal"
]


# =========================================
# Image Settings
# =========================================
IMAGE_SIZE = 224


# =========================================
# Training Settings
# =========================================
BATCH_SIZE = 16

LEARNING_RATE = 0.0001

NUM_EPOCHS = 10

RANDOM_SEED = 42


# =========================================
# Safety Thresholds
# =========================================
CONFIDENCE_THRESHOLD = 0.95

UNCERTAINTY_THRESHOLD = 0.30

OOD_THRESHOLD = 0.50

NOISE_THRESHOLD = 50.0

BLUR_THRESHOLD = 100.0


# =========================================
# Checkpoint Paths
# =========================================
BEST_MODEL_PATH = (
    "../../outputs/checkpoints/best_model.pth"
)

LATEST_MODEL_PATH = (
    "../../outputs/checkpoints/latest_model.pth"
)


# =========================================
# Report Files
# =========================================
FAILURE_ANALYSIS_CSV = (
    "../../outputs/reports/"
    "failure_analysis_summary.csv"
)

SILENT_FAILURE_CSV = (
    "../../outputs/reports/"
    "silent_failures.csv"
)

TRADEOFF_ANALYSIS_CSV = (
    "../../outputs/reports/"
    "tradeoff_analysis.csv"
)

EVALUATION_REPORT_TXT = (
    "../../outputs/reports/"
    "evaluation_report.txt"
)