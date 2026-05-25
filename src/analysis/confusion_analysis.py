import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =========================
# 상위 경로 추가
# =========================
sys.path.append("../")

from datasets.dataset import get_dataloaders
from models.classifier import LungCancerClassifier


# =========================
# Device 설정
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# =========================
# 클래스 이름
# =========================
class_names = [
    "benign",
    "malignant",
    "normal"
]


# =========================
# DataLoader
# =========================
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=16
)


# =========================
# 모델 로드
# =========================
model = LungCancerClassifier(num_classes=3)

MODEL_PATH = (
    "../../outputs/checkpoints/best_model.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully!\n")


# =========================
# 예측 수행
# =========================
all_preds = []

all_labels = []


with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(
            outputs,
            1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# =========================
# Confusion Matrix 생성
# =========================
cm = confusion_matrix(
    all_labels,
    all_preds
)


# =========================
# 시각화
# =========================
fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=True
)

plt.title(
    "Confusion Matrix"
)

SAVE_DIR = (
    "../../outputs/figures"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

SAVE_PATH = os.path.join(
    SAVE_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Classification Report
# =========================
report = classification_report(

    all_labels,
    all_preds,
    target_names=class_names
)

print("===================================")
print("Classification Report")
print("===================================\n")

print(report)


# =========================
# 완료 출력
# =========================
print("\n===================================")
print("Confusion Analysis Completed!")
print("===================================")

print(f"\nSaved To:\n{SAVE_PATH}")