import os
import sys

import torch
import numpy as np

from PIL import Image

# =========================
# 상위 경로 추가
# =========================
sys.path.append("../")

from models.classifier import LungCancerClassifier
from datasets.transforms import get_val_transform


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
# 모델 로드
# =========================
model = LungCancerClassifier(
    num_classes=3
)

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

print("OOD Detector Loaded!\n")


# =========================
# 이미지 경로
# =========================
IMAGE_PATH = (
    "../../data/failure_cases/noise/"
    "malignant/Malignant case (10).jpg"
)


# =========================
# 이미지 로드
# =========================
image = Image.open(
    IMAGE_PATH
).convert("RGB")

transform = get_val_transform()

input_tensor = transform(
    image
).unsqueeze(0).to(device)


# =========================
# 추론
# =========================
with torch.no_grad():

    outputs = model(input_tensor)

    probs = torch.softmax(
        outputs,
        dim=1
    )

    confidence, pred = torch.max(
        probs,
        dim=1
    )

confidence = confidence.item()

pred_class = class_names[
    pred.item()
]


# =========================
# MSP 기반 OOD Score
# =========================
ood_score = 1.0 - confidence


# =========================
# Threshold 설정
# =========================
OOD_THRESHOLD = 0.25


# =========================
# Decision Logic
# =========================
if ood_score >= OOD_THRESHOLD:

    decision = "REJECT"

    reason = (
        "Out-of-Distribution "
        "Input Detected"
    )

else:

    decision = "ACCEPT"

    reason = (
        "Input appears "
        "in-distribution"
    )


# =========================
# 출력
# =========================
print("===================================")
print("OOD Detection Result")
print("===================================\n")

print(f"Image: {IMAGE_PATH}\n")

print(f"Prediction: {pred_class}")

print(f"Confidence: {confidence:.4f}")

print(f"OOD Score: {ood_score:.4f}\n")

print(f"Decision: {decision}")

print(f"Reason: {reason}")