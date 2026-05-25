import os
import sys

import cv2
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

print("Integrated Safety Net Loaded!\n")


# =========================
# 이미지 경로
# =========================
IMAGE_PATH = (
    "../../data/failure_cases/noise/"
    "malignant/Malignant case (10).jpg"
)


# =========================
# Image Quality Analysis
# =========================
def analyze_image_quality(image_path):

    gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Blur
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Noise
    noise_score = np.std(gray)

    # Brightness
    brightness_score = np.mean(gray)

    return (
        blur_score,
        noise_score,
        brightness_score
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
# Uncertainty 계산
# =========================
entropy = -torch.sum(

    probs * torch.log(probs + 1e-10)

).item()


# =========================
# OOD Score 계산
# =========================
ood_score = 1.0 - confidence


# =========================
# Quality Score 계산
# =========================
blur_score, noise_score, brightness_score = (
    analyze_image_quality(
        IMAGE_PATH
    )
)


# =========================
# Decision Engine
# =========================
decision = "ACCEPT"

reasons = []


# -------------------------
# Reject Conditions
# -------------------------
if confidence < 0.70:

    decision = "REJECT"

    reasons.append(
        "Low confidence"
    )

if entropy > 0.50:

    decision = "REJECT"

    reasons.append(
        "High uncertainty"
    )

if noise_score > 70:

    decision = "REJECT"

    reasons.append(
        "Severe image noise"
    )

if blur_score < 100:

    decision = "REJECT"

    reasons.append(
        "Blurred image"
    )

if ood_score > 0.30:

    decision = "REJECT"

    reasons.append(
        "Possible OOD input"
    )


# -------------------------
# FLAG Conditions
# -------------------------
if decision != "REJECT":

    if confidence < 0.90:

        decision = "FLAG"

        reasons.append(
            "Moderate confidence"
        )

    if entropy > 0.25:

        decision = "FLAG"

        reasons.append(
            "Moderate uncertainty"
        )


# =========================
# 결과 출력
# =========================
print("===================================")
print("Final Safety Net Result")
print("===================================\n")

print(f"Image: {IMAGE_PATH}\n")

print(f"Prediction: {pred_class}")

print(f"Confidence: {confidence:.4f}")

print(f"Uncertainty: {entropy:.4f}")

print(f"OOD Score: {ood_score:.4f}")

print(f"Blur Score: {blur_score:.2f}")

print(f"Noise Score: {noise_score:.2f}")

print(f"Brightness Score: {brightness_score:.2f}\n")

print(f"Decision: {decision}")

print("\nReasons:")

for reason in reasons:

    print(f"- {reason}")