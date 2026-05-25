import os
import sys

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# =========================
# pytorch-gradcam
# =========================
from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

# =========================
# 상위 경로 추가
# =========================
sys.path.append("../")

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
# Transform
# =========================
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
# 분석할 이미지
# =========================
IMAGE_PATH = (
    "../../data/failure_cases/noise/"
    "malignant/"
    "Malignant case (10).jpg"
)


# =========================
# 이미지 로드
# =========================
rgb_image = Image.open(
    IMAGE_PATH
).convert("RGB")

rgb_image = rgb_image.resize(
    (224, 224)
)

rgb_image = np.array(rgb_image) / 255.0

input_tensor = transform(
    Image.fromarray(
        (rgb_image * 255).astype(np.uint8)
    )
).unsqueeze(0).to(device)


# =========================
# 예측 수행
# =========================
with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

predicted_class = class_names[
    predicted.item()
]

confidence_score = confidence.item()

print("===================================")
print("Prediction Result")
print("===================================\n")

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence_score:.4f}\n")


# =========================
# Grad-CAM 대상 레이어
# =========================
target_layers = [
    model.model.layer4[-1]
]


# =========================
# Grad-CAM 생성
# =========================
cam = GradCAM(
    model=model,
    target_layers=target_layers
)


targets = [
    ClassifierOutputTarget(
        predicted.item()
    )
]


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# =========================
# Heatmap 생성
# =========================
visualization = show_cam_on_image(

    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# =========================
# 저장
# =========================
SAVE_DIR = (
    "../../outputs/figures/gradcam_examples"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

SAVE_PATH = os.path.join(
    SAVE_DIR,
    "gradcam_result.png"
)

cv2.imwrite(

    SAVE_PATH,

    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)


# =========================
# 결과 출력
# =========================
print("===================================")
print("Grad-CAM Analysis Completed!")
print("===================================\n")

print(f"Saved To:\n{SAVE_PATH}")