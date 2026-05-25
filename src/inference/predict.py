import os
import sys

import torch
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm


# 상위 경로 추가
sys.path.append("../")

from datasets.dataset import get_dataloaders
from models.classifier import LungCancerClassifier


# Device 설정
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
# 모델 불러오기
# =========================
model = LungCancerClassifier(num_classes=3)

MODEL_PATH = "../../outputs/checkpoints/best_model.pth"

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
# DataLoader
# =========================
_, _, test_loader = get_dataloaders(
    batch_size=1
)


# =========================
# 결과 저장 리스트
# =========================
results = []


# =========================
# Prediction 시작
# =========================
with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        # 모델 출력
        outputs = model(images)

        # Softmax 확률 계산
        probabilities = F.softmax(
            outputs,
            dim=1
        )

        # 최고 확률 및 예측 클래스
        confidence, predicted = torch.max(
            probabilities,
            1
        )

        true_label = labels.item()

        predicted_label = predicted.item()

        confidence_score = confidence.item()

        # 결과 저장
        results.append({

            "true_label":
                class_names[true_label],

            "predicted_label":
                class_names[predicted_label],

            "confidence_score":
                round(confidence_score, 4),

            "correct":
                true_label == predicted_label
        })


# =========================
# DataFrame 생성
# =========================
df = pd.DataFrame(results)


# =========================
# 저장 경로 생성
# =========================
SAVE_DIR = "../../outputs/predictions"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# CSV 저장
# =========================
SAVE_PATH = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

df.to_csv(
    SAVE_PATH,
    index=False
)

print("\nPrediction Results Saved!")

print(f"\nSaved to: {SAVE_PATH}")


# =========================
# Accuracy 계산
# =========================
accuracy = df["correct"].mean() * 100

print(f"\nTest Accuracy: {accuracy:.2f}%")