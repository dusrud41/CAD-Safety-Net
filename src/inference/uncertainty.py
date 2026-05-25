import os
import sys

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

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
# Entropy 계산
# =========================
def calculate_entropy(probabilities):

    entropy = -torch.sum(
        probabilities *
        torch.log(probabilities + 1e-10)
    )

    return entropy.item()


# =========================
# 예측 함수
# =========================
def predict_with_uncertainty(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

        uncertainty = calculate_entropy(
            probabilities
        )

    predicted_class = class_names[
        predicted.item()
    ]

    return {

        "prediction":
            predicted_class,

        "confidence":
            round(confidence.item(), 4),

        "uncertainty":
            round(uncertainty, 4),

        "probabilities":
            probabilities.cpu().numpy()
    }


# =========================
# 테스트 실행
# =========================
if __name__ == "__main__":

    TEST_IMAGE = (
        "../../data/failure_cases/noise/"
        "malignant/"
        "Malignant case (10).jpg"
    )

    result = predict_with_uncertainty(
        TEST_IMAGE
    )

    print("===================================")
    print("Uncertainty Analysis")
    print("===================================\n")

    print(f"Image: {TEST_IMAGE}\n")

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}"
    )

    print(
        f"Uncertainty: "
        f"{result['uncertainty']}"
    )

    print("\nClass Probabilities:")

    probs = result["probabilities"][0]

    for idx, prob in enumerate(probs):

        print(
            f"{class_names[idx]}: "
            f"{prob:.4f}"
        )