import os
import sys

import torch
import torch.nn.functional as F
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm


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
# 테스트용 Transform
# =========================
test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
# 분석할 Dataset 목록
# =========================
datasets_to_evaluate = {

    "original":
        "../../data/test",

    "blur":
        "../../data/failure_cases/blur",

    "noise":
        "../../data/failure_cases/noise",

    "low_quality":
        "../../data/failure_cases/low_quality",

    "contrast_mismatch":
        "../../data/failure_cases/contrast_mismatch"
}


# =========================
# 결과 저장 리스트
# =========================
summary_results = []

failure_cases = []


# =========================
# Dataset별 평가 함수
# =========================
def evaluate_dataset(dataset_name, dataset_path):

    print(f"\nEvaluating [{dataset_name}] Dataset...")

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=test_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    correct = 0
    total = 0

    confidence_scores = []

    with torch.no_grad():

        for image, label in tqdm(dataloader):

            image = image.to(device)
            label = label.to(device)

            # =========================
            # 모델 예측
            # =========================
            output = model(image)

            probabilities = F.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

            true_label = label.item()

            predicted_label = predicted.item()

            confidence_score = confidence.item()

            confidence_scores.append(
                confidence_score
            )

            # =========================
            # Accuracy 계산
            # =========================
            if true_label == predicted_label:
                correct += 1

            total += 1

            # =========================
            # Failure Case 저장
            # =========================
            if true_label != predicted_label:

                image_path = dataset.samples[total - 1][0]

                failure_cases.append({

                    "dataset":
                        dataset_name,

                    "image_path":
                        image_path,

                    "true_label":
                        class_names[true_label],

                    "predicted_label":
                        class_names[predicted_label],

                    "confidence_score":
                        round(confidence_score, 4)
                })

    # =========================
    # 결과 계산
    # =========================
    accuracy = (correct / total) * 100

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores)
    )

    summary_results.append({

        "dataset":
            dataset_name,

        "accuracy":
            round(accuracy, 2),

        "average_confidence":
            round(avg_confidence, 4)
    })

    print(f"\n{dataset_name} Accuracy: {accuracy:.2f}%")

    print(
        f"{dataset_name} Average Confidence: "
        f"{avg_confidence:.4f}"
    )


# =========================
# 전체 Dataset 평가
# =========================
for dataset_name, dataset_path in datasets_to_evaluate.items():

    evaluate_dataset(
        dataset_name,
        dataset_path
    )


# =========================
# 결과 저장 경로 생성
# =========================
SAVE_DIR = "../../outputs/reports"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =========================
# Summary 저장
# =========================
summary_df = pd.DataFrame(summary_results)

summary_save_path = os.path.join(
    SAVE_DIR,
    "failure_analysis_summary.csv"
)

summary_df.to_csv(
    summary_save_path,
    index=False
)


# =========================
# Failure Cases 저장
# =========================
failure_df = pd.DataFrame(failure_cases)

failure_save_path = os.path.join(
    SAVE_DIR,
    "failure_cases.csv"
)

failure_df.to_csv(
    failure_save_path,
    index=False
)


# =========================
# 완료 출력
# =========================
print("\n===================================")
print("Failure Analysis Completed!")
print("===================================")

print(
    f"\nSummary Saved To:\n"
    f"{summary_save_path}"
)

print(
    f"\nFailure Cases Saved To:\n"
    f"{failure_save_path}"
)