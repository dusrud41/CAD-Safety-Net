import os
import sys

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader

# =========================
# 상위 경로 추가
# =========================
sys.path.append("../")

from datasets.transforms import get_val_transform
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
# 평가할 데이터셋 경로
# =========================
dataset_paths = {

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
# Threshold 범위
# =========================
thresholds = np.arange(
    0.1,
    1.0,
    0.05
)

results = []


# =========================
# Threshold Loop
# =========================
for threshold in thresholds:

    total_cases = 0

    correct_malignant = 0

    total_malignant = 0

    false_positive = 0

    total_non_malignant = 0

    rejected_cases = 0

    unsafe_acceptances = 0

    # -------------------------
    # Dataset Loop
    # -------------------------
    for dataset_name, dataset_path in dataset_paths.items():

        print(
            f"\nEvaluating [{dataset_name}] "
            f"| threshold={threshold:.2f}"
        )

        dataset = datasets.ImageFolder(

            root=dataset_path,
            transform=get_val_transform()
        )

        loader = DataLoader(

            dataset,
            batch_size=16,
            shuffle=False
        )

        # =====================
        # Inference
        # =====================
        for images, labels in tqdm(loader):

            images = images.to(device)

            labels = labels.to(device)

            with torch.no_grad():

                outputs = model(images)

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                confidences, preds = torch.max(
                    probs,
                    dim=1
                )

            # =====================
            # Sample-level analysis
            # =====================
            for i in range(len(labels)):

                total_cases += 1

                true_label = labels[i].item()

                pred_label = preds[i].item()

                confidence = (
                    confidences[i].item()
                )

                # ---------------------
                # Reject if confidence low
                # ---------------------
                if confidence < threshold:

                    rejected_cases += 1
                    continue

                # ---------------------
                # malignant detection
                # ---------------------
                if true_label == 1:

                    total_malignant += 1

                    if pred_label == 1:

                        correct_malignant += 1

                    else:

                        unsafe_acceptances += 1

                # ---------------------
                # false positive
                # ---------------------
                else:

                    total_non_malignant += 1

                    if pred_label == 1:

                        false_positive += 1

    # =========================
    # Metric 계산
    # =========================
    detection_rate = (

        correct_malignant
        / (total_malignant + 1e-10)
    )

    false_positive_rate = (

        false_positive
        / (total_non_malignant + 1e-10)
    )

    reject_rate = (

        rejected_cases
        / (total_cases + 1e-10)
    )

    unsafe_acceptance_rate = (

        unsafe_acceptances
        / (total_malignant + 1e-10)
    )

    results.append({

        "threshold":
            threshold,

        "detection_rate":
            detection_rate,

        "false_positive_rate":
            false_positive_rate,

        "reject_rate":
            reject_rate,

        "unsafe_acceptance_rate":
            unsafe_acceptance_rate
    })


# =========================
# DataFrame 생성
# =========================
results_df = pd.DataFrame(
    results
)

print("\n===================================")
print("Failure-aware Tradeoff Analysis")
print("===================================\n")

print(results_df)


# =========================
# 저장 경로
# =========================
SAVE_DIR = "../../outputs/figures"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =========================
# Detection Rate Graph
# =========================
plt.figure(figsize=(8, 6))

plt.plot(

    results_df["threshold"],
    results_df["detection_rate"],
    marker="o"
)

plt.xlabel("Threshold")

plt.ylabel("Detection Rate")

plt.title(
    "Detection Rate vs Threshold"
)

plt.grid(True)

plt.savefig(

    os.path.join(
        SAVE_DIR,
        "detection_rate_curve.png"
    ),

    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# False Positive Rate Graph
# =========================
plt.figure(figsize=(8, 6))

plt.plot(

    results_df["threshold"],
    results_df["false_positive_rate"],
    marker="o"
)

plt.xlabel("Threshold")

plt.ylabel("False Positive Rate")

plt.title(
    "False Positive Rate vs Threshold"
)

plt.grid(True)

plt.savefig(

    os.path.join(
        SAVE_DIR,
        "false_positive_curve.png"
    ),

    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Reject Rate Graph
# =========================
plt.figure(figsize=(8, 6))

plt.plot(

    results_df["threshold"],
    results_df["reject_rate"],
    marker="o"
)

plt.xlabel("Threshold")

plt.ylabel("Reject Rate")

plt.title(
    "Reject Rate vs Threshold"
)

plt.grid(True)

plt.savefig(

    os.path.join(
        SAVE_DIR,
        "reject_rate_curve.png"
    ),

    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Unsafe Acceptance Graph
# =========================
plt.figure(figsize=(8, 6))

plt.plot(

    results_df["threshold"],
    results_df["unsafe_acceptance_rate"],
    marker="o"
)

plt.xlabel("Threshold")

plt.ylabel("Unsafe Acceptance Rate")

plt.title(
    "Unsafe Acceptance Rate vs Threshold"
)

plt.grid(True)

plt.savefig(

    os.path.join(
        SAVE_DIR,
        "unsafe_acceptance_curve.png"
    ),

    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# CSV 저장
# =========================
CSV_PATH = (
    "../../outputs/reports/tradeoff_analysis.csv"
)

results_df.to_csv(
    CSV_PATH,
    index=False
)


# =========================
# 완료 출력
# =========================
print("\n===================================")
print("Tradeoff Analysis Completed!")
print("===================================")

print(f"\nSaved CSV:\n{CSV_PATH}")

print(
    "\nSaved Figures:\n"
    "detection_rate_curve.png\n"
    "false_positive_curve.png\n"
    "reject_rate_curve.png\n"
    "unsafe_acceptance_curve.png"
)