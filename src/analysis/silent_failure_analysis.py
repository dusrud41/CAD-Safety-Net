import os
import pandas as pd


# =========================
# 파일 경로
# =========================
FAILURE_CASES_PATH = (
    "../../outputs/reports/failure_cases.csv"
)

SAVE_DIR = "../../outputs/reports"


# =========================
# Silent Failure 기준
# =========================
CONFIDENCE_THRESHOLD = 0.90


# =========================
# CSV 불러오기
# =========================
df = pd.read_csv(
    FAILURE_CASES_PATH
)

print("Failure Cases Loaded!\n")


# =========================
# Silent Failure 추출
# =========================
silent_failures = df[

    df["confidence_score"]
    >= CONFIDENCE_THRESHOLD
]


# =========================
# 저장
# =========================
os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

SAVE_PATH = os.path.join(
    SAVE_DIR,
    "silent_failures.csv"
)

silent_failures.to_csv(
    SAVE_PATH,
    index=False
)


# =========================
# Dataset별 분석
# =========================
dataset_counts = (
    silent_failures["dataset"]
    .value_counts()
)

print("===================================")
print("Silent Failure Analysis")
print("===================================\n")

print(
    f"Confidence Threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)

print(
    f"\nTotal Silent Failures: "
    f"{len(silent_failures)}\n"
)

print("Silent Failure Count by Dataset:\n")

print(dataset_counts)


# =========================
# 상위 위험 사례 출력
# =========================
print("\n===================================")
print("Top High-Confidence Failures")
print("===================================\n")

top_failures = silent_failures.sort_values(
    by="confidence_score",
    ascending=False
)

print(
    top_failures.head(10)
)


# =========================
# 완료 출력
# =========================
print("\n===================================")
print("Silent Failure CSV Saved!")
print("===================================")

print(f"\nSaved To:\n{SAVE_PATH}")