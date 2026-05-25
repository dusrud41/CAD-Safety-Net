import os
import pandas as pd


# =========================
# 파일 경로
# =========================
FAILURE_SUMMARY_PATH = (
    "../../outputs/reports/"
    "failure_analysis_summary.csv"
)

SILENT_FAILURE_PATH = (
    "../../outputs/reports/"
    "silent_failures.csv"
)

TRADEOFF_PATH = (
    "../../outputs/reports/"
    "tradeoff_analysis.csv"
)

REPORT_SAVE_PATH = (
    "../../outputs/reports/"
    "evaluation_report.txt"
)


# =========================
# CSV 로드
# =========================
failure_df = pd.read_csv(
    FAILURE_SUMMARY_PATH
)

silent_df = pd.read_csv(
    SILENT_FAILURE_PATH
)

tradeoff_df = pd.read_csv(
    TRADEOFF_PATH
)


# =========================
# 기본 Accuracy 결과
# =========================
original_acc = failure_df[
    failure_df["dataset"] == "original"
]["accuracy"].values[0]

blur_acc = failure_df[
    failure_df["dataset"] == "blur"
]["accuracy"].values[0]

noise_acc = failure_df[
    failure_df["dataset"] == "noise"
]["accuracy"].values[0]

low_quality_acc = failure_df[
    failure_df["dataset"] == "low_quality"
]["accuracy"].values[0]

contrast_acc = failure_df[
    failure_df["dataset"] == "contrast_mismatch"
]["accuracy"].values[0]


# =========================
# Silent Failure 분석
# =========================
total_silent_failures = len(
    silent_df
)

top_failure = silent_df.iloc[0]

top_failure_dataset = (
    top_failure["dataset"]
)

top_failure_confidence = (
    top_failure["confidence_score"]
)


# =========================
# Tradeoff 결과
# =========================
best_threshold_row = tradeoff_df.sort_values(

    by="unsafe_acceptance_rate"

).iloc[0]

best_threshold = (
    best_threshold_row["threshold"]
)

best_detection_rate = (
    best_threshold_row["detection_rate"]
)

best_reject_rate = (
    best_threshold_row["reject_rate"]
)


# =========================
# Report 작성
# =========================
report = f"""
===================================
Medical AI Safety Evaluation Report
===================================

1. Base Classification Performance
-----------------------------------

Original Accuracy:
{original_acc:.2f}%

Blur Accuracy:
{blur_acc:.2f}%

Noise Accuracy:
{noise_acc:.2f}%

Low Quality Accuracy:
{low_quality_acc:.2f}%

Contrast Mismatch Accuracy:
{contrast_acc:.2f}%


2. Silent Failure Analysis
-----------------------------------

Total Silent Failures:
{total_silent_failures}

Highest Confidence Failure:
Dataset: {top_failure_dataset}

Confidence:
{top_failure_confidence:.4f}


3. Tradeoff Analysis
-----------------------------------

Recommended Threshold:
{best_threshold:.2f}

Detection Rate:
{best_detection_rate:.4f}

Reject Rate:
{best_reject_rate:.4f}


4. Key Findings
-----------------------------------

- The model achieved high accuracy
  on the original dataset.

- Noise corruption caused
  significant performance degradation.

- High-confidence incorrect predictions
  (silent failures) were observed.

- Confidence alone was insufficient
  for reliable medical AI prediction.

- MSP-based OOD detection
  showed limitations.

- Multi-factor safety validation
  improved robustness.


5. Final Conclusion
-----------------------------------

This project demonstrates that
medical AI systems require
more than classification accuracy.

Reliable deployment requires:
- confidence analysis
- uncertainty estimation
- image quality validation
- OOD detection
- integrated safety decision systems

===================================
"""


# =========================
# 저장
# =========================
with open(

    REPORT_SAVE_PATH,
    "w"

) as f:

    f.write(report)


# =========================
# 출력
# =========================
print("===================================")
print("Evaluation Report Generated!")
print("===================================\n")

print(report)

print(f"\nSaved To:\n{REPORT_SAVE_PATH}")