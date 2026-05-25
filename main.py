import os


# =====================================
# VENV Python 경로
# =====================================
PYTHON_PATH = (
    "/home/yexngyxung/"
    "dusrud23/"
    "UnderstandingOfImages_26-1/"
    "venv/bin/python3"
)


def run_command(command):

    print("\n===================================")
    print(f"Running: {command}")
    print("===================================\n")

    os.system(command)


if __name__ == "__main__":

    print("\n===================================")
    print("Medical AI Safety Pipeline")
    print("===================================\n")

    # =====================================
    # 1. Training
    # =====================================
    run_command(
        f"cd src/training && "
        f"{PYTHON_PATH} train.py"
    )

    # =====================================
    # 2. Prediction
    # =====================================
    run_command(
        f"cd src/inference && "
        f"{PYTHON_PATH} predict.py"
    )

    # =====================================
    # 3. Failure Analysis
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} failure_analysis.py"
    )

    # =====================================
    # 4. Silent Failure Analysis
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} "
        f"silent_failure_analysis.py"
    )

    # =====================================
    # 5. Confusion Matrix Analysis
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} "
        f"confusion_analysis.py"
    )

    # =====================================
    # 6. Tradeoff Analysis
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} "
        f"tradeoff_analysis.py"
    )

    # =====================================
    # 7. Grad-CAM Analysis
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} "
        f"gradcam_analysis.py"
    )

    # =====================================
    # 8. Evaluation Report
    # =====================================
    run_command(
        f"cd src/analysis && "
        f"{PYTHON_PATH} "
        f"evaluation_report.py"
    )

    # =====================================
    # 9. Final Safety Decision
    # =====================================
    run_command(
        f"cd src/safety && "
        f"{PYTHON_PATH} "
        f"decision_engine.py"
    )

    print("\n===================================")
    print("Pipeline Completed Successfully!")
    print("===================================\n")