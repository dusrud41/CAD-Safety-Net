# README.md

## Silent Failure Detection and Reliability Strategies for Chest CAD Systems

Chest CT–based Computer-Aided Diagnosis (CAD) system with a Safety Net framework for detecting unsafe AI predictions under corrupted or out-of-distribution input conditions.

(흉부 CT 기반 CAD 시스템에 Safety Net 구조를 적용하여 corrupted input 및 OOD 상황에서 위험한 AI 예측을 탐지하는 프로젝트)

---

# 1. Project Overview

This project implements a lung cancer CT classification system using a ResNet18-based CNN model and introduces a safety-aware pipeline for improving prediction reliability.

The system classifies CT images into:

* Normal
* Benign
* Malignant

In addition, the project analyzes:

* Confidence estimation
* Uncertainty estimation
* OOD detection
* Image quality validation
* Silent failure detection

(본 프로젝트는 ResNet18 기반 CNN 모델을 활용하여 폐 CT 영상을 분류하고, 예측 신뢰성을 향상시키기 위한 Safety-Aware Pipeline을 구현한 프로젝트이다.)

---

# 2. Main Features

## Baseline Classification

* ResNet18-based 3-class classifier
* Softmax confidence prediction
* Cross-Entropy training

(ResNet18 기반 3-class 분류 모델)

---

## Failure Case Analysis

* Blur corruption
* Noise corruption
* Contrast mismatch
* Low-quality input analysis

(Blur, Noise 등의 손상 입력 환경 분석)

---

## Safety Net System

* Confidence estimation
* Entropy-based uncertainty analysis
* OOD detection
* Image quality validation
* Rule-based decision engine

(여러 안전 신호를 활용한 Safety Net 시스템)

---

## Explainability

* Grad-CAM visualization
* Attention region analysis

(Grad-CAM 기반 모델 시각화)

---

# 3. Installation

## Required Libraries

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install opencv-python
pip install scikit-learn
pip install tqdm
```

| Library       | Purpose                                  |
| ------------- | ---------------------------------------- |
| torch         | Deep Learning Framework                  |
| torchvision   | Pretrained ResNet18 & Image Utilities    |
| numpy         | Numerical Computation                    |
| pandas        | CSV / Report Processing                  |
| matplotlib    | Visualization                            |
| opencv-python | Image Quality Analysis                   |
| pillow        | Image Loading                            |
| scikit-learn  | Confusion Matrix & Classification Report |
| tqdm          | Progress Bar                             |

(프로젝트 실행에 필요한 라이브러리 설치)

```bash
$ pip install -r requirements.txt
```

('requirements.txt' 실행 시 전체 라이브러리 설치 가능)

The project supports both CPU and GPU execution.

---

# 4. How to Run

Run the following command:

```bash
$ python3 main.py
or
$ python main.py
```

The `main.py` file executes:

* Model loading
* Dataset preprocessing
* Classification
* Safety Net analysis
* Threshold evaluation
* Visualization generation


(`main.py` 실행 시 전체 파이프라인이 자동으로 실행됨)

---

# 5. Project Structure

```text
Project/
│
├── data/
│   │
│   ├── raw/
│   │   │
│   │   └── IQ-OTHNCCD/
│   │       │
│   │       ├── normal/
│   │       │   ├── normal case (1).jpg
│   │       │   ├── normal case (2).jpg
│   │       │   ├── normal case (3).jpg
│   │       │   └── ...
│   │       │
│   │       ├── benign/
│   │       │   ├── benign case (1).jpg
│   │       │   ├── benign case (2).jpg
│   │       │   ├── benign case (3).jpg
│   │       │   └── ...
│   │       │
│   │       └── malignant/
│   │           ├── malignant case (1).jpg
│   │           ├── malignant case (2).jpg
│   │           ├── malignant case (3).jpg
│   │           └── ...
│   │
│   ├── processed/
│   │   │
│   │   ├── resized/
│   │   │   ├── normal/
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   │
│   │   └── augmented/
│   │       ├── normal/
│   │       ├── benign/
│   │       └── malignant/
│   │
│   ├── train/
│   │   │
│   │   ├── normal/
│   │   │   ├── normal case (...).jpg
│   │   │   └── ...
│   │   │
│   │   ├── benign/
│   │   │   ├── benign case (...).jpg
│   │   │   └── ...
│   │   │
│   │   └── malignant/
│   │       ├── malignant case (...).jpg
│   │       └── ...
│   │
│   ├── val/
│   │   │
│   │   ├── normal/
│   │   ├── benign/
│   │   └── malignant/
│   │
│   └── test/
│       │
│       ├── normal/
│       ├── benign/
│       └── malignant/
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── failure_case_visualization.ipynb
│   └── experiments.ipynb
│
├── src/
│   │
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── split_dataset.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── classifier.py
│   │   ├── backbone.py
│   │   └── safety_net.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── losses.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   └── uncertainty.py
│   │
│   ├── safety/
│   │   ├── ood_detector.py
│   │   ├── image_quality.py
│   │   ├── confidence_checker.py
│   │   └── decision_engine.py
│   │
│   ├── analysis/
│   │   ├── failure_analysis.py
│   │   ├── confusion_analysis.py
│   │   ├── create_blur_dataset.py
│   │   ├── create_contrast_mismatch_dataset.py
│   │   ├── create_low_quality_dataset.py
│   │   ├── evaluation_report.py
│   │   ├── silent_failure_analysis.py
│   │   ├── tradeoff_analysis.py
│   │   └── gradcam_analysis.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   └── visualization.py
│   │
│   └── config/
│       └── config.py
│
├── outputs/
│   │
│   ├── checkpoints/
│   │   └── best_model.pth
│   │
│   ├── logs/
│   │
│   ├── predictions/
│   │   └── test_predictions.csv
│   │
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── detection_rate_curve.png
│   │   ├── false_positive_curve.png
│   │   ├── reject_rate_curve.png
│   │   ├── unsafe_acceptance_curve.png
│   │   │
│   │   └── gradcam_examples/
│   │       └── gradcam_result.png
│   │
│   └── reports/
│       ├── evaluation_report.txt
│       ├── failure_analysis_summary.csv
│       ├── failure_case.csv
│       ├── tradeoff_analysis.csv
│       └── silent_failures.csv
│
├── requirements.txt
├── README.md
└── main.py

```

---

# 6. File Description

## main.py

Main execution file for the entire pipeline.

(전체 시스템 실행 메인 파일)

---

## models/

Contains CNN model architectures and model loading code.

Example:

* ResNet18 classifier
* Training utilities

(CNN 모델 구조 및 모델 관련 코드)

---

## datasets/

Dataset loading and preprocessing modules.

Includes:

* Train/test split
* Resize
* Normalization

(데이터셋 로딩 및 전처리 코드)

---

## safety_net/

Core safety modules.

Includes:

* Confidence analysis
* Uncertainty estimation
* OOD detection
* Decision engine
* Threshold optimization

(Safety Net 핵심 모듈)

---

## utils/

Utility functions used across the project.

Examples:

* Metrics
* Logging
* Helper functions

(프로젝트 전반에서 사용하는 보조 함수)

---

## visualization/

Visualization modules.

Includes:

* Grad-CAM
* Confusion matrix
* Threshold graphs

(시각화 관련 코드)

---

## results/

Stores:

* Experimental outputs
* Graphs
* CSV summaries
* Heatmaps

(실험 결과 저장 폴더)

---

# 7. Safety Pipeline

```text
Input CT Image
      ↓
Baseline Classifier
      ↓
Safety Signal Analysis
 ├ Confidence
 ├ Uncertainty
 ├ OOD Detection
 └ Image Quality Check
      ↓
Decision Engine
      ↓
ACCEPT / FLAG / REJECT
```

(입력 영상에 대해 여러 안전 신호를 분석한 후 최종 안전 결정을 수행)

---

# 8. Experimental Analysis

The project evaluates:

* Classification accuracy
* Failure cases under corrupted input
* Unsafe prediction detection
* Safety-performance tradeoff
* Threshold optimization

(분류 성능 및 위험 예측 탐지 성능 분석)

---

# 9. Technologies Used

* Python (Python 3.12)
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

(프로젝트에서 사용한 주요 기술 스택)

---

# 10. Future Work

Future improvements may include:

* Real clinical dataset validation
* Advanced OOD detection algorithms
* Better uncertainty calibration
* Explainable AI enhancement

(향후 실제 의료 환경 적용 및 안전성 향상 연구 가능)
