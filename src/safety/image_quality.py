import cv2
import numpy as np


# =========================
# Blur Detection
# =========================
def detect_blur(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return laplacian_var


# =========================
# Noise Estimation
# =========================
def estimate_noise(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    noise = np.std(gray)

    return noise


# =========================
# Brightness Check
# =========================
def check_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    return brightness


# =========================
# 전체 품질 분석
# =========================
def analyze_image_quality(image_path):

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            "Failed to load image."
        )

    blur_score = detect_blur(image)

    noise_score = estimate_noise(image)

    brightness_score = check_brightness(image)

    return {

        "blur_score":
            round(blur_score, 2),

        "noise_score":
            round(noise_score, 2),

        "brightness_score":
            round(brightness_score, 2)
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

    result = analyze_image_quality(
        TEST_IMAGE
    )

    print("===================================")
    print("Image Quality Analysis")
    print("===================================\n")

    print(f"Image: {TEST_IMAGE}\n")

    print(
        f"Blur Score: "
        f"{result['blur_score']}"
    )

    print(
        f"Noise Score: "
        f"{result['noise_score']}"
    )

    print(
        f"Brightness Score: "
        f"{result['brightness_score']}"
    )