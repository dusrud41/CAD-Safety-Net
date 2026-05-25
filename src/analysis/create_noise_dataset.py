import os
import cv2
import numpy as np

from tqdm import tqdm


# 원본 데이터 경로
INPUT_DIR = "../../data/test"

# Noise 데이터 저장 경로
OUTPUT_DIR = "../../data/failure_cases/noise"


# Noise 강도
NOISE_STD = 25


def create_output_dirs():

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        save_path = os.path.join(
            OUTPUT_DIR,
            cls
        )

        os.makedirs(save_path, exist_ok=True)


def apply_gaussian_noise(image):

    # float 변환
    image = image.astype(np.float32)

    # Gaussian Noise 생성
    noise = np.random.normal(
        loc=0,
        scale=NOISE_STD,
        size=image.shape
    )

    # Noise 추가
    noisy_image = image + noise

    # 범위 제한
    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    return noisy_image.astype(np.uint8)


def process_class_images(class_name):

    input_class_dir = os.path.join(
        INPUT_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    image_files = os.listdir(input_class_dir)

    for image_name in tqdm(image_files):

        input_path = os.path.join(
            input_class_dir,
            image_name
        )

        output_path = os.path.join(
            output_class_dir,
            image_name
        )

        # 이미지 읽기
        image = cv2.imread(input_path)

        if image is None:
            continue

        # Noise 적용
        noisy_image = apply_gaussian_noise(image)

        # 저장
        cv2.imwrite(
            output_path,
            noisy_image
        )


def main():

    print("Creating Noise Failure Dataset...\n")

    create_output_dirs()

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        print(f"Processing {cls} images...")

        process_class_images(cls)

    print("\nNoise Dataset Creation Completed!")


if __name__ == "__main__":

    main()