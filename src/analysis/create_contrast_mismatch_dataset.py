import os
import cv2
import numpy as np

from tqdm import tqdm


# 원본 데이터 경로
INPUT_DIR = "../../data/test"

# 저장 경로
OUTPUT_DIR = "../../data/failure_cases/contrast_mismatch"


# Contrast / Brightness 설정
ALPHA = 0.5   # contrast 감소
BETA = 20     # brightness 증가


def create_output_dirs():

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        save_path = os.path.join(
            OUTPUT_DIR,
            cls
        )

        os.makedirs(save_path, exist_ok=True)


def apply_contrast_mismatch(image):

    # contrast 및 brightness 변환
    transformed = cv2.convertScaleAbs(
        image,
        alpha=ALPHA,
        beta=BETA
    )

    return transformed


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

        # Contrast mismatch 적용
        mismatch_image = apply_contrast_mismatch(image)

        # 저장
        cv2.imwrite(
            output_path,
            mismatch_image
        )


def main():

    print("Creating Contrast Mismatch Dataset...\n")

    create_output_dirs()

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        print(f"Processing {cls} images...")

        process_class_images(cls)

    print("\nContrast Mismatch Dataset Creation Completed!")


if __name__ == "__main__":

    main()