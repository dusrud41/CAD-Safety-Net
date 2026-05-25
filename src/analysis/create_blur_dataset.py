import os
import cv2

from tqdm import tqdm


# 원본 데이터 경로
INPUT_DIR = "../../data/test"

# Blur 데이터 저장 경로
OUTPUT_DIR = "../../data/failure_cases/blur"


# Blur 강도
KERNEL_SIZE = (9, 9)


def create_output_dirs():

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        save_path = os.path.join(
            OUTPUT_DIR,
            cls
        )

        os.makedirs(save_path, exist_ok=True)


def apply_blur(image):

    blurred = cv2.GaussianBlur(
        image,
        KERNEL_SIZE,
        sigmaX=0
    )

    return blurred


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

        # Blur 적용
        blurred_image = apply_blur(image)

        # 저장
        cv2.imwrite(
            output_path,
            blurred_image
        )


def main():

    print("Creating Blur Failure Dataset...\n")

    create_output_dirs()

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        print(f"Processing {cls} images...")

        process_class_images(cls)

    print("\nBlur Dataset Creation Completed!")


if __name__ == "__main__":

    main()