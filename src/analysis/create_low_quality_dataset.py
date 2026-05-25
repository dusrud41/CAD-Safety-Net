import os
import cv2

from tqdm import tqdm


# 원본 데이터 경로
INPUT_DIR = "../../data/test"

# 저장 경로
OUTPUT_DIR = "../../data/failure_cases/low_quality"


# JPEG 압축 품질
JPEG_QUALITY = 10

# 다운스케일 비율
SCALE_FACTOR = 0.4


def create_output_dirs():

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        save_path = os.path.join(
            OUTPUT_DIR,
            cls
        )

        os.makedirs(save_path, exist_ok=True)


def apply_low_quality(image):

    height, width = image.shape[:2]

    # =========================
    # 1. 해상도 감소
    # =========================
    small = cv2.resize(
        image,
        (
            int(width * SCALE_FACTOR),
            int(height * SCALE_FACTOR)
        )
    )

    # =========================
    # 2. 다시 원래 크기로 복원
    # =========================
    restored = cv2.resize(
        small,
        (width, height)
    )

    # =========================
    # 3. JPEG 압축 적용
    # =========================
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        JPEG_QUALITY
    ]

    _, encoded_img = cv2.imencode(
        ".jpg",
        restored,
        encode_param
    )

    decoded_img = cv2.imdecode(
        encoded_img,
        1
    )

    return decoded_img


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

        # 저화질 변환
        low_quality_image = apply_low_quality(image)

        # 저장
        cv2.imwrite(
            output_path,
            low_quality_image
        )


def main():

    print("Creating Low Quality Failure Dataset...\n")

    create_output_dirs()

    classes = ["normal", "benign", "malignant"]

    for cls in classes:

        print(f"Processing {cls} images...")

        process_class_images(cls)

    print("\nLow Quality Dataset Creation Completed!")


if __name__ == "__main__":

    main()