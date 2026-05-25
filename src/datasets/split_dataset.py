"""
데이터 자동 분할 코드
"""

import os
import random
import shutil

# 랜덤 고정
random.seed(42)

# 원본 데이터 경로
SOURCE_DIR = "../../data/raw/IQ-OTHNCCD"

# 분할 데이터 저장 경로
BASE_OUTPUT_DIR = "../../data"
z
# 데이터 분할 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 클래스 목록
classes = ["normal", "benign", "malignant"]

# train / val / test 폴더 생성
for split in ["train", "val", "test"]:
    split_path = os.path.join(BASE_OUTPUT_DIR, split)

    os.makedirs(split_path, exist_ok=True)

    for cls in classes:
        os.makedirs(os.path.join(split_path, cls), exist_ok=True)

# 클래스별 데이터 분할
for cls in classes:

    class_dir = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_dir)

    # 이미지 섞기
    random.shuffle(images)

    total_count = len(images)

    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # 파일 복사
    for split_name, split_images in split_data.items():

        for image_name in split_images:

            src_path = os.path.join(class_dir, image_name)

            dst_path = os.path.join(
                BASE_OUTPUT_DIR,
                split_name,
                cls,
                image_name
            )

            shutil.copy2(src_path, dst_path)

    print(f"\n[{cls.upper()}]")
    print(f"Total: {total_count}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

print("\nDataset split completed successfully.")