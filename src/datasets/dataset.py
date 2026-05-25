import os

from torchvision import datasets
from torch.utils.data import DataLoader

from datasets.transforms import (
    get_train_transform,
    get_val_transform
)


# 데이터 경로 설정
BASE_DATA_DIR = "../../data"

TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "val")
TEST_DIR = os.path.join(BASE_DATA_DIR, "test")


def get_datasets():

    # Train Dataset
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transform()
    )

    # Validation Dataset
    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_val_transform()
    )

    # Test Dataset
    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_val_transform()
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=16, num_workers=2):

    train_dataset, val_dataset, test_dataset = get_datasets()

    # Train Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation Loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Test Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print("Dataset Loaded Successfully!\n")

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 클래스 이름 출력
    train_dataset, _, _ = get_datasets()

    print("\nClasses:")
    print(train_dataset.classes)