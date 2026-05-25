import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


# 상위 경로 추가
sys.path.append("../")

from datasets.dataset import get_dataloaders
from models.classifier import LungCancerClassifier


# Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10


# DataLoader 불러오기
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)


# 모델 생성
model = LungCancerClassifier(num_classes=3)

model = model.to(device)


# Loss Function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Best Accuracy 저장 변수
best_val_accuracy = 0.0


# Checkpoint 저장 경로
CHECKPOINT_DIR = "../../outputs/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# Training Loop
for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    # =========================
    # Train
    # =========================
    model.train()

    train_loss = 0.0

    correct = 0
    total = 0

    train_bar = tqdm(train_loader)

    for images, labels in train_bar:

        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss 계산
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Weight 업데이트
        optimizer.step()

        train_loss += loss.item()

        # Accuracy 계산
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        train_bar.set_description(
            f"Loss: {loss.item():.4f}"
        )

    train_accuracy = 100 * correct / total

    avg_train_loss = train_loss / len(train_loader)

    # =========================
    # Validation
    # =========================
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    # =========================
    # 결과 출력
    # =========================
    print(f"Train Loss: {avg_train_loss:.4f}")

    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # =========================
    # Best Model 저장
    # =========================
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Best model saved!")

print("\nTraining Completed.")