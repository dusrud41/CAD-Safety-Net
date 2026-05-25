from torchvision import transforms


# Train Transform
def get_train_transform():

    train_transform = transforms.Compose([

        # Grayscale → RGB 3채널 변환
        transforms.Grayscale(num_output_channels=3),

        # 이미지 크기 통일
        transforms.Resize((224, 224)),

        # Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=10),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),

        # Tensor 변환
        transforms.ToTensor(),

        # ImageNet Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform


# Validation / Test Transform
def get_val_transform():

    val_transform = transforms.Compose([

        # Grayscale → RGB 3채널 변환
        transforms.Grayscale(num_output_channels=3),

        # 이미지 크기 통일
        transforms.Resize((224, 224)),

        # Tensor 변환
        transforms.ToTensor(),

        # ImageNet Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return val_transform