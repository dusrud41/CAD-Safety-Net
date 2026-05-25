import torch
import torch.nn as nn

from torchvision import models


class LungCancerClassifier(nn.Module):

    def __init__(
        self,
        num_classes=3,
        pretrained=True
    ):

        super().__init__()

        # =========================
        # Load ResNet18
        # =========================
        self.model = models.resnet18(
            pretrained=pretrained
        )

        # =========================
        # Replace Final FC Layer
        # =========================
        in_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# =====================================
# Test
# =====================================
if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = LungCancerClassifier()

    model = model.to(device)

    print(model)

    dummy_input = torch.randn(
        4,
        3,
        224,
        224
    ).to(device)

    output = model(dummy_input)

    print("\nOutput Shape:")
    print(output.shape)