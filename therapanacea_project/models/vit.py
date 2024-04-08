import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor


def get_vit_architecture(
    architecture: str = "vit_b_16",
    n_classes: int = 1,
    pretrained: bool = True,
) -> "ViT":
    """
    Get a ViT architecture with specified parameters.

    Args:
        architecture (str): ResNet architecture name.
        n_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        ResNet: ResNet model with specified architecture and modifications.
    """
    if architecture == "vit_b_16":
        model = ViT(
            n_classes=n_classes,
            pretrained=pretrained,
        )
    else:
        raise ValueError("Unsupported architecture. ")

    return model


class ViT(torch.nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.feature_extractor, feature_dim = self.get_feature_extractor()
        # print(f"Feature Extractor: {self.feature_extractor}")
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.n_classes),
        )

    def get_feature_extractor(self) -> tuple[torch.nn.Module, int]:

        vit_b_16_pretrained = vit_b_16(pretrained=self.pretrained)
        for parameter in vit_b_16_pretrained.parameters():
            parameter.requires_grad = False

        feature_extractor = create_feature_extractor(
            vit_b_16_pretrained, return_nodes=["getitem_5"]
        )
        feature_dim = 768

        return feature_extractor, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is (batch_size, 3, h, w)
        """
        batch_size = x.size(0)
        features = self.feature_extractor(x).get("getitem_5")
        features = features.view(batch_size, -1)
        out = self.classifier_head(features)

        return out
