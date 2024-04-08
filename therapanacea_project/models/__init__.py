import torch

from therapanacea_project.models.resnet import get_resnet_architecture
from therapanacea_project.models.vit import get_vit_architecture


def get_model_architecture(
    architecture: str,
    n_classes: int = 1,
    fine_tune: bool = True,
    pretrained: bool = True,
) -> torch.nn.Module:
    """
    Retrieves the specified model architecture.

    Args:
        architecture (str): The name of the architecture to retrieve. Currently supported
            architectures include 'resnet{depth}' for ResNet models and 'vit_b_16' for
            Vision Transformer models.
        n_classes (int, optional): Number of output classes for the model. Defaults to 1.
        fine_tune (bool, optional): If True, allows fine-tuning of the pretrained model's
            parameters. Defaults to True.
        pretrained (bool, optional): If True, loads pretrained weights for the model.
            Defaults to True.

    Raises:
        ValueError: If the specified architecture is not supported.

    Returns:
        torch.nn.Module: The specified model architecture.
    """
    if architecture[:6] == "resnet":
        model = get_resnet_architecture(
            architecture=architecture,
            n_classes=n_classes,
            fine_tune=fine_tune,
            pretrained=pretrained,
        )
    elif architecture == "vit_b_16":
        model = get_vit_architecture(
            architecture=architecture,
            n_classes=n_classes,
            pretrained=pretrained,
        )
    else:
        raise ValueError("Unsupported architecture.")

    return model
