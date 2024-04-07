from torch import nn
from torchvision.models import ResNet, resnet18, resnet34, resnet50


def get_resnet_architecture(
    architecture: str = "resnet18",
    n_classes: int = 1,
    fine_tune: bool = True,
    pretrained: bool = True,
) -> ResNet:
    """
    Get a ResNet architecture with specified parameters.

    Args:
        architecture (str): ResNet architecture name.
            ("resnet18", "resnet34", or "resnet50")
        n_classes (int): Number of output classes.
        fine_tune (bool): Whether to fine-tune the model.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        ResNet: ResNet model with specified architecture and modifications.
    """
    if architecture == "resnet18":
        model = resnet18(pretrained=pretrained, progress=False)
    elif architecture == "resnet34":
        model = resnet34(pretrained=pretrained, progress=False)
    elif architecture == "resnet50":
        model = resnet50(pretrained=pretrained, progress=False)
    else:
        raise ValueError(
            "Unsupported ResNet architecture. "
            "Please choose from 'resnet18', 'resnet34', or 'resnet50'."
        )

    for param in model.parameters():
        param.requires_grad = False

    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_classes),
    )

    return model
