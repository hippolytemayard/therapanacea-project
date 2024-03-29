from torchvision.models import resnet18, ResNet


def get_resnet18_architecture(
    n_classes: int = 1, fine_tune: bool = True, pretrained: bool = True
) -> ResNet:

    model = resnet18(pretrained=pretrained, progress=False)
    from torch import nn

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_classes),
    )

    return model
