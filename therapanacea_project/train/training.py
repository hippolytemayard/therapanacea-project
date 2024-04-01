import logging
from pathlib import Path

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)
from torchvision import transforms

from therapanacea_project.dataset.dataloader import (
    get_train_val_dataloaders,
)
from therapanacea_project.models.resnet18 import get_resnet18_architecture
from therapanacea_project.train.training_loop import training_loop
from therapanacea_project.train.validation_loop import validation_loop
from therapanacea_project.utils.io import read_txt_object

if __name__ == "__main__":
    writer = SummaryWriter("./experiments/")
    device = "cpu"

    model = get_resnet18_architecture()
    model = model.to(device)

    loss_training, acc_training = [], []
    loss_validation, acc_validation = [], []

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    path_train_label = Path("./data/label_train.txt")
    images_list = Path("./data/train_img/").glob("*.jpg")
    images_list = sorted(list(images_list))

    labels = read_txt_object(
        path_train_label,
    )
    labels = [int(label) for label in labels]

    train_loader, val_loader = get_train_val_dataloaders(
        images_list=images_list,
        labels=labels,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )

    epochs = 3
    lr = 0.01
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    metrics_collection = MetricCollection(
        [
            BinaryAccuracy(threshold=0.5),
            BinaryPrecision(threshold=0.5),
            BinaryRecall(threshold=0.5),
        ]
    )

    for epoch in range(1, epochs + 1):

        logging.info(f"EPOCH {epoch}")
        print(f"epoch : {epoch}")

        training_loop(
            model=model,
            loader=train_loader,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            writer=writer,
            device=device,
        )

        # validation_loop(
        #    model=model,
        #    loader=train_loader,
        #    epoch=epoch,
        #    criterion=criterion,
        #    step="Validation",
        #    writer=writer,
        #    device=device,
        # )
        validation_loop(
            model=model,
            train_loader=val_loader,
            epoch=epoch,
            criterion=criterion,
            step="Validation",
            writer=writer,
            device=device,
        )
