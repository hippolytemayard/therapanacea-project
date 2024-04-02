import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
    BinarySpecificity,
)
from torchvision import transforms

from therapanacea_project.dataset.dataloader import get_train_val_dataloaders
from therapanacea_project.models.resnet18 import get_resnet18_architecture
from therapanacea_project.train.training_loop import training_loop
from therapanacea_project.train.validation_loop import validation_loop
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import read_txt_object


def train_model_from_config(
    config: OmegaConf,
    device: torch.device,
):
    save_model_path = (
        Path(config.TRAINING.PATH_MODEL)
        / f"best_model_exp_{config.EXPERIMENT}.pt"
    )

    writer = (
        SummaryWriter(config.TRAINING.TENSORBOARD_DIR)
        if config.TRAINING.ENABLE_TENSORBOARD
        else None
    )

    model = get_resnet18_architecture().to(device)

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

    path_train_label = Path(config.TRAINING.DATASET.PATH_LABELS)
    images_list = Path(config.TRAINING.DATASET.IMAGES_DIR).glob("*.jpg")
    images_list = sorted(list(images_list))

    labels = read_txt_object(
        path_train_label,
    )
    labels = [int(label) for label in labels]

    train_loader, val_loader = get_train_val_dataloaders(
        images_list=images_list,
        labels=labels,
        batch_size=config.TRAINING.BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        val_size=config.TRAINING.DATASET.VALIDATION_SPLIT,
    )

    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.TRAINING.LEARNING_RATE, momentum=0.9
    )

    metrics_collection = MetricCollection(
        [
            # BinaryAccuracy(threshold=0.5),
            # BinaryPrecision(threshold=0.5),
            # BinaryRecall(threshold=0.5),
            # BinarySpecificity(threshold=0.5),
            BinaryROC(thresholds=[0.5]),
        ]
    ).to(device)

    best_hter = 10**3

    for epoch in range(1, config.TRAINING.EPOCHS + 1):

        logging.info(f"EPOCH {epoch}")
        training_loop(
            model=model,
            loader=train_loader,
            criterion=criterion,
            metrics_collection=metrics_collection,
            epoch=epoch,
            optimizer=optimizer,
            writer=writer,
            device=device,
        )

        dict_metrics = validation_loop(
            model=model,
            loader=val_loader,
            epoch=epoch,
            criterion=criterion,
            metrics_collection=metrics_collection,
            writer=writer,
            device=device,
        )

        fpr, tpr, thresholds = dict_metrics["BinaryROC"]
        fnr = 1 - tpr
        hter = (fpr + fnr) / 2

        if hter < best_hter:
            logging.info(
                f"Validation | model improved from {best_hter} to {hter} | saving model"
            )
            best_hter = hter
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "val_metrics": {"fpr": fpr, "fnr": fnr, "hter": hter},
            }
            torch.save(save_dict, save_model_path)

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_path = Path(
        "therapanacea_project/configs/training/training_config.yaml"
    )
    config = load_yaml(config_path)

    make_exists(config.TRAINING.PATH_MODEL)
    make_exists(config.TRAINING.TENSORBOARD_DIR)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"device : {device}")

    train_model_from_config(config=config, device=device)
