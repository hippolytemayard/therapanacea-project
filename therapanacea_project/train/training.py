import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
    BinarySpecificity,
    Specificity,
)

from therapanacea_project.dataset.dataloader import get_train_val_dataloaders
from therapanacea_project.models.resnet18 import get_resnet18_architecture
from therapanacea_project.train.training_loop import training_loop
from therapanacea_project.train.validation_loop import validation_loop
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import read_txt_object

from therapanacea_project.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from therapanacea_project.metrics.utils import instantiate_metrics_from_config


def train_model_from_config(
    config: OmegaConf,
    device: torch.device,
):

    writer = (
        SummaryWriter(config.TRAINING.TENSORBOARD_DIR)
        if config.TRAINING.ENABLE_TENSORBOARD
        else None
    )

    model = get_resnet18_architecture(
        n_classes=config.TRAINING.DATASET.N_CLASSES,
        fine_tune=config.TRAINING.FINE_TUNE,
        pretrained=config.TRAINING.PRETRAINED,
    ).to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"The model has {trainable_params} trainable parameters")

    path_train_label = Path(config.TRAINING.DATASET.PATH_LABELS)
    images_list = Path(config.TRAINING.DATASET.IMAGES_DIR).glob("*.jpg")
    images_list = sorted(list(images_list))

    labels = read_txt_object(
        path_train_label,
    )
    labels = [int(label) for label in labels]

    train_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.TRAINING
    )
    val_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.VALIDATION
    )

    train_loader, val_loader, train_labels_distribution = (
        get_train_val_dataloaders(
            images_list=images_list,
            labels=labels,
            batch_size=config.TRAINING.BATCH_SIZE,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            val_size=config.TRAINING.DATASET.VALIDATION_SPLIT,
        )
    )

    logging.info(
        f"Train dataset classes distribution: {train_labels_distribution}"
    )

    weight = (
        torch.tensor(
            [
                sum(train_labels_distribution.values()) / n_samples
                for _, n_samples in train_labels_distribution.items()
            ]
        )
        if config.TRAINING.DATASET.WEIGHT_LOSS
        else None
    )

    logging.info(f"weight: {weight}")

    # criterion = nn.BCELoss(weight=weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.13]).to(device))
    optimizer = optim.SGD(
        model.parameters(), lr=config.TRAINING.LEARNING_RATE, momentum=0.9
    )

    metrics_collection = instantiate_metrics_from_config(
        metrics_config=config.VALIDATION.METRICS
    ).to(device)

    best_hter = 10**3

    for epoch in range(1, config.TRAINING.EPOCHS + 1):

        logging.info(f"EPOCH {epoch}")
        training_loop(
            model=model,
            loader=train_loader,
            criterion=criterion,
            with_logits=config.TRAINING.WITH_LOGITS,
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
            with_logits=config.TRAINING.WITH_LOGITS,
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
            torch.save(
                save_dict, Path(config.TRAINING.PATH_MODEL) / "best_model.pt"
            )

    return


if __name__ == "__main__":
    config_path = Path(
        "therapanacea_project/configs/training/training_config.yaml"
    )
    config = load_yaml(config_path)

    make_exists(config.EXPERIMENT_FOLDER)
    make_exists(config.ROOT_EXPERIMENT)
    make_exists(config.TRAINING.PATH_MODEL)
    make_exists(config.TRAINING.TENSORBOARD_DIR)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.TRAINING.PATH_LOGS),
            logging.StreamHandler(),
        ],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device : {device}")

    train_model_from_config(config=config, device=device)
