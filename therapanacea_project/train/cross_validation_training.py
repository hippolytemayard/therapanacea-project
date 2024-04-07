import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    Accuracy,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
    BinarySpecificity,
    Specificity,
)

from therapanacea_project.dataset.dataloader import get_single_dataloader
from therapanacea_project.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from therapanacea_project.losses import losses_dict
from therapanacea_project.metrics.utils import instantiate_metrics_from_config
from therapanacea_project.models.resnet18 import get_resnet18_architecture
from therapanacea_project.optimizer import optimizer_dict
from therapanacea_project.train.training_loop import training_loop
from therapanacea_project.train.validation_loop import validation_loop
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import read_txt_object


def cross_validation_training_from_config(
    config: OmegaConf,
    device: Union[str, torch.device],
) -> None:
    """
    Perform cross-validation training based on the provided configuration.

    Args:
        config (OmegaConf): Configuration object containing training
            parameters.
        device (Union[str, torch.device]): Device to use for training
            (e.g., 'cpu' or 'cuda').

    Returns:
        None
    """

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

    skf = StratifiedKFold(n_splits=config.TRAINING.DATASET.KFOLD)

    logging.info(f"Starting {config.TRAINING.DATASET.KFOLD}-Fold training")

    for fold_id, (train_index, val_index) in enumerate(
        skf.split(images_list, labels)
    ):

        train_images, val_images = (
            np.asarray(images_list)[train_index].tolist(),
            np.asarray(images_list)[val_index].tolist(),
        )
        train_labels, val_labels = (
            np.asarray(labels)[train_index].tolist(),
            np.asarray(labels)[val_index].tolist(),
        )

        train_labels_distribution = {
            0: len(train_labels) - sum(train_labels),
            1: sum(train_labels),
        }

        logging.info(
            f"Train dataset classes distribution: {train_labels_distribution}"
        )

        train_loader = get_single_dataloader(
            images_list=train_images,
            labels=train_labels,
            transform=train_transforms,
            batch_size=config.TRAINING.BATCH_SIZE,
        )
        val_loader = get_single_dataloader(
            images_list=val_images,
            labels=val_labels,
            transform=val_transforms,
            batch_size=config.TRAINING.BATCH_SIZE,
        )

        logging.info(f"Training Fold {fold_id}")

        train_one_fold(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels_distribution=train_labels_distribution,
            fold_id=fold_id,
            device=device,
        )


def train_one_fold(
    config: OmegaConf,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels_distribution: dict[int, int],
    fold_id: int,
    device: Union[str, torch.device],
) -> None:
    """
    Perform training for a single fold.

    Args:
        config (OmegaConf): Configuration object containing training parameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        train_labels_distribution (Dict[int, int]): Distribution of training labels.
        fold_id (int): Fold ID.
        device (torch.device): Device to use for training.

    Returns:
        None
    """

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

    pos_weight = (
        sum(train_labels_distribution.values()) / train_labels_distribution[1]
    ) / (sum(train_labels_distribution.values()) / train_labels_distribution[0])

    logging.info(f"weight: {pos_weight}")

    if config.TRAINING.DATASET.WEIGHT_LOSS:
        print(losses_dict[config.TRAINING.LOSS])
        criterion = losses_dict[config.TRAINING.LOSS](
            pos_weight=torch.tensor(pos_weight).to(device)
        )
    else:
        criterion = losses_dict[config.TRAINING.LOSS]()

    optimizer = optimizer_dict[config.TRAINING.OPTIMIZER](
        params=model.parameters(), lr=config.TRAINING.LEARNING_RATE
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

        far, frr, hter = (
            dict_metrics[metric_]
            for metric_ in config.VALIDATION.METRICS_OF_INTEREST
        )

        if hter < best_hter:
            logging.info(
                f"Validation | model improved from {best_hter} to {hter} | saving model"
            )
            best_hter = hter
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "val_metrics": {"far": far, "frr": frr, "hter": hter},
            }
            torch.save(
                save_dict,
                Path(config.TRAINING.PATH_MODEL)
                / f"best_model_fold{fold_id}.pt",
            )

            best_metrics = dict_metrics

    logging.info(f"best metrics : {best_metrics}")

    return


if __name__ == "__main__":
    config_path = Path(
        "therapanacea_project/configs/training/training_cross_validation.yaml"
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

    cross_validation_training_from_config(
        config=config,
        device=device,
    )
