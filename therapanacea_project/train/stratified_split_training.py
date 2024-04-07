import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from therapanacea_project.dataset.dataloader import get_train_val_dataloaders
from therapanacea_project.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from therapanacea_project.losses import losses_dict
from therapanacea_project.metrics.utils import instantiate_metrics_from_config
from therapanacea_project.models.resnet import get_resnet_architecture
from therapanacea_project.optimizer import optimizer_dict
from therapanacea_project.train.utils.training_loop import training_loop
from therapanacea_project.train.utils.validation_loop import validation_loop
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import read_txt_object


def stratified_split_train_model_from_config(
    config: OmegaConf,
    device: torch.device,
):
    """
    Train a model based on the provided configuration.

    Args:
        config (OmegaConf): Experiment configuration.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
    """

    writer = (
        SummaryWriter(config.TRAINING.TENSORBOARD_DIR)
        if config.TRAINING.ENABLE_TENSORBOARD
        else None
    )

    model = get_resnet_architecture(
        architecture=config.TRAINING.BACKBONE,
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

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.13]).to(device))
    optimizer = optimizer_dict[config.TRAINING.OPTIMIZER](
        params=model.parameters(), lr=config.TRAINING.LEARNING_RATE
    )

    metrics_collection = instantiate_metrics_from_config(
        metrics_config=config.VALIDATION.METRICS
    ).to(device)

    best_hter = float("inf")

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
                save_dict, Path(config.TRAINING.PATH_MODEL) / "best_model.pt"
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="therapanacea_project/configs/training/stratified_split/training_resnet18.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

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

    stratified_split_train_model_from_config(config=config, device=device)
