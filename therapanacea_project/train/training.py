import argparse
import logging
from typing import Union

import torch
from omegaconf import OmegaConf

from therapanacea_project.train.cross_validation_training import (
    cross_validation_training_from_config,
)
from therapanacea_project.train.stratified_split_training import (
    stratified_split_train_model_from_config,
)
from therapanacea_project.utils.files import load_yaml, make_exists


def train_model_from_config(
    config: OmegaConf,
    device: Union[str, torch.device],
) -> None:
    """
    Perform training based on the provided configuration.
    Either a random split or a cross validation training.

    Args:
        config (OmegaConf): Configuration object containing training
            parameters.
        device (Union[str, torch.device]): Device to use for training
            (e.g., 'cpu' or 'cuda').

    Returns:
        None
    """

    if config.TRAINING.CROSS_VALIDATION:
        cross_validation_training_from_config(config=config, device=device)

    else:
        stratified_split_train_model_from_config(config=config, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train model using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="therapanacea_project/configs/training/cross_validation/training_resnet18.yaml",
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

    train_model_from_config(
        config=config,
        device=device,
    )
