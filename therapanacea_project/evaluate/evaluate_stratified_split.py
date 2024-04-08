import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from therapanacea_project.utils.files import load_yaml


def get_model_evaluation_from_checkpoint(
    config: OmegaConf,
    cross_val: bool = False,
) -> dict[str, float]:
    """
    Load evaluation metrics from training config and return
    them as a dictionary.

    Args:
        config (Path): Training configuration file.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    if not cross_val:
        experiment_model_folder = Path(config.TRAINING.PATH_MODEL)
        saved_model_path = experiment_model_folder / "best_model.pt"

    else:
        saved_model_path = config

    state_dict = torch.load(str(saved_model_path))
    val_metrics = state_dict["val_metrics"]
    val_metrics = {k: v.detach().cpu().item() for k, v in val_metrics.items()}

    return val_metrics


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
    saved_models = get_model_evaluation_from_checkpoint(config=config)
    print(saved_models)
