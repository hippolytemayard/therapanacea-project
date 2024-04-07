import argparse

import pandas as pd
from omegaconf import OmegaConf

from therapanacea_project.evaluate.evaluate_stratified_split import (
    get_model_evaluation_from_checkpoint,
)
from therapanacea_project.utils.files import load_yaml


def cross_val_evaluation_from_config(
    config: OmegaConf,
) -> pd.DataFrame:
    """
    Perform cross-validation evaluation by loading evaluation metrics from
    saved PyTorch model checkpoints and return them as a pandas DataFrame.

    Args:
        config (Path): Training configuration file.
        PyTorch model checkpoints.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each fold.
    """
    experiment_model_folder = config.TRAINING.PATH_MODEL
    saved_models = list(experiment_model_folder.glob("*.pt"))

    list_df_metrics = []
    for saved_model in saved_models:

        val_metrics = get_model_evaluation_from_checkpoint(
            saved_model_path=saved_model
        )
        val_metrics["fold"] = saved_model.name

        list_df_metrics.append(pd.DataFrame(data=val_metrics, index=[0]))
    df_metrics = pd.concat(list_df_metrics).set_index("fold")

    return df_metrics


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

    saved_models = cross_val_evaluation_from_config(config=config)
    print(saved_models)
