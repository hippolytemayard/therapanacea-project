from pathlib import Path

import pandas as pd

from therapanacea_project.evaluate.evaluate import (
    get_model_evaluation_from_checkpoint,
)


def cross_val_evaluation_from_checkpoints(
    experiment_model_folder: Path,
) -> pd.DataFrame:
    """
    Perform cross-validation evaluation by loading evaluation metrics from
    saved PyTorch model checkpoints and return them as a pandas DataFrame.

    Args:
        experiment_model_folder (Path): Path to the folder containing saved
        PyTorch model checkpoints.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each fold.
    """
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
    experiment_model_folder = Path(
        "/home/ubuntu/code/therapanacea-project/experiments/experiment_23/saved_models/"
    )
    saved_models = cross_val_evaluation_from_checkpoints(
        experiment_model_folder
    )
    print(saved_models)
