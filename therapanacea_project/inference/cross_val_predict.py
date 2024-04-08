import argparse
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from omegaconf import OmegaConf

from therapanacea_project.dataset.dataloader import get_single_dataloader
from therapanacea_project.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from therapanacea_project.inference.predict import predict
from therapanacea_project.models import get_model_architecture
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import write_predictions_to_file


def cross_val_predict_from_config(
    config: OmegaConf,
    device: Union[str, torch.device] = "cpu",
) -> list[int]:
    """
    Predict labels for images specified in the configuration file.

    Args:
        config (OmegaConf): Configuration containing inference parameters.
        device (Union[str, torch.device], optional): Device. Defaults to "cpu".

    Returns:
        List[int]: List of predicted labels.
    """

    images_list = Path(config.INFERENCE.DATASET.IMAGES_DIR).glob("*.jpg")
    images_list = sorted(list(images_list))

    inference_transforms = instantiate_transforms_from_config(
        transform_config=config.INFERENCE.DATASET.TRANSFORMS.INFERENCE
    )

    inference_loader = get_single_dataloader(
        images_list=images_list,
        labels=None,
        transform=inference_transforms,
        batch_size=config.INFERENCE.BATCH_SIZE,
        inference=True,
    )

    saved_models_folder = Path(config.INFERENCE.PATH_MODEL)
    saved_models = list(saved_models_folder.glob("*.pt"))

    model = get_model_architecture(
        architecture=config.INFERENCE.BACKBONE,
        n_classes=config.INFERENCE.DATASET.N_CLASSES,
        fine_tune=config.INFERENCE.FINE_TUNE,
        pretrained=config.INFERENCE.PRETRAINED,
    ).to(device)

    k_fold_predictions = []

    for saved_model in saved_models:

        model_state_dict = torch.load(str(saved_model))
        model.load_state_dict(model_state_dict["model"])

        predictions = predict(
            model=model,
            loader=inference_loader,
            threshold=config.INFERENCE.PREDICTION_THRESHOLD,
            device=device,
        )

        k_fold_predictions.append(predictions)

    return k_fold_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train model using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="therapanacea_project/configs/inference/cross_validation/inference_resnet18.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    predictions_dir = Path(config.INFERENCE.PATH_PREDICTIONS)
    make_exists(predictions_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_fold_predictions = cross_val_predict_from_config(
        config=config, device=device
    )

    df_k_fold_predictions = pd.DataFrame(k_fold_predictions)
    df_k_fold_predictions.to_csv(predictions_dir / "val_preds_kfold.csv")

    # Vote over the folds
    predictions = df_k_fold_predictions.mode(axis=0, dropna=True).iloc[0]

    write_predictions_to_file(
        predictions=predictions.to_list(),
        file_path=predictions_dir / "val_preds_vote.txt",
    )
