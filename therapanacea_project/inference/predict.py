from pathlib import Path
from typing import Union

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from therapanacea_project.dataset.dataloader import get_single_dataloader
from therapanacea_project.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from therapanacea_project.models.resnet18 import get_resnet18_architecture
from therapanacea_project.utils.files import load_yaml, make_exists
from therapanacea_project.utils.io import write_predictions_to_file


def predict_from_config(
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

    model = get_resnet18_architecture(
        n_classes=config.INFERENCE.DATASET.N_CLASSES,
        fine_tune=False,
        pretrained=False,
    ).to(device)

    path_model = Path(config.INFERENCE.PATH_MODEL) / "best_model.pt"
    model_state_dict = torch.load(path_model)
    model.load_state_dict(model_state_dict["model"])

    predictions = predict(
        model=model,
        loader=inference_loader,
        threshold=config.INFERENCE.PREDICTION_THRESHOLD,
        device=device,
    )
    return predictions


def predict(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
    device: Union[str, torch.device] = "cpu",
) -> list[int]:
    """
    Generate predictions for a given model and data loader.

    Args:
        model (nn.Module): model.
        loader (DataLoader): Inference data loader.
        threshold (float, optional): Threshold for binary classification.
            Defaults to 0.5.
        device (Union[str, torch.device], optional): Device. Defaults to "cpu".

    Returns:
        List[int]: List of predicted labels.
    """
    model.eval()

    predictions = []

    for data in tqdm(loader):
        data = data.to(device)
        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)
        prediction = (prediction > threshold).int()

        predictions.extend(prediction.tolist())

    return predictions


if __name__ == "__main__":

    config_path = Path(
        "therapanacea_project/configs/inference/inference_config.yaml"
    )
    config = load_yaml(config_path)
    predictions_dir = Path(config.INFERENCE.PATH_PREDICTIONS)
    make_exists(predictions_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = predict_from_config(config=config, device=device)

    write_predictions_to_file(
        predictions=predictions, file_path=predictions_dir / "val_preds.txt"
    )
