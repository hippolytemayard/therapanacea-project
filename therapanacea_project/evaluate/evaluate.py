from pathlib import Path

import torch


def get_model_evaluation_from_checkpoint(
    saved_model_path: Path,
) -> dict[str, float]:
    """
    Load evaluation metrics from a saved PyTorch model checkpoint and return
    them as a dictionary.

    Args:
        saved_model_path (Path): Path to the saved PyTorch model checkpoint.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    state_dict = torch.load(str(saved_model_path))
    val_metrics = state_dict["val_metrics"]
    val_metrics = {k: v.detach().cpu().item() for k, v in val_metrics.items()}

    return val_metrics


if __name__ == "__main__":
    saved_model_path = Path(
        "/home/ubuntu/code/therapanacea-project/experiments/experiment_2/saved_models/best_model.pt"
    )
    saved_models = get_model_evaluation_from_checkpoint(
        saved_model_path=saved_model_path
    )
    print(saved_models)
