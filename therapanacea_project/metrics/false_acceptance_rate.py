import torch
from torchmetrics import Metric


class FalseAcceptanceRate(Metric):
    """
    Custom torchmetrics implementation of the False Acceptance Rate (FAR)
    see https://lightning.ai/docs/torchmetrics/stable/pages/implement.html

    FAR measures the proportion of instances where an impostor is incorrectly
    accepted.

    Args:
        threshold (float): Threshold value for binary classification
            (default: 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.add_state(
            "total_negatives", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_acceptances", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.threshold = threshold

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric state based on predictions and ground truth labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits.
            labels (torch.Tensor): Ground truth labels.
        """
        predictions = (predictions > self.threshold).int()
        false_acceptances = torch.sum(predictions & (labels == 0)).float()
        self.false_acceptances += false_acceptances.long()
        self.total_negatives += torch.sum(labels == 0).long()

    def compute(self) -> torch.Tensor:
        """
        Computes the False Acceptance Rate (FAR).

        Returns:
            torch.Tensor: The computed False Acceptance Rate.
        """
        far = self.false_acceptances / self.total_negatives
        return far


if __name__ == "__main__":
    predictions = torch.tensor([0, 1, 1, 0, 0])
    ground_truth = torch.tensor([0, 1, 0, 1, 0])

    far_metric = FalseAcceptanceRate()

    far_metric(predictions, ground_truth)

    FAR = far_metric.compute()
    print("False Acceptance Rate (FAR):", FAR)

    far_metric.reset()
