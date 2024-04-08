import torch
from torchmetrics import Metric

from therapanacea_project.metrics.false_acceptance_rate import (
    FalseAcceptanceRate,
)
from therapanacea_project.metrics.false_rejection_rate import FalseRejectionRate


class HalfTotalErrorRate(Metric):
    """
    Custom Torchmetrics implementation of the Half Total Error Rate (HTER).

    HTER is a biometric error metric that combines the False Acceptance Rate
    and the False Rejection Rate (FRR) equally. It is calculated as the
        average of FAR and FRR.

    Args:
        threshold (float): Threshold value for binary classification
            (default: 0.5).

    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.far_metric = FalseAcceptanceRate(threshold)
        self.frr_metric = FalseRejectionRate(threshold)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric state based on predictions and ground truth labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits.
            labels (torch.Tensor): Ground truth labels.
        """
        self.far_metric.update(predictions, labels)
        self.frr_metric.update(predictions, labels)

    def compute(self) -> torch.Tensor:
        """
        Computes the Half Total Error Rate (HTER).

        Returns:
            torch.Tensor: The computed Half Total Error Rate.
        """
        far = self.far_metric.compute()
        frr = self.frr_metric.compute()
        hter = (far + frr) / 2
        return hter


if __name__ == "__main__":
    predictions = torch.tensor([0, 1, 1, 0, 0])
    ground_truth = torch.tensor([0, 1, 0, 1, 0])

    hter_metric = HalfTotalErrorRate()
    hter_metric(predictions, ground_truth)

    HTER = hter_metric.compute()
    print("Half Total Error Rate (HTER):", HTER)
