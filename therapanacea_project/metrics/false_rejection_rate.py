import torch
from torchmetrics import Metric


class FalseRejectionRate(Metric):
    """
    Custom torchmetrics implementation of the False Rejection Rate (FRR)
    see https://lightning.ai/docs/torchmetrics/stable/pages/implement.html

    FRR measures the proportion of instances where a genuine is incorrectly
    rejected by the classifier. It is calculated as the ratio of false
    rejections to the total number of identification attempts for genuine
    instances.

    Args:
        threshold (float): Threshold value for binary classification
            (default: 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.add_state(
            "false_rejections", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_attempts", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.threshold = threshold

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric state based on predictions and ground truth labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits.
            labels (torch.Tensor): Ground truth labels.
        """

        false_rejections = torch.sum(
            (predictions < self.threshold) & (labels == 1)
        ).float()
        self.false_rejections += false_rejections
        self.total_attempts += labels.size(0)

    def compute(self) -> torch.Tensor:
        """
        Computes the False Acceptance Rate (FAR).

        Returns:
            torch.Tensor: The computed False Acceptance Rate.
        """
        frr = self.false_rejections / self.total_attempts
        return frr


if __name__ == "__main__":
    predictions = torch.tensor([0, 1, 1, 0, 0])
    ground_truth = torch.tensor([0, 1, 0, 1, 0])

    frr_metric = FalseRejectionRate()
    frr_metric(predictions, ground_truth)

    FRR = frr_metric.compute()
    print("False Rejection Rate (FAR):", FRR)

    frr_metric.reset()
