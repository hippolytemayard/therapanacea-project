import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from therapanacea_project.train.validation_loop import validation_loop


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestValidationLoop(unittest.TestCase):
    def setUp(self):

        self.model = nn.Linear(10, 1)
        self.data_loader = DataLoader(
            MockDataset(100), batch_size=10, shuffle=True
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.metrics_collection = MetricCollection(
            metrics=Accuracy(task="binary")
        ).to(self.device)

    def test_validation_loop(self):
        # Call the validation loop function
        metrics = validation_loop(
            self.model,
            self.data_loader,
            epoch=0,
            criterion=self.criterion,
            metrics_collection=self.metrics_collection,
        )

        self.assertIsInstance(metrics, dict)

    def test_validation_loop_no_metrics(self):
        metrics = validation_loop(
            self.model,
            self.data_loader,
            epoch=0,
            criterion=self.criterion,
        )

        self.assertIsNone(metrics)

    def test_validation_loop_with_writer(self):
        writer = Mock()

        # Call the validation loop function with writer
        metrics = validation_loop(
            self.model,
            self.data_loader,
            epoch=0,
            criterion=self.criterion,
            metrics_collection=self.metrics_collection,
            writer=writer,
        )

        # Assert that metrics are computed
        self.assertIsInstance(metrics, dict)

        # Assert that writer methods are called
        writer.add_scalar.assert_called()

    def test_validation_loop_device(self):
        # Call the validation loop function with GPU device
        self.model.to(self.device)
        metrics = validation_loop(
            self.model,
            self.data_loader,
            epoch=0,
            criterion=self.criterion,
            metrics_collection=self.metrics_collection,
            device=self.device,
        )

        # Assert that metrics are computed
        self.assertIsInstance(metrics, dict)


if __name__ == "__main__":
    unittest.main()
