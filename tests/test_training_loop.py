import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from therapanacea_project.train.utils.training_loop import training_loop


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.data_loader = DataLoader(
            MockDataset(100), batch_size=10, shuffle=True
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.metrics_collection = MetricCollection(
            metrics=Accuracy(task="binary")
        ).to(self.device)

    def test_training_loop(self):
        # Call the training loop function
        training_loop(
            model=self.model,
            loader=self.data_loader,
            criterion=self.criterion,
            epoch=0,
            optimizer=self.optimizer,
            metrics_collection=self.metrics_collection,
        )

        # Assert that model parameters have changed after optimization
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)
            self.assertIsNotNone(param.grad)

    def test_training_loop_with_metrics(self):
        # Create a mock writer
        writer = Mock()

        # Call the training loop function with metrics and writer
        training_loop(
            model=self.model,
            loader=self.data_loader,
            criterion=self.criterion,
            epoch=0,
            optimizer=self.optimizer,
            metrics_collection=self.metrics_collection,
            writer=writer,
        )

        # Assert that writer methods are called
        writer.add_scalar.assert_called()

    # TODO
    """
    def test_training_loop_device(self):
        # Call the training loop function with GPU device

        self.model.to(self.device)
        training_loop(
            model=self.model,
            loader=self.data_loader,
            criterion=self.criterion,
            epoch=0,
            optimizer=self.optimizer,
            metrics_collection=self.metrics_collection,
            device=self.device,
        )

        # Assert that model is moved to the correct device
        self.assertEqual(next(self.model.parameters()).device, self.device)
    """


if __name__ == "__main__":
    unittest.main()
