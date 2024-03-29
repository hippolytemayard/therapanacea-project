import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


def training_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epoch: int,
    optimizer: Optimizer,
    metrics_collection: MetricCollection = None,
    log_interval: int = 200,
    writer=None,
    device="cpu",
):
    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device, torch.float)

        optimizer.zero_grad()

        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)

        loss = criterion(prediction, target)

        if writer is not None:
            writer.add_scalar(
                "Training Loss (batch)",
                loss,
                epoch * len(loader) + batch_idx,
            )

        if metrics_collection is not None:
            metrics_collection(prediction, target)

        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.data.item(),
                )
            )

    if metrics_collection is not None:
        metrics = metrics_collection.compute()
        print(metrics)

        if writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"{k}", v.detach().cpu().item(), epoch)
