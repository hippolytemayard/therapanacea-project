import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    criterion,
    metrics_collection: MetricCollection = None,
    writer=None,
    device="cpu",
):

    model.eval()

    val_loss = 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device, torch.float)

        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)

        val_loss += criterion(prediction, target).data.item()

        if metrics_collection is not None:
            metrics_collection(prediction, target.long())

    val_loss /= len(loader)

    logging.info(f"Validation | loss = {val_loss}")

    if writer is not None:
        writer.add_scalar("validation loss", val_loss, epoch)

    if metrics_collection is not None:
        metrics = metrics_collection.compute()

        for k, v in metrics.items():
            if k == "BinaryROC":
                fpr, tpr, _ = v
                fnr = 1 - tpr
                logging.info(f"Validation | fpr = {fpr.detach().cpu().item()}")
                logging.info(f"Validation | fnr = {fnr.detach().cpu().item()}")

                if writer is not None:
                    writer.add_scalar(
                        "Validation fpr", fpr.detach().cpu().item(), epoch
                    )
                    writer.add_scalar(
                        "Validation fnr", fnr.detach().cpu().item(), epoch
                    )

            else:
                logging.info(f"Validation | {k} = {v.detach().cpu().item()}")

                if writer is not None:
                    writer.add_scalar(
                        f"Validation {k}", v.detach().cpu().item(), epoch
                    )

        metrics_collection.reset()

        return metrics
