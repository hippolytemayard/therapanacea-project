import logging
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    criterion: nn.modules.loss._Loss,
    with_logits: bool = False,
    metrics_collection: Optional[MetricCollection] = None,
    writer=None,
    device: Union[str, torch.device] = "cpu",
):
    """
    Model validation loop

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader providing the validation data.
        epoch (int): The current epoch number.
        criterion (nn.modules.loss._Loss): The loss function
        with_logits (bool, optional): Indicates whether the model output
            includes logits. Defaults to False.
        metrics_collection (Optional[MetricCollection], optional): Collection
            of metrics to compute during validation. Defaults to None.
        writer (optional): Object for writing validation progress to a log.
            Defaults to None.
        device (Union[str, torch.device], optional): Device on which to
            perform validation. Defaults to "cpu".

    Returns:
        Optional[dict]: A dictionary containing computed metrics,
            if metrics_collection is provided; otherwise, returns None.
    """
    model.eval()

    val_loss = 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device, torch.float)

        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)

        val_loss += (
            criterion(prediction, target).data.item()
            if not with_logits
            else criterion(output, target.unsqueeze(-1)).data.item()
        )

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
