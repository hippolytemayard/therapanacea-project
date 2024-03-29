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
    step="Validation",
    writer=None,
    device="cpu",
):

    # Passer le modèle en mode eval
    model.eval()
    # initialiser les valeurs de loss et de prediction correctes
    val_loss, correct = 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        val_loss += criterion(output, target).data.item()

        if metrics_collection is not None:
            prediction = torch.sigmoid(output)
            batch_metrics = metrics_collection(prediction, target)

        correct += prediction.eq(target.data).cpu().sum()

    val_loss /= len(loader)

    accuracy = 100.0 * correct.to(torch.float32) / len(loader.dataset)

    if writer is not None:
        writer.add_scalar(f"Avg {step} Loss (epoch)", val_loss, epoch)
        writer.add_scalar(f"{step} Accuracy (epoch)", accuracy, epoch)

    print(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            step, val_loss, correct, len(loader.dataset), accuracy
        )
    )

    if metrics_collection is not None:
        metrics = metrics_collection.compute()
        print(metrics)

        if writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"{k}", v.detach().cpu().item(), epoch)
