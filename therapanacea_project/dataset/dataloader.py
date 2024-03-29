from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from therapanacea_project.dataset.classification_dataset import (
    ClassificationDataset,
)

# from torch.utils.data.distributed import DistributedSampler


def get_train_val_dataloaders(
    images_list: list,
    labels: list,
    train_transforms,
    val_transforms,
    val_size: float = 0.25,
) -> Tuple[DataLoader, DataLoader]:

    train_images, val_images, train_labels, val_labels = train_test_split(
        images_list,
        labels,
        test_size=val_size,
        random_state=42,
        stratify=labels,
    )

    train_loader = get_single_dataloader(
        images_list=train_images,
        labels=train_labels,
        transform=train_transforms,
        batch_size=4,
    )
    val_loader = get_single_dataloader(
        images_list=val_images,
        labels=val_labels,
        transform=val_transforms,
        batch_size=4,
    )

    return train_loader, val_loader


def get_single_dataloader(
    images_list: list,
    labels: list,
    transform: transforms,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:

    dataset = ClassificationDataset(
        images_list=images_list, images_labels=labels, transforms=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader
