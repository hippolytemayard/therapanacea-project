from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from therapanacea_project.dataset.classification_dataset import (
    ClassificationDataset,
)


def get_train_val_dataloaders(
    images_list: list,
    labels: list,
    batch_size: int,
    train_transforms,
    val_transforms,
    val_size: float = 0.25,
) -> tuple[DataLoader, DataLoader, dict[int, float]]:
    """
    Function to get training and validation data loaders from a stratified
    train/val split.

    Args:
        images_list (list): List of image file paths.
        labels (list): List of corresponding labels for images.
        batch_size (int): Batch size for DataLoader.
        train_transforms: Transformations to apply to training data.
        val_transforms: Transformations to apply to validation data.
        val_size (float, optional): Percentage of data to use for validation.
            Defaults to 0.25.

    Returns:
        tuple[DataLoader, DataLoader, dict[int, float]]: Tuple containing
            train DataLoader, validation DataLoader, and dictionary
            representing the distribution of labels in the training set.
    """

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
        batch_size=batch_size,
    )
    val_loader = get_single_dataloader(
        images_list=val_images,
        labels=val_labels,
        transform=val_transforms,
        batch_size=batch_size,
    )

    train_labels_distribution = {
        0: len(train_labels) - sum(train_labels),
        1: sum(train_labels),
    }

    return train_loader, val_loader, train_labels_distribution


def get_single_dataloader(
    images_list: list,
    labels: list,
    transform: transforms,
    batch_size: int,
    num_workers: int = 4,
    inference: bool = False,
) -> DataLoader:
    """
    Function to get a single DataLoader from image path and labels files.

    Args:
        images_list (list): List of image file paths.
        labels (list): List of corresponding labels for images.
        transform: Transformations to apply to data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. Defaults to 4.
        inference (bool): Inference or train/val mode.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = ClassificationDataset(
        images_list=images_list,
        images_labels=labels,
        transforms=transform,
        inference=inference,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader
