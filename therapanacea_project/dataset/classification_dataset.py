from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


class ClassificationDataset(Dataset):
    """
    Custom dataset class for classification tasks.
    """

    def __init__(
        self,
        images_list: list[str],
        images_labels: Union[list[int], None],
        transforms: Union[Compose, None] = None,
        inference: bool = False,
    ) -> None:
        """
        Constructor for the ClassificationDataset class.

        Args:
            images_list (list[str]): List of file paths to images.
            images_labels (Union[list[int],None]): List of corresponding
                labels for images. if None inference mode.
            transforms (Union[Compose, None], optional): Optional
                transformations to apply to images. Defaults to None.
            inference (bool): Inference mode dataset.
        """
        self.images_list = images_list
        self.images_labels = images_labels
        self.transforms = transforms
        self.inference = inference

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, idx) -> Union[tuple[torch.Tensor, int], torch.Tensor]:
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Tuple containing the image tensor and
                its corresponding label or only tensor if inference mode.
        """
        image = read_image(str(self.images_list[idx]))
        image = image.to(torch.float) / 255.0

        if self.transforms is not None:
            image = self.transforms(image)

        if self.inference and self.images_labels is None:
            return image

        label = self.images_labels[idx]

        assert isinstance(label, int)
        assert isinstance(image, torch.Tensor)

        return image, label
