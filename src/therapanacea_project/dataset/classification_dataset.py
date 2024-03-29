from typing import List, Union, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


class ClassificationDataset(Dataset):

    def __init__(
        self,
        images_list: List[str],
        images_labels: List[int],
        transforms: Union[Compose, None] = None,
    ) -> None:
        self.images_list = images_list
        self.images_labels = images_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image = read_image(str(self.images_list[idx]))
        # image = image.to(torch.float)
        image = image.to(torch.float) / 255.0

        label = self.images_labels[idx]

        assert isinstance(label, int)
        assert isinstance(image, torch.Tensor)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
