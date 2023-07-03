import torchvision
import numpy as np
from typing import Tuple, Any

__all__=['FashionMNIST']
class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        FashionMNIST fitted to albumentations.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            image = self.transform(image=image.numpy().astype(np.float32))['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
