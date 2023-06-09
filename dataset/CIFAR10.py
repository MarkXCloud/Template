import torch
import torchvision
from typing import Tuple,Any,List

__all__=['CIFAR10']

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def __getitem__(self,  index: int) -> Tuple[Any, Any]:
        """
        CIFAR10 fitted to albumentations.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    @staticmethod
    def collate_fn(batch:List[Tuple]):
        batch = list(zip(*batch))
        batch_image = torch.stack(batch[0])
        batch_target = torch.stack(batch[1])
        return batch_image,batch_target