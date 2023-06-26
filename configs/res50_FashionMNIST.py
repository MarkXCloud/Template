import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import FashionMNIST
from modelings.losses import LossCls
from optim.scheduler import WarmUpMultiStep
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from evaluation import Accuracy
from template import Saver

# Task definition
log_name = "example_project"
epoch = 30

# dataset definition
# input shape
img_size = (3, 96, 96)
batch_size = 24
input_size = (batch_size,) + img_size

# class info
num_classes = 10
classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# dataset & dataloader
# target_transforms is for the transform on label, we first transform it
# into LongTensor, then use one-hot encoding, at that time the tensor has shape (1, 10),
# finally we eliminate the first dim and it turns into shape (10,)
train_set = FashionMNIST(root='/data/wangzili',
                         train=True,
                         transform=A.Compose([
                             A.Resize(img_size[-2], img_size[-1]),
                             ToTensorV2()]),
                         target_transform=T.Compose([
                             lambda x: torch.LongTensor([x]),
                             lambda x: F.one_hot(x, num_classes).squeeze(0).to(torch.float32)]),
                         )

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

test_set = FashionMNIST(root='/data/wangzili',
                        train=False,
                        transform=A.Compose([
                            A.Resize(img_size[-2], img_size[-1]),
                            ToTensorV2()]),
                        target_transform=lambda x: torch.tensor(x),
                        )

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# modeling related configuration
model = timm.create_model(model_name='resnet50',
                          in_chans=1,
                          pretrained=False,
                          num_classes=num_classes)
# modeling = torch.nn.SyncBatchNorm.convert_sync_batchnorm(modeling)

loss_fn = LossCls(loss_fn=nn.CrossEntropyLoss())

lr=1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
# Note: in accelerate, the AcceleratedScheduler steps along with num_process
num_iters_per_epoch = len(train_loader)
scheduler = WarmUpMultiStep(optimizer=optimizer,
                            start_factor=0.01,
                            warmup_iter=4*num_iters_per_epoch,
                            step_milestones=[16*num_iters_per_epoch,
                                             20*num_iters_per_epoch],
                            gamma=0.5)

# test configuration
metric = Accuracy()

# saver
saver = Saver(save_interval=5, higher_is_better=True, monitor="accuracy")
