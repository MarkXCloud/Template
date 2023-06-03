import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data.dataset import CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
import evaluate
from utils import Saver
from utils.Paradigm import ImageClassificationParadigm



# Task definition
paradigm = ImageClassificationParadigm()
log_name = "example_project"
epoch = 1

# dataset definition
# input shape
img_size = (3, 224, 224)
batch_size = 24
input_size = (batch_size,) + img_size

# class info
num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# dataset & dataloader
# target_transforms is for the transform on label, we first transform it
# into LongTensor, then use one-hot encoding, at that time the tensor has shape (1, 10),
# finally we eliminate the first dim and it turns into shape (10,)
train_set = CIFAR10(root='/data/wangzili',
                    train=True,
                    transform=A.Compose([
                        A.Resize(img_size[-2], img_size[-1]),
                        A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)),
                        ToTensorV2()]),
                    target_transform=T.Compose([
                        lambda x: torch.LongTensor([x]),
                        lambda x: F.one_hot(x, num_classes),
                        lambda x: x.squeeze(0),
                        lambda x: x.to(torch.float32)]),
                    )

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)

test_set = CIFAR10(root='/data/wangzili',
                   train=False,
                   transform=A.Compose([
                       A.Resize(img_size[-2], img_size[-1]),
                       A.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)),
                       ToTensorV2()]),
                   target_transform=lambda x: torch.tensor(x),
                   )

test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

# model and other configuration
model = timm.create_model(model_name='resnet50',
                          pretrained=False,
                          num_classes=num_classes)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=2),
                torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)]
)

# test configuration
metric_tokens = ["accuracy"]
metric = evaluate.combine(metric_tokens)

# saver
saver = Saver(save_interval=1, higher_is_better=True, monitor="accuracy")