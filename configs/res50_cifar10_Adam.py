import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import evaluate
from utils import Saver

num_classes = 10
model = timm.create_model(model_name='resnet26',
                          pretrained=False,
                          num_classes=num_classes)
loss = nn.MSELoss()

img_size = (3, 224, 224)
batch_size=64
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# target_transforms is for the transform on label, we first transform it
# into LongTensor, then use one-hot encoding, at that time the tensor has shape (1, 10),
# finally we eliminate the first dim and it turns into shape (10,)
train_set = torchvision.datasets.CIFAR10(root='/data',
                                         train=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(size=img_size[-1]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))]),
                                         target_transform=torchvision.transforms.Compose([
                                             lambda x: torch.LongTensor([x]),
                                             lambda x: F.one_hot(x, num_classes),
                                             lambda x: x.squeeze(0),
                                             lambda x: x.to(torch.float32)]),
                                         )
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

test_set = torchvision.datasets.CIFAR10(root='/data',
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(size=img_size[-1]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))]),
                                         target_transform=transforms.Compose([
                                             lambda x: torch.LongTensor([x]),
                                             lambda x: F.one_hot(x, num_classes),
                                             lambda x: x.squeeze(0),
                                             lambda x: x.to(torch.float32)]),
                                         )
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3,weight_decay=1e-5)

epoch = 3


metric_tokens = ["accuracy"]
metric = evaluate.combine(metric_tokens)

saver = Saver(save_step=1,higher_is_better=True,monitor="accuracy")