import torch
import timm
import torch.nn.functional as F
from dataset import CIFAR10
from modelings.losses import ClsCrossEntropy
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from evaluation import Accuracy
from accelerate.utils import ProjectConfiguration
from template.util import SaverConfiguration

# experiment tracker name
wandb_log_name = "example_project"

# dataset definition
# input shape
img_size = (3, 224, 224)

# class info
num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# dataset
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
                        lambda x: F.one_hot(x, num_classes).squeeze(0).to(torch.float32)]),
                    )

test_set = CIFAR10(root='/data/wangzili',
                   train=False,
                   transform=A.Compose([
                       A.Resize(img_size[-2], img_size[-1]),
                       A.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)),
                       ToTensorV2()]),
                   target_transform=lambda x: torch.tensor(x),
                   )

# model related configuration
model = timm.create_model(model_name='resnet50',
                          pretrained=False,
                          num_classes=num_classes)

loss_fn = ClsCrossEntropy()

lr = 5e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
# Note: in accelerate, the AcceleratedScheduler steps along with num_process
iter_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                   start_factor=0.01,
                                                   total_iters=500
                                                   )
epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                       milestones=[16, 20],
                                                       gamma=0.5)

# test configuration
metric = Accuracy(topk=(1, 5))

# saver
saver_config = SaverConfiguration(
    higher_is_better=True,
    monitor='accuracy@top1',
    save_dir='./runs'
)
project_config = ProjectConfiguration(
    automatic_checkpoint_naming=True,
    total_limit=3
)
