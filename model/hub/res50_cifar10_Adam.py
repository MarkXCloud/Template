import torch
import torch.nn.functional as F
from torchvision import transforms

model = dict(
    model_name='resnet50',
    pretrained=False,
    num_classes=10
)

loss = dict(
    name='MSE'
)

train_set = dict(
    name='cifar10',
    params=dict(
        root='YOUR/PATH/TO/CIFAR10',
        batch_size=32,
        num_workers=4,
        transforms=transforms.Compose([transforms.Resize(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # target_transforms is for the transform on label, we first transform it
        # into LongTensor, then use one-hot encoding, at that time the tensor has shape (1, 10),
        # finally we eliminate the first dim and it turns into shape (10,)
        target_transforms=transforms.Compose([lambda x: torch.LongTensor([x]),
                                              lambda x: F.one_hot(x, 10),
                                              lambda x: x.squeeze(0),
                                              lambda x: x.to(torch.float32)]),
        is_Train=True,
        is_distributed=False
    ),

    num_classes=10,
    cls=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
)

test_set = dict(
    name='cifar10',
    params=dict(
        root='/data/wangzili',
        batch_size=32,
        num_workers=4,
        transforms=transforms.Compose([transforms.Resize(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # target_transforms is for the transform on label, we first transform it
        # into LongTensor, then use one-hot encoding, at that time the tensor has shape (1, 10),
        # finally we eliminate the first dim and it turns into shape (10,)
        target_transforms=transforms.Compose([lambda x: torch.LongTensor([x]),
                                              lambda x: F.one_hot(x, 10),
                                              lambda x: x.squeeze(0),
                                              lambda x: x.to(torch.float32)]),
        is_Train=False,
        is_distributed=False
    ),
    num_classes=10,
    cls=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
)

optimizer = dict(
    name='Adam',
    params=dict(
        lr=1e-3,
        weight_decay=0.0
    )
)

trainer = dict(
    epoch=100,
    seed=0
)
