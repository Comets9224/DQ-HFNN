import torchvision
import torchvision.transforms as transforms
import torch

DOWNLOAD_CIFAR10 = True

def load_dataset_cifar10(train_transform=None, test_transform=None):
    if train_transform is None:
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2470, 0.2435, 0.2616]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    if test_transform is None:
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2470, 0.2435, 0.2616]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    import os
    # 使用绝对路径，指向项目根目录的data/cifar10
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'cifar10')

    train_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=DOWNLOAD_CIFAR10,
        transform=train_transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=DOWNLOAD_CIFAR10,
        transform=test_transform
    )

    return train_data, test_data