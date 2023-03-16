from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def create_cifar100_dls():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]

    dataset_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = CIFAR100('data/', download=True, transform=dataset_transform, train=True)
    test_dataset = CIFAR100('data/', download=True, transform=dataset_transform, train=False)

    train_dl = DataLoader(train_dataset, batch_size=32)
    val_dl = DataLoader(test_dataset, batch_size=32)

    return train_dl, val_dl
