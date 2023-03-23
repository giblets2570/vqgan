from torchvision.datasets import CIFAR100, Caltech256, ImageNet, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# this will convert everything to between -1 and 1
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def create_cifar100_dls(use_color_jitter=True, batch_size=32):
    transform_list = [
        transforms.RandomHorizontalFlip(p=0.5)
    ] + ([transforms.ColorJitter()] if use_color_jitter else []) + [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    dataset_transform = transforms.Compose(transform_list)
    train_dataset = CIFAR100('data/', download=True,
                             transform=dataset_transform, train=True)
    test_dataset = CIFAR100('data/', download=True,
                            transform=dataset_transform, train=False)

    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

    return train_dl, val_dl


if __name__ == '__main__':
    train_dataset = CIFAR10('data/', download=True)
    print(train_dataset)
