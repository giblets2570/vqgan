from torchvision.datasets import CIFAR100, Caltech256, ImageNet, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
import torch
from PIL import Image
import webdataset as wds
import numpy as np
from glob import glob
import random

# this will convert everything to between -1 and 1
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class MSCOCODataset(IterableDataset):

    def __init__(self, ids, transforms, shuffle=True):
        self.ids = ids
        self.transforms = transforms
        self.shuffle = shuffle

    def __iter__(self):
        for _id in self.ids:
            ds = wds.WebDataset(f'./data/mscoco/{_id}.tar')
            if self.shuffle:
                ds = ds.shuffle(1000)

            def preprocess(sample):
                image, text = sample
                image = Image.fromarray(
                    (image * 255).astype(np.uint8)).resize((128, 128))
                image = self.transforms(image)
                return image, text
            ds = iter(ds.decode('rgb').to_tuple("jpg", "txt").map(preprocess))

            for sample in ds:

                yield sample


def create_dls(use_color_jitter=True, batch_size=32, dataset='cifar100'):
    transform_list = [
        transforms.RandomHorizontalFlip(p=0.5)
    ] + ([transforms.ColorJitter()] if use_color_jitter else []) + [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    dataset_transform = transforms.Compose(transform_list)

    if dataset == 'cifar100':
        train_dataset = CIFAR100('data/', download=True,
                                 transform=dataset_transform, train=True)
        test_dataset = CIFAR100('data/', download=True,
                                transform=dataset_transform, train=False)
        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        val_dl = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=3)

    elif dataset == 'cifar10':
        train_dataset = CIFAR10('data/', download=True,
                                transform=dataset_transform, train=True)
        test_dataset = CIFAR10('data/', download=True,
                               transform=dataset_transform, train=False)
        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False)

    elif dataset == 'mscoco':
        stats = glob('./data/mscoco/*_stats.json')
        ids = [stat.split('/')[-1].split('_')[0] for stat in stats]
        random.seed(42)
        random.shuffle(ids)
        train_ids = ids[:int(len(ids) * 0.8)]
        test_ids = ids[int(len(ids) * 0.8):]

        train_dataset = MSCOCODataset(train_ids, dataset_transform)
        test_dataset = MSCOCODataset(test_ids, dataset_transform)

        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=3)
        val_dl = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=3)

    return train_dl, val_dl


if __name__ == '__main__':
    dls = create_dls(dataset='mscoco')
    image = next(iter(dls[0]))[0]
    print(dls)
