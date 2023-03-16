import torch
import torch.nn as nn

class CNNDiscriminator(nn.Module):

    def __init__(self):
        super().__init__(dropout_prob=0.5)
        # This will break the image into chunks of 4x4 to tell if it is real
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
        )

        self.bn_layers = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64)
        )

        self.pooling_layers = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )
        self.activation = nn.ReLU()
        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        
        for conv_layer, bn_layer, pool_layer in zip(self.conv_layers, self.bn_layers, self.pooling_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = self.activation(x)
            x = pool_layer(x)

        return x




if __name__ == '__main__':
    from torchvision.datasets import CIFAR100
    from torchvision import transforms

    dataset_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CIFAR100('data/', download=True, transform=dataset_transform)
    model = CNNEncoder()

    data = torch.stack((train_dataset[0][0], train_dataset[1][0]))


    output = model(data)
