from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.perceptual_loss import PerceptualLoss
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch
import torchvision
from vqgan.cifar100_data import MEAN, STD


def get_sample_imgs(r_image):

    trans = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )

    return trans(r_image[:6])


class AE(pl.LightningModule):

    def __init__(self, feat_model='mobilenet_v2'):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()
        self.perceptual_loss = PerceptualLoss(model=feat_model)

    def training_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        r_image = self.decoder(z)

        r_loss = F.mse_loss(r_image, image)
        perceptual_loss = self.perceptual_loss(r_image, image)

        loss = r_loss + perceptual_loss

        # Logging to TensorBoard (if installed) by default
        self.log("r_loss", r_loss, prog_bar=True)
        self.log("perceptual_loss", perceptual_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        r_image = self.decoder(z)

        if batch_idx == 0:
            sample_imgs = get_sample_imgs(r_image)

            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("reconstructed_images", grid, self.trainer.current_epoch)

        r_loss = F.mse_loss(r_image, image)
        perceptual_loss = self.perceptual_loss(r_image, image)

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("val_loss", r_loss + perceptual_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss"
        }


if __name__ == "__main__":
    from argparse import ArgumentParser
    from vqgan.cifar100_data import create_cifar100_dls

    parser = ArgumentParser()
    parser.add_argument('--feat-model', default='mobilenet_v2', type=str)
    parser.add_argument('--no-color-jitter', action='store_true', default=False)

    args = parser.parse_args()

    use_color_jitter = not args.no_color_jitter
    train_dl, val_dl = create_cifar100_dls(use_color_jitter)

    ae = AE(feat_model=args.feat_model)

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model=ae, train_dataloaders=train_dl, val_dataloaders=val_dl)
