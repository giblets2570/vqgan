from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.codebook import CodeBook
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


class VQVAE(pl.LightningModule):

    def __init__(self, feat_model='mobilenet_v2'):
        super().__init__()
        self.encoder = CNNEncoder()
        self.codebook = CodeBook()
        self.decoder = CNNDecoder()
        self.perceptual_loss = PerceptualLoss(model=feat_model)

    def training_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        sg1_loss = F.mse_loss(z_r, z.detach())
        sg2_loss = F.mse_loss(z, z_r.detach())

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        r_loss = F.mse_loss(r_image, image)
        perceptual_loss = self.perceptual_loss(r_image, image)

        loss = r_loss + perceptual_loss + sg1_loss + sg2_loss

        # Logging to TensorBoard (if installed) by default
        self.log("r_loss", r_loss, prog_bar=True)
        self.log("perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("sg_loss", sg1_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        if batch_idx == 0:
            sample_imgs = get_sample_imgs(r_image)

            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("reconstructed_images", grid, self.trainer.current_epoch)

        r_loss = F.mse_loss(r_image, image)
        perceptual_loss = self.perceptual_loss(r_image, image)

        sg1_loss = F.mse_loss(z_r, z.detach())

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("val_sg_loss", sg1_loss, prog_bar=True)

        self.log("val_loss", r_loss + perceptual_loss + 2 * sg1_loss)

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

    args = parser.parse_args()

    train_dl, val_dl = create_cifar100_dls()

    vqvae = VQVAE(feat_model=args.feat_model)

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model=vqvae, train_dataloaders=train_dl, val_dataloaders=val_dl)
