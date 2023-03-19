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


def get_sample_imgs(image, r_image):

    trans = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )

    return trans(image[:6]), trans(r_image[:6])


class VQVAE(pl.LightningModule):

    def __init__(
        self,
        feat_model='mobilenet_v2',
        n_codes=256,
        embedding_dim=128,
        spacing=8,
        n_pools=3
    ):
        super().__init__()
        self.encoder = CNNEncoder(out_channels=embedding_dim, spacing=spacing, n_pools=n_pools)
        self.codebook = CodeBook(embedding_dim=embedding_dim, n_codes=n_codes)
        self.decoder = CNNDecoder(in_channels=embedding_dim, spacing=spacing, n_transposes=n_pools)
        if feat_model is None:
            self.perceptual_loss = None
        else:
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

        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(r_image, image)
            self.log("perceptual_loss", perceptual_loss, prog_bar=True)
        else:
            perceptual_loss = 0

        loss = r_loss + perceptual_loss + sg1_loss + sg2_loss

        # Logging to TensorBoard (if installed) by default
        self.log("r_loss", r_loss, prog_bar=True)
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
            sample_imgs, sample_rimgs = get_sample_imgs(image, r_image)
            grid = torchvision.utils.make_grid(torch.cat((sample_imgs, sample_rimgs)), nrow=6)
            self.logger.experiment.add_image("real_images", grid, self.trainer.current_epoch)

        r_loss = F.mse_loss(r_image, image)
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(r_image, image)
            self.log("val_perceptual_loss", perceptual_loss, prog_bar=True)
        else:
            perceptual_loss = 0  # for the below logger of val_loss
        sg1_loss = F.mse_loss(z_r, z.detach())

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
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
    parser.add_argument('--embedding-dim', default=128, type=int)
    parser.add_argument('--spacing', default=8, type=int)
    parser.add_argument('--n-codes', default=256, type=int)
    parser.add_argument('--n-pools', default=3, type=int)
    parser.add_argument('--batch-size', default=32, type=int)

    args = parser.parse_args()

    train_dl, val_dl = create_cifar100_dls(batch_size=args.batch_size)

    if args.feat_model == 'none':
        args.feat_model = None

    vqvae = VQVAE(
        feat_model=args.feat_model,
        embedding_dim=args.embedding_dim,
        spacing=args.spacing
    )

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(
        model=vqvae,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
