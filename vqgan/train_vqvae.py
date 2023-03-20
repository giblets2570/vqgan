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
import random
from copy import deepcopy


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
        latent_dim=128,
        spacing=8,
        m=3,
        beta=0.02,
        dropout_prob=0.5,
        use_noise=False,
        use_codebook_sampling=False
    ):
        super().__init__()
        self.encoder = CNNEncoder(
            out_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        self.codebook = CodeBook(
            latent_dim=latent_dim, n_codes=n_codes, use_sampling=use_codebook_sampling)
        self.decoder = CNNDecoder(
            in_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        if feat_model is None:
            self.perceptual_loss = None
        else:
            self.perceptual_loss = PerceptualLoss(model=feat_model)
        self.beta = beta
        self.use_noise = use_noise

    def training_step(self, batch, batch_idx):
        image, _ = batch

        if self.use_noise:
            # choose 30% of the input images
            bs = image.shape[0]
            n = (bs * 3) // 10
            mask = torch.full((bs, ), False).to(image.device)
            noise_idx = random.choices(range(bs), k=n)
            for idx in noise_idx:
                mask[idx] = True
            inp_image = deepcopy(image)
            noise = torch.randn_like(
                inp_image[mask]) * torch.sqrt(torch.tensor(0.1, device=image.device))
            inp_image[mask] = inp_image[mask] + noise
        else:
            inp_image = image

        z = self.encoder(inp_image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        sg_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("sg_loss", sg_loss / 2, prog_bar=True)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        r_loss = F.mse_loss(r_image, image)
        self.log("r_loss", r_loss, prog_bar=True)

        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(r_image, image)
            self.log("perceptual_loss", perceptual_loss, prog_bar=True)
        else:
            perceptual_loss = 0

        loss = r_loss + perceptual_loss + self.beta * sg_loss

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
            grid = torchvision.utils.make_grid(
                torch.cat((sample_imgs, sample_rimgs)), nrow=6)
            self.logger.experiment.add_image(
                "real_images", grid, self.trainer.current_epoch)

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
    parser.add_argument('--feat-model', default='none', type=str)
    parser.add_argument('--latent-dim', default=128, type=int)
    parser.add_argument('--n-codes', default=256, type=int)
    parser.add_argument('--m', default=3, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--dropout-prob', default=0.5, type=float)
    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--use-noise', action='store_true')
    parser.add_argument('--use-codebook-sampling', action='store_true')

    args = parser.parse_args()

    train_dl, val_dl = create_cifar100_dls(batch_size=args.batch_size)

    if args.feat_model == 'none':
        args.feat_model = None

    vqvae = VQVAE(
        feat_model=args.feat_model,
        latent_dim=args.latent_dim,
        dropout_prob=args.dropout_prob,
        n_codes=args.n_codes,
        m=args.m,
        beta=args.beta,
        use_noise=args.use_noise,
        use_codebook_sampling=args.use_codebook_sampling
    )

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(
        model=vqvae,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
