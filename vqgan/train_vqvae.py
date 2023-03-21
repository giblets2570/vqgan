from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.codebook import CodeBook
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch
import torchvision
import random
from copy import deepcopy
from lpips import LPIPS


class VQVAE(pl.LightningModule):

    def __init__(
        self,
        feat_model='vgg',
        n_codes=256,
        latent_dim=128,
        m=3,
        beta=0.02,
        dropout_prob=0.5,
        use_noise=False,
        use_codebook_sampling=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = CNNEncoder(
            out_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        self.codebook = CodeBook(
            latent_dim=latent_dim, n_codes=n_codes, use_sampling=use_codebook_sampling)
        self.decoder = CNNDecoder(
            in_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        if feat_model is None:
            self.perceptual_loss = None
        else:
            self.perceptual_loss = LPIPS(net=feat_model, lpips=False)
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

        c_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("c_loss", c_loss / 2, prog_bar=True)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        if self.perceptual_loss is not None:
            r_loss = 0
            perceptual_loss = self.perceptual_loss(r_image, image).mean()
            self.log("perceptual_loss", perceptual_loss, prog_bar=True)
        else:
            r_loss = F.mse_loss(r_image, image)
            self.log("r_loss", r_loss, prog_bar=True)
            perceptual_loss = 0

        loss = r_loss + perceptual_loss + self.beta * c_loss

        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        if batch_idx == 0:
            def denorm_imgs(imgs):
                return (imgs + 1) / 2
            grid = torchvision.utils.make_grid(
                torch.cat((denorm_imgs(image)[:6], denorm_imgs(r_image)[:6])), nrow=6)
            self.logger.experiment.add_image(
                "recontructed", grid, self.trainer.current_epoch)

        r_loss = F.mse_loss(r_image, image)
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(r_image, image).mean()
            self.log("val_perceptual_loss", perceptual_loss, prog_bar=True)
        else:
            perceptual_loss = 0  # for the below logger of val_loss
        c1_loss = F.mse_loss(z_r, z.detach())

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_c_loss", c1_loss, prog_bar=True)

        self.log("val_loss", r_loss + perceptual_loss + 2 * c1_loss)

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
