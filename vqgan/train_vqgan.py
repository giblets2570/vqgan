from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.discriminator import CNNDiscriminator
from vqgan.codebook import CodeBook
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch
import itertools
import torchvision
from lpips import LPIPS
import random
from copy import deepcopy


class VQGAN(pl.LightningModule):

    def __init__(
        self,
        feat_model='vgg',
        n_codes=256,
        latent_dim=128,
        m=2,
        beta=0.02,
        dropout_prob=0.1,
        use_noise=False,
        use_codebook_sampling=True
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.encoder = CNNEncoder(
            out_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        self.codebook = CodeBook(
            latent_dim=latent_dim, n_codes=n_codes, use_sampling=use_codebook_sampling)
        self.decoder = CNNDecoder(
            in_channels=latent_dim, m=m, dropout_prob=dropout_prob)
        self.discriminator = CNNDiscriminator(m=m, dropout_prob=dropout_prob)
        self.perceptual_loss = LPIPS(net=feat_model, lpips=False)
        self.beta = beta
        self.use_noise = use_noise

    def __compute_lamb(self, p_loss, gan_loss, current_epoch):
        lamb = 0
        delta = 1e-6
        numerators = torch.autograd.grad(
            p_loss, self.decoder.output_layer.parameters(), retain_graph=True)
        denominators = torch.autograd.grad(
            gan_loss, self.decoder.output_layer.parameters(), retain_graph=True)

        for numerator, denominator in zip(numerators, denominators):
            lamb += (numerator.mean() /
                     (denominator.mean() + delta))
        lamb = lamb.clip(-1, 1)
        return lamb

    def __add_noise(self, image):
        bs = image.shape[0]
        n = (bs * 3) // 10
        mask = torch.full((bs, ), False).to(image.device)
        noise_idx = random.choices(range(bs), k=n)
        for idx in noise_idx:
            mask[idx] = True
        inp_image = deepcopy(image)
        noise = torch.randn_like(
            inp_image[mask]
        ) * torch.sqrt(torch.tensor(0.1, device=image.device))
        inp_image[mask] = inp_image[mask] + noise
        return inp_image

    def training_step(self, batch, batch_idx):
        image, _ = batch
        inp_image = self.__add_noise(image) if self.use_noise else image

        z = self.encoder(inp_image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        c_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("c_loss", c_loss / 2, prog_bar=True)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        p_loss = self.perceptual_loss(r_image, image).mean()
        self.log("p_loss", p_loss, prog_bar=True)
        r_loss = F.mse_loss(r_image, image)
        self.log("r_loss", r_loss, prog_bar=True)

        # gan loss
        disc_image = self.discriminator(image)
        valid = torch.ones_like(disc_image)
        real_loss = F.binary_cross_entropy_with_logits(
            disc_image, valid)

        fake = torch.zeros_like(valid)
        fake_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(r_image), fake)

        g_loss = (real_loss + fake_loss) / 2
        self.log("g_loss", g_loss, prog_bar=True)

        lamb = self.__compute_lamb(p_loss, g_loss, self.trainer.current_epoch)
        self.log("lamb", lamb, prog_bar=True)
        loss = p_loss + self.beta * c_loss + lamb * g_loss
        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        c_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("val_c_loss", c_loss / 2, prog_bar=True)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        p_loss = self.perceptual_loss(r_image, image).mean()
        self.log("val_p_loss", p_loss, prog_bar=True)
        r_loss = F.mse_loss(r_image, image)
        self.log("val_r_loss", r_loss, prog_bar=True)

        # gan loss
        disc_image = self.discriminator(image)
        valid = torch.ones_like(disc_image)
        real_loss = F.binary_cross_entropy_with_logits(
            disc_image, valid)

        fake = torch.zeros_like(valid)
        fake_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(r_image), fake)

        g_loss = (real_loss + fake_loss) / 2
        self.log("val_g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss"
        }


if __name__ == "__main__":
    from vqgan.cifar100_data import create_cifar100_dls

    train_dl, val_dl = create_cifar100_dls()

    vqgan = VQGAN()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model=vqgan, train_dataloaders=train_dl,
                val_dataloaders=val_dl)
