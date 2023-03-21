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

        optimizer_vae, optimizer_d = self.optimizers()

        # first train the vae

        self.toggle_optimizer(optimizer_vae)

        z = self.encoder(inp_image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        c1_loss = F.mse_loss(z_r, z.detach())
        c2_loss = F.mse_loss(z, z_r.detach())

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        # r_loss = F.mse_loss(r_image, image)
        p_loss = self.perceptual_loss(r_image, image).mean()

        # Logging to TensorBoard (if installed) by default
        self.log("p_loss", p_loss, prog_bar=True)
        self.log("c_loss", c1_loss, prog_bar=True)

        # we need to add in the discriminator loss
        disc_r_image = self.discriminator(r_image)
        valid = torch.ones_like(disc_r_image)

        g_loss = F.binary_cross_entropy_with_logits(
            disc_r_image, valid)
        self.log("g_loss", g_loss, prog_bar=True)

        lamb = 0
        delta = 1e-6
        numerators = torch.autograd.grad(
            p_loss, self.decoder.output_layer.parameters(), retain_graph=True)
        denominators = torch.autograd.grad(
            g_loss, self.decoder.output_layer.parameters(), retain_graph=True)
        for numerator, denominator in zip(numerators, denominators):
            lamb += (numerator.mean().abs() /
                     (denominator.mean().abs() + delta))
        lamb = lamb.clip(0, 1)

        self.log('lamb', lamb, prog_bar=True)

        vae_loss = p_loss + c1_loss + c2_loss + 0.01 * g_loss

        self.manual_backward(vae_loss)
        optimizer_vae.step()
        optimizer_vae.zero_grad()

        self.untoggle_optimizer(optimizer_vae)

        self.toggle_optimizer(optimizer_d)

        fake = torch.zeros_like(valid)

        real_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(image), valid)

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)
        r_image = self.decoder(z_r)

        fake_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(r_image), fake)

        d_loss = lamb * (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        r_loss = F.mse_loss(r_image, image)
        p_loss = self.perceptual_loss(r_image, image).mean()

        if batch_idx == 0:
            def denorm_imgs(imgs):
                return (imgs + 1) / 2
            grid = torchvision.utils.make_grid(
                torch.cat((denorm_imgs(image)[:6], denorm_imgs(r_image)[:6])), nrow=6)
            self.logger.experiment.add_image(
                "recontructed", grid, self.trainer.current_epoch)

        c1_loss = F.mse_loss(z_r, z.detach())
        disc_r_image = self.discriminator(r_image)
        valid = torch.ones_like(disc_r_image)
        g_loss = F.binary_cross_entropy_with_logits(
            disc_r_image, valid)

        fake = torch.zeros_like(valid)

        real_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(image), valid)
        fake_loss = F.binary_cross_entropy_with_logits(
            disc_r_image, fake)

        d_loss = (real_loss + fake_loss) / 2

        # Logging to TensorBoard (if installed) by default
        self.log("val_p_loss", p_loss, prog_bar=True)
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_c_loss", c1_loss, prog_bar=True)
        self.log("val_g_loss", g_loss, prog_bar=True)
        self.log("val_d_loss", d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_vae = optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.codebook.parameters(),
                self.decoder.parameters()
            ),
            lr=1e-3
        )
        optim.lr_scheduler.ReduceLROnPlateau(opt_vae)
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3
        )
        return [
            {
                "optimizer": opt_vae,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(opt_vae),
                    "monitor": "val_p_loss"
                }
            }, {
                "optimizer": opt_d,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(opt_d),
                    "monitor": "val_d_loss"
                }
            }
        ]


if __name__ == "__main__":
    from vqgan.cifar100_data import create_cifar100_dls

    train_dl, val_dl = create_cifar100_dls()

    vqgan = VQGAN()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model=vqgan, train_dataloaders=train_dl,
                val_dataloaders=val_dl)
