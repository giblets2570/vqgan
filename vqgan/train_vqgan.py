from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.discriminator import CNNDiscriminator
from vqgan.codebook import CodeBook
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch
import itertools
from vqgan.perceptual_loss import PerceptualLoss


class VQGAN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = CNNEncoder()
        self.codebook = CodeBook()
        self.decoder = CNNDecoder()
        self.discriminator = CNNDiscriminator()
        self.perceptual_loss = PerceptualLoss()

    def training_step(self, batch, batch_idx):
        image, _ = batch

        optimizer_vae, optimizer_d = self.optimizers()

        # first train the vae

        self.toggle_optimizer(optimizer_vae)

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        sg1_loss = F.mse_loss(z_r, z.detach())
        sg2_loss = F.mse_loss(z, z_r.detach())

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        # r_loss = F.mse_loss(r_image, image)
        r_loss = self.perceptual_loss(r_image, image)

        # Logging to TensorBoard (if installed) by default
        self.log("r_loss", r_loss, prog_bar=True)
        self.log("sg_loss", sg1_loss, prog_bar=True)

        # we need to add in the discriminator loss
        valid = torch.ones(image.size(0), 4, 4).type_as(image)

        g_loss = F.binary_cross_entropy_with_logits(self.discriminator(r_image), valid)
        self.log("g_loss", g_loss, prog_bar=True)

        vae_loss = r_loss + sg1_loss + sg2_loss + g_loss

        self.manual_backward(vae_loss)
        optimizer_vae.step()
        optimizer_vae.zero_grad()

        self.untoggle_optimizer(optimizer_vae)

        self.toggle_optimizer(optimizer_d)

        fake = torch.zeros_like(valid)

        real_loss = F.binary_cross_entropy_with_logits(self.discriminator(image), valid)

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)
        r_image = self.decoder(z_r)

        fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(r_image), fake)

        d_loss = (real_loss + fake_loss) / 2
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

        sg1_loss = F.mse_loss(z_r, z.detach())
        valid = torch.ones(image.size(0), 4, 4).type_as(image)
        g_loss = F.binary_cross_entropy_with_logits(self.discriminator(r_image), valid)

        fake = torch.zeros_like(valid)

        real_loss = F.binary_cross_entropy_with_logits(self.discriminator(image), valid)
        fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(r_image), fake)

        d_loss = (real_loss + fake_loss) / 2

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_sg_loss", sg1_loss, prog_bar=True)
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
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3
        )
        return [opt_vae, opt_d], []




if __name__ == "__main__":
    from vqgan.cifar100_data import create_cifar100_dls

    train_dl, val_dl = create_cifar100_dls()

    vqgan = VQGAN()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model=vqgan, train_dataloaders=train_dl, val_dataloaders=val_dl)