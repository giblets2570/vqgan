from vqgan.encoder import CNNEncoder
from vqgan.decoder import CNNDecoder
from vqgan.codebook import CodeBook
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch


class VQVAE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.codebook = CodeBook()
        self.decoder = CNNDecoder()

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

        loss = r_loss + sg1_loss + sg2_loss

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

        r_loss = F.mse_loss(r_image, image)

        sg1_loss = F.mse_loss(z_r, z.detach())

        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)
        self.log("val_sg_loss", sg1_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    from vqgan.cifar100_data import create_cifar100_dls

    train_dl, val_dl = create_cifar100_dls()

    vqvae = VQVAE()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model=vqvae, train_dataloaders=train_dl, val_dataloaders=val_dl)
