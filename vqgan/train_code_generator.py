import pytorch_lightning as pl
import torch.optim as optim
from vqgan.code_generator import CodeGenerator
from vqgan.train_vqvae import VQVAE
import itertools
import torch
from einops import rearrange
import torch.nn.functional as F
from vqgan.metrics import PerPositionAccuracy


class TrainCodeGenerator(pl.LightningModule):

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        vqvae = VQVAE.load_from_checkpoint(checkpoint_path)
        self.encoder = vqvae.encoder
        self.codebook = vqvae.codebook
        self.code_generator = CodeGenerator(
            n_codes=self.codebook.n_codes + 1, embedding_dim=128)
        # we add an extra token that will act as start/stop
        for param in itertools.chain(
                self.codebook.parameters(), self.encoder.parameters()):
            param.requires_grad = False

        n_positions = 32 // self.encoder.m

        self.train_pp_acc = PerPositionAccuracy(
            n_positions=n_positions, n_codes=self.codebook.n_codes)
        self.val_pp_acc = PerPositionAccuracy(
            n_positions=n_positions, n_codes=self.codebook.n_codes)

    def training_step(self, batch, batch_idx):
        image, _ = batch
        bs = image.shape[0]

        with torch.no_grad():
            z = self.encoder(image)
            codes = self.codebook(z, sample=False)
            # codes will be in the shape  h x w
            codes = rearrange(codes, 'b x y -> b (x y)')

            # create the source and target for the transformer decoder
            start = torch.full(
                (bs, 1), self.codebook.n_codes, device=codes.device)

            source_codes = torch.cat((start, codes), dim=1)[:, :-1]

        # now we pass the codes through the code generator
        outputs = self.code_generator(source_codes)
        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('loss', loss, prog_bar=True)
        self.train_pp_acc(outputs.argmax(-1), codes)
        return loss

    def on_train_epoch_end(self):
        self.logger.experiment.add_histogram(
            'train_pp_acc', self.train_pp_acc.compute())
        self.train_pp_acc.reset()

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        bs = image.shape[0]

        z = self.encoder(image)
        codes = self.codebook(z, sample=False)
        # codes will be in the shape  h x w
        codes = rearrange(codes, 'b x y -> b (x y)')

        # create the source and target for the transformer decoder
        start = torch.full(
            (bs, 1), self.codebook.n_codes, device=codes.device)

        source_codes = torch.cat((start, codes), dim=1)[:, :-1]

        # now we pass the codes through the code generator
        outputs = self.code_generator(source_codes)
        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('val_loss', loss, prog_bar=True)
        self.val_pp_acc(outputs.argmax(-1), codes)

    def on_validation_epoch_end(self):
        self.logger.experiment.add_histogram(
            'val_pp_acc', self.val_pp_acc.compute())
        self.val_pp_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss"
        }


if __name__ == '__main__':
    from vqgan.cifar100_data import create_cifar100_dls

    module = TrainCodeGenerator(
        './lightning_logs/version_2/checkpoints/epoch=39-step=62520.ckpt')
    train_dl, val_dl = create_cifar100_dls(batch_size=32)

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(
        model=module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
