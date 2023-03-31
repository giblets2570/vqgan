import pytorch_lightning as pl
import torch.optim as optim
from vqgan.train_vqvae import VQVAE
from vqgan.train_vqgan import VQGAN
import itertools
import torch
from einops import rearrange
import torch.nn.functional as F
from vqgan.metrics import PerPositionAccuracy
import pandas as pd
from transformers import GPT2Config, GPT2LMHeadModel


class CodeGenerator(pl.LightningModule):

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.save_hyperparameters()
        self.checkpoint_path = checkpoint_path
        vqvae = VQGAN.load_from_checkpoint(checkpoint_path)
        self.encoder = vqvae.encoder
        self.codebook = vqvae.codebook
        self.n_positions = (32 // 2 ** self.encoder.m) ** 2

        self.gpt2 = self.create_gpt2_model()

        for param in itertools.chain(
                self.codebook.parameters(), self.encoder.parameters()):
            param.requires_grad = False

        self.train_pp_acc = PerPositionAccuracy(
            n_positions=self.n_positions, n_codes=self.codebook.n_codes)
        self.val_pp_acc = PerPositionAccuracy(
            n_positions=self.n_positions, n_codes=self.codebook.n_codes)

    def create_gpt2_model(self):
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=self.codebook.n_codes + 1,
            n_positions=self.n_positions,
            bos_token_id=self.codebook.n_codes,
            eos_token_id=self.codebook.n_codes
        ))

    def training_step(self, batch, batch_idx):
        image, _ = batch
        bs = image.shape[0]

        with torch.no_grad():
            z = self.encoder(image)
            codes = self.codebook(z)
            # codes will be in the shape  h x w
            codes = rearrange(codes, 'b x y -> b (x y)')

            # create the source and target for the transformer decoder
            start = torch.full(
                (bs, 1), self.codebook.n_codes, device=codes.device)

            source_codes = torch.cat((start, codes), dim=1)[:, :-1]

        # now we pass the codes through the code generator
        outputs = self.gpt2(source_codes).logits
        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('loss', loss, prog_bar=True)
        self.train_pp_acc(outputs.argmax(-1), codes)
        return loss

    def on_train_epoch_end(self):
        train_pp_acc = self.train_pp_acc.compute()
        ax = pd.DataFrame(train_pp_acc.cpu().numpy()).plot()
        self.logger.experiment.add_figure(
            'train_pp_acc', ax.figure, global_step=self.trainer.current_epoch)
        self.train_pp_acc.reset()

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        bs = image.shape[0]

        z = self.encoder(image)
        codes = self.codebook(z)
        # codes will be in the shape  h x w
        codes = rearrange(codes, 'b x y -> b (x y)')

        # create the source and target for the transformer decoder
        start = torch.full(
            (bs, 1), self.codebook.n_codes, device=codes.device)

        source_codes = torch.cat((start, codes), dim=1)[:, :-1]

        # now we pass the codes through the code generator
        outputs = self.gpt2(source_codes).logits
        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('val_loss', loss, prog_bar=True)
        self.val_pp_acc(outputs.argmax(-1), codes)

    def on_validation_epoch_end(self):
        val_pp_acc = self.val_pp_acc.compute()
        ax = pd.DataFrame(val_pp_acc.cpu().numpy()).plot()
        self.logger.experiment.add_figure(
            'val_pp_acc', ax.figure, global_step=self.trainer.current_epoch)
        self.val_pp_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss"
        }


if __name__ == '__main__':
    from vqgan.cifar100_data import create_dls
    from pytorch_lightning.loggers import TensorBoardLogger

    module = CodeGenerator(
        './lightning_logs/vqvae/version_1/checkpoints/epoch=98-step=154737.ckpt')
    train_dl, val_dl = create_dls(batch_size=32)

    trainer = pl.Trainer(
        max_epochs=300,
        logger=TensorBoardLogger(
            save_dir='lightning_logs/',
            name='transformer',
        )
    )
    trainer.fit(
        model=module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
