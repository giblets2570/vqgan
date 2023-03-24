import pytorch_lightning as pl
import torch
from einops import rearrange
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from vqgan.train_code_generator import CodeGenerator


class ConditionalCodeGenerator(CodeGenerator):

    def __init__(self, checkpoint_path: str, n_conditions: int):
        self.n_conditions = n_conditions
        super().__init__(checkpoint_path)

    def create_gpt2_model(self):
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=self.codebook.n_codes + 1 + self.n_conditions,
            n_positions=self.n_positions + 1,  # +1 for the condition
            bos_token_id=self.codebook.n_codes,
            eos_token_id=self.codebook.n_codes
        ))

    def training_step(self, batch, batch_idx):
        image, condition = batch
        bs = image.shape[0]

        # make the condition n_codes + condition + 1
        condition = self.codebook.n_codes + 1 + condition

        with torch.no_grad():
            z = self.encoder(image)
            codes = self.codebook(z)
            # codes will be in the shape  h x w
            codes = rearrange(codes, 'b x y -> b (x y)')

            # create the source and target for the transformer decoder
            start = torch.full(
                (bs, 1), self.codebook.n_codes, device=codes.device)

            source_codes = torch.cat((start, codes), dim=1)[:, :-1]

            # add the condition to the beginning
            source_codes = torch.cat(
                (condition.view(-1, 1), source_codes), dim=1)

        # now we pass the codes through the code generator
        outputs = self.gpt2(source_codes).logits
        # remove one element for the condition
        outputs = outputs[:, 1:]

        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('loss', loss, prog_bar=True)
        self.train_pp_acc(outputs.argmax(-1), codes)
        return loss

    def validation_step(self, batch, batch_idx):
        image, condition = batch
        bs = image.shape[0]

        # make the condition n_codes + condition + 1
        condition = self.codebook.n_codes + 1 + condition

        z = self.encoder(image)
        codes = self.codebook(z)
        # codes will be in the shape  h x w
        codes = rearrange(codes, 'b x y -> b (x y)')

        # create the source and target for the transformer decoder
        start = torch.full(
            (bs, 1), self.codebook.n_codes, device=codes.device)

        source_codes = torch.cat((start, codes), dim=1)[:, :-1]
        source_codes = torch.cat(
            (condition.view(-1, 1), source_codes), dim=1)

        # now we pass the codes through the code generator
        outputs = self.gpt2(source_codes).logits
        outputs = outputs[:, 1:]

        loss = F.cross_entropy(outputs.transpose(2, 1), codes)
        self.log('val_loss', loss, prog_bar=True)
        self.val_pp_acc(outputs.argmax(-1), codes)


if __name__ == '__main__':
    from vqgan.cifar100_data import create_dls
    from pytorch_lightning.loggers import TensorBoardLogger

    model = ConditionalCodeGenerator(
        './lightning_logs/vqvae/version_1/checkpoints/epoch=98-step=154737.ckpt', n_conditions=100)
    train_dl, val_dl = create_dls(batch_size=32)

    trainer = pl.Trainer(
        max_epochs=300,
        logger=TensorBoardLogger(
            save_dir='lightning_logs/',
            name='conditional_transformer',
        )
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
