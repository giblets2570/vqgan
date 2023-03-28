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


class MovingAverage:

    def __init__(self, n_values):
        self.n_values = n_values
        self.values = []

    def add(self, value):
        self.values.append(value)
        self.values = self.values[-self.n_values:]
        return self

    def get(self):
        return sum(self.values) / len(self.values)


class VQGAN2(pl.LightningModule):

    def __init__(
        self,
        feat_model='vgg',
        n_codes=256,
        latent_dim=128,
        m=2,
        beta=0.02,
        dropout_prob=0.1,
        use_noise=False,
        use_codebook_sampling=True,
        n_warmup_epochs=5,
        n_critic=5
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
        self.lambda_ma = MovingAverage(5)

    def __compute_lamb(self, p_loss, gan_loss, current_epoch):
        lamb = 0
        if current_epoch >= self.hparams.n_warmup_epochs:
            delta = 1e-6
            numerators = torch.autograd.grad(
                p_loss, self.decoder.output_layer.parameters(), retain_graph=True)
            denominators = torch.autograd.grad(
                gan_loss, self.decoder.output_layer.parameters(), retain_graph=True)

            for numerator, denominator in zip(numerators, denominators):
                lamb += (torch.linalg.vector_norm(numerator) /
                         (torch.linalg.vector_norm(denominator) + delta))
            lamb = lamb.clip(0, 500)
        return self.lambda_ma.add(lamb).get()

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

    def __plot_images(self, image, r_image):
        def denorm_imgs(imgs):
            return (imgs + 1) / 2
        grid = torchvision.utils.make_grid(
            torch.cat((denorm_imgs(image)[:6], denorm_imgs(r_image)[:6])), nrow=6)
        self.logger.experiment.add_image(
            "recontructed", grid, self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):
        image, _ = batch

        optimizer_d, optimizer_t, optimizer_vae = self.optimizers()

        # first train the discriminator
        self.toggle_optimizer(optimizer_d)
        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)
        self.decoder.requires_grad_()  # so I can compute lamb
        r_image = self.decoder(z_r)
        p_loss = self.perceptual_loss(r_image, image).mean()

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

        optimizer_d.zero_grad()
        self.manual_backward(lamb * g_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=0.2)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        if self.hparams.n_critic == 0 or (batch_idx + 1) % self.hparams.n_critic == 0:

            self.toggle_optimizer(optimizer_t)

            z = self.encoder(image)
            z_q = self.codebook(z)
            z_r = self.codebook.decode(z_q)
            r_image = self.decoder(z_r)

            # we need to add in the discriminator loss
            # I dont want the gradients to reach the encoder
            t_loss = F.binary_cross_entropy_with_logits(
                self.discriminator(r_image), valid)
            self.log("t_loss", t_loss, prog_bar=True)

            optimizer_t.zero_grad()
            self.manual_backward(lamb * t_loss)
            self.clip_gradients(optimizer_t, gradient_clip_val=0.2)
            optimizer_t.step()

            self.untoggle_optimizer(optimizer_t)

        self.toggle_optimizer(optimizer_vae)

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)

        c_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("c_loss", c_loss / 2, prog_bar=True)

        z_r = z + (z_r - z).detach()  # trick to pass gradients
        r_image = self.decoder(z_r)

        r_loss = F.mse_loss(r_image, image)
        p_loss = self.perceptual_loss(r_image, image).mean()

        # Logging to TensorBoard (if installed) by default
        self.log("r_loss", r_loss, prog_bar=True)
        self.log("p_loss", p_loss, prog_bar=True)

        vae_loss = p_loss + self.hparams.beta * c_loss
        optimizer_vae.zero_grad()
        self.manual_backward(vae_loss)
        self.clip_gradients(optimizer_vae, gradient_clip_val=0.2)
        optimizer_vae.step()
        self.untoggle_optimizer(optimizer_vae)

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        z = self.encoder(image)
        z_q = self.codebook(z)
        z_r = self.codebook.decode(z_q)
        r_image = self.decoder(z_r)

        if batch_idx == 0:
            self.__plot_images(image, r_image)

        disc_image = self.discriminator(image)
        valid = torch.ones_like(disc_image)
        real_loss = F.binary_cross_entropy_with_logits(
            disc_image, valid)

        fake = torch.zeros_like(valid)
        fake_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(r_image), fake)

        g_loss = (real_loss + fake_loss) / 2

        self.log("val_g_loss", g_loss, prog_bar=True)
        p_loss = self.perceptual_loss(r_image, image).mean()  # to find lambda
        self.log("val_p_loss", p_loss, prog_bar=True)

        c_loss = F.mse_loss(z_r, z.detach()) + F.mse_loss(z, z_r.detach())
        self.log("val_c_loss", c_loss / 2, prog_bar=True)

        r_loss = F.mse_loss(r_image, image)
        # Logging to TensorBoard (if installed) by default
        self.log("val_r_loss", r_loss, prog_bar=True)

        # we need to add in the discriminator loss
        t_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(r_image), valid)
        self.log("val_t_loss", t_loss, prog_bar=True)

        self.log("val_loss", p_loss + c_loss + g_loss + t_loss)

    def configure_optimizers(self):
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3
        )
        opt_t = optim.Adam(
            self.decoder.parameters(),
            lr=1e-3
        )
        opt_vae = optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.codebook.parameters(),
                self.decoder.parameters()
            ),
            lr=1e-3
        )
        return [opt_d, opt_t, opt_vae], [
            optim.lr_scheduler.ReduceLROnPlateau(opt_d, patience=3),
            optim.lr_scheduler.ReduceLROnPlateau(opt_t, patience=3),
            optim.lr_scheduler.ReduceLROnPlateau(opt_vae, patience=3),
        ]


if __name__ == "__main__":
    from argparse import ArgumentParser
    from vqgan.dataset import create_dls
    from pytorch_lightning.loggers import TensorBoardLogger

    parser = ArgumentParser()
    parser.add_argument('--feat-model', default='vgg', type=str)
    parser.add_argument('--latent-dim', default=128, type=int)
    parser.add_argument('--n-codes', default=256, type=int)
    parser.add_argument('--m', default=3, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--dropout-prob', default=0.5, type=float)
    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--use-noise', action='store_true')
    parser.add_argument('--use-codebook-sampling', action='store_true')
    parser.add_argument('--dataset', default='cifar100')

    args = parser.parse_args()

    train_dl, val_dl = create_dls(batch_size=args.batch_size)

    if args.feat_model == 'none':
        args.feat_model = None

    # vqgan2 = VQGAN2(
    #     feat_model=args.feat_model,
    #     latent_dim=args.latent_dim,
    #     dropout_prob=args.dropout_prob,
    #     n_codes=args.n_codes,
    #     m=args.m,
    #     beta=args.beta,
    #     use_noise=args.use_noise,
    #     use_codebook_sampling=args.use_codebook_sampling
    # )
    vqgan2 = VQGAN2.load_from_checkpoint(
        'lightning_logs/vqgan2/version_13/checkpoints/epoch=84-step=49980.ckpt')

    vqgan2.hparams.n_critic = 0
    vqgan2.hparams.n_warmup_epochs = 20

    trainer = pl.Trainer(
        max_epochs=300,
        logger=TensorBoardLogger(
            save_dir='lightning_logs/',
            name='vqgan2',
            sub_dir=f'nc={args.n_codes},ld={args.latent_dim},m={args.m},b={args.beta},d={args.dropout_prob}'
        ),
    )
    trainer.fit(
        model=vqgan2,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
