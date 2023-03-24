from vqgan.train_vqvae import VQVAE
from vqgan.train_vqgan import VQGAN
import streamlit as st
from pathlib import Path
from glob import glob
import torch
from vqgan.train_conditional_code_generator import ConditionalCodeGenerator
from PIL import Image
from einops import rearrange
import math
DEVICE = 'cuda'

st.title('Conditional image generation')

checkpoint_path = st.text_input(
    'Input checkpoint path', value="lightning_logs/conditional_transformer/version_1/checkpoints/epoch=62-step=98469.ckpt")


model = None
if checkpoint_path:

    if Path(checkpoint_path).is_dir():
        # find the checkpoint here
        files = glob(checkpoint_path + '/**')
        checkpoint_path = [f for f in files if f.endswith('.ckpt')][0]

    st.text(f"Using {checkpoint_path}")

    generator = ConditionalCodeGenerator.load_from_checkpoint(
        checkpoint_path).eval().to(DEVICE)

    condition = st.slider("Pick a condition", min_value=0,
                          max_value=generator.n_conditions - 1)

    bs = st.slider("Number to generate", min_value=1, max_value=8, value=1)

    # make the condition n_codes + condition + 1
    condition = torch.tensor(
        generator.codebook.n_codes + 1 + condition).long().view(1, 1).to(DEVICE)

    condition = condition.repeat(bs, 1)

    sampling = st.checkbox('Use sampling')

    # create the source and target for the transformer decoder
    start = torch.full(
        (bs, 1), generator.codebook.n_codes, device=DEVICE)

    source_codes = torch.cat((condition, start), dim=1)
    next_token = 0
    output_tokens = []
    for i in range(generator.n_positions):
        logits = generator.gpt2(source_codes).logits
        if sampling:
            next_token = torch.distributions.Categorical(
                logits=logits).sample()[:, -1:]
        else:
            next_token = logits.argmax(-1)[:, -1:]
        output_tokens.append(next_token)
        source_codes = torch.cat((source_codes, next_token), dim=1)

    output_tokens = torch.cat(output_tokens, dim=1)
    # reshape the output tokens
    h_w = int(math.sqrt(output_tokens.shape[-1]))

    output_tokens = output_tokens.reshape(-1, h_w, h_w)

    z_r = generator.codebook.decode(output_tokens)

    decoder = None
    if 'vqvae' in generator.checkpoint_path:
        decoder = VQVAE.load_from_checkpoint(
            generator.checkpoint_path).decoder.eval().to(DEVICE)

    with torch.no_grad():
        out = rearrange(decoder(z_r), 'b c h w -> b h w c')
        images = (((out.cpu().numpy() + 1) / 2) * 255).astype('uint8')

    images = [Image.fromarray(image).resize((128, 128))
              for image in images]

    cols = st.columns(4)

    for i, col in enumerate(cols):
        with col:
            col_images = [st.image(images[j]) for j in range(i, bs, 4)]
