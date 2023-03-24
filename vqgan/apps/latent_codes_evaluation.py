from vqgan.train_vqvae import VQVAE
from vqgan.train_vqgan import VQGAN
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
import seaborn
from pathlib import Path
from glob import glob
from vqgan.cifar100_data import create_dls
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


DEVICE = 'cuda'

st.title('Latent codes evaluation')

checkpoint_path = st.text_input(
    'Input checkpoint path', value="lightning_logs/vqvae/version_1/checkpoints")

model = None
if checkpoint_path:

    if Path(checkpoint_path).is_dir():
        # find the checkpoint here
        files = glob(checkpoint_path + '/*')
        checkpoint_path = [f for f in files if f.endswith('.ckpt')][0]

    st.text(f"Using {checkpoint_path}")

    try:
        model = VQVAE.load_from_checkpoint(checkpoint_path).eval().to(DEVICE)
    except:
        model = VQGAN.load_from_checkpoint(checkpoint_path).eval().to(DEVICE)

    codes = model.codebook.embedding.weight.detach().cpu().numpy()

    with st.spinner(text="Computing TSNE..."):
        codes_embedded = TSNE(n_components=2, learning_rate='auto',
                              init='random', perplexity=3).fit_transform(codes)

    ax = seaborn.scatterplot(x=codes_embedded[:, 0], y=codes_embedded[:, 1])

    st.pyplot(ax.figure)

    plt.clf()

    _, val_dl = create_dls()

    n_occurances = np.zeros((codes.shape[0], ))

    progress_text = "Finding the most frequently used codes"
    my_bar = st.progress(0.0, text=progress_text)
    with torch.no_grad():
        for batch_idx, (image, _) in enumerate(val_dl):
            z = model.encoder(image.to(DEVICE))
            z_q = model.codebook(z, sample=False)
            z_q = z_q.view(-1)
            for n in z_q:
                n_occurances[n] += 1

            my_bar.progress((batch_idx + 1) / len(val_dl), text=progress_text)

    st.text(f'{(n_occurances == 0).sum()} codes do not occur in the validation set')
    norm_n_occurances = n_occurances / n_occurances.sum()
    df = pd.DataFrame()
    df['n_occurances'] = norm_n_occurances
    ax = df[['n_occurances']].plot.bar()
    st.pyplot(ax.figure)
    plt.clf()

    ax = df.sort_values('n_occurances', ascending=False).cumsum()[
        ['n_occurances']].plot.bar()
    st.pyplot(ax.figure)
    plt.clf()

    ax = df[['n_occurances']].plot.hist()
    st.pyplot(ax.figure)
    plt.clf()

    ax = seaborn.scatterplot(
        x=codes_embedded[:, 0], y=codes_embedded[:, 1], hue=norm_n_occurances)

    st.pyplot(ax.figure)
    plt.clf()

    # Try some clustering
    n_clusters_ = 0
    n_noise_ = 0
    labels = None
    db = None
    eps = 0.0

    with st.spinner("Finding clusters..."):
        while n_clusters_ < np.sqrt(model.codebook.n_codes):
            eps += 0.5
            db = DBSCAN(eps=eps).fit(codes)
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

    st.text(f'eps={eps}')
    st.text("Estimated number of clusters: %d" % n_clusters_)
    st.text("Estimated number of noise points: %d" % n_noise_)
