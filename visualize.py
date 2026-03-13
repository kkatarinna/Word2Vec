import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_DIR   = "data"
OUT_DIR    = "plots"
TOP_N      = 150       # words to plot
TSNE_PERPLEXITY = 30
SEED       = 42

def load():
    W_in = np.load(os.path.join(DATA_DIR, "W_in.npy"))

    with open(os.path.join(DATA_DIR, "idx2word.json")) as f:
        raw      = json.load(f)
        idx2word = {int(k): v for k, v in raw.items()}

    tokens = np.load(os.path.join(DATA_DIR, "tokens.npy"))

    print(f"[load]  W_in {W_in.shape}")
    return W_in, tokens, idx2word


def select_top(W_in, tokens, idx2word, top_n):
    counts      = np.bincount(tokens, minlength=W_in.shape[0])
    counts[0]   = 0                             # exclude <UNK>
    top_idx     = np.argsort(counts)[::-1][:top_n]
    words       = [idx2word[i] for i in top_idx]
    vecs        = W_in[top_idx]
    freqs       = counts[top_idx]
    print(f"[select] {top_n} words  |  most frequent: {words[:5]}")
    return vecs, words, freqs


def reduce_pca(vecs):
    reduced = PCA(n_components=2, random_state=SEED).fit_transform(vecs)
    print(f"[pca]   {vecs.shape[1]}D → 2D")
    return reduced

def reduce_tsne(vecs):
    # First reduce to 50D with PCA for t-SNE stability, then t-SNE to 2D
    n_components = min(50, vecs.shape[1], vecs.shape[0] - 1)
    vecs50 = PCA(n_components=n_components, random_state=SEED).fit_transform(vecs)
    reduced = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=1000,
        random_state=SEED,
        init="pca",
    ).fit_transform(vecs50)
    print(f"[tsne]  {vecs.shape[1]}D → 50D → 2D")
    return reduced

def plot(coords, words, freqs, title, filename):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("#0f0f17")
    ax.set_facecolor("#0f0f17")

    # dot size and colour scaled by frequency
    sizes  = 20 + 200 * (freqs / freqs.max()) ** 0.5
    colors = freqs / freqs.max()

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap="cool",
        s=sizes, alpha=0.85,
        linewidths=0.3, edgecolors="white",
        zorder=2,
    )

    # labels with outline so they're readable on dark background
    outline = [
        pe.Stroke(linewidth=2, foreground="#0f0f17"),
        pe.Normal(),
    ]
    for i, word in enumerate(words):
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7.5,
            color="white",
            path_effects=outline,
            zorder=3,
        )

    # colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Relative frequency", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # styling
    ax.set_title(title, color="white", fontsize=14, pad=14, fontweight="bold")
    ax.tick_params(colors="#555577")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222233")
    ax.grid(True, color="#1c1c2e", linewidth=0.5, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[save]  {path}")

if __name__ == "__main__":
    print("=" * 55)
    print("  word2vec — Embedding Visualization")
    print("=" * 55)

    W_in, tokens, idx2word = load()
    vecs, words, freqs     = select_top(W_in, tokens, idx2word, TOP_N)

    # PCA plot
    pca_coords = reduce_pca(vecs)
    plot(pca_coords, words, freqs,
         title=f"word2vec Embeddings — PCA  (top {TOP_N} words)",
         filename="pca_embeddings.png")

    # t-SNE plot
    tsne_coords = reduce_tsne(vecs)
    plot(tsne_coords, words, freqs,
         title=f"word2vec Embeddings — t-SNE  (top {TOP_N} words, perplexity={TSNE_PERPLEXITY})",
         filename="tsne_embeddings.png")

    print(f"\nDone — check the plots/ folder")