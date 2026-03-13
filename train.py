import numpy as np
import time
import os
import json
from negative_sampleing import (
    generate_skipgram_pairs,
    build_negative_distribution,
    negative_sampling,
    tokens,
    vocab_size,
    idx2word,
)

EMBED_DIM     = 100
WINDOW_SIZE   = 2
K_NEGATIVES   = 5
EPOCHS        = 5
BATCH_SIZE    = 256
LEARNING_RATE = 0.025
LR_MIN        = 0.0001
SEED          = 42
DATA_DIR      = "data"

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def init_embeddings(vocab_size: int, embed_dim: int, seed: int = SEED):
    """
    W_in  : center word embeddings  — what we care about after training
    W_out : context word embeddings — auxiliary, discarded after training

    Small uniform init for W_in, zeros for W_out.
    """
    rng   = np.random.default_rng(seed)
    bound = 0.5 / embed_dim
    W_in  = rng.uniform(-bound, bound, (vocab_size, embed_dim)).astype(np.float32)
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    print(f"[init]  W_in {W_in.shape}  W_out {W_out.shape}")
    return W_in, W_out

def sgns_step(
    centers:   np.ndarray,   # (B,)     int32  — center word indices
    contexts:  np.ndarray,   # (B,)     int32  — positive context indices
    negatives: np.ndarray,   # (B, K)   int32  — noise word indices
    W_in:      np.ndarray,   # (V, D)   float32
    W_out:     np.ndarray,   # (V, D)   float32
    lr:        float,
) -> float:
    """
    One mini-batch forward + backward + SGD update.
    Returns mean batch loss.

    Loss per pair:
        L = -log σ(v_c · v_pos) - Σ_k log σ(-v_c · v_neg_k)

    Gradients:
        δ_pos        = σ(s_pos) - 1                      (B,)
        δ_neg        = σ(s_neg)                          (B, K)

        ∂L/∂v_c      = δ_pos · v_pos + Σ_k δ_neg_k · v_neg_k
        ∂L/∂v_pos    = δ_pos · v_c
        ∂L/∂v_neg_k  = δ_neg_k · v_c
    """
    # ---- forward: embedding lookup ----
    v_c   = W_in[centers]               # (B, D)
    v_pos = W_out[contexts]             # (B, D)
    v_neg = W_out[negatives]            # (B, K, D)

    # ---- positive branch ----
    s_pos   = (v_c * v_pos).sum(axis=1)                        # (B,)
    sig_pos = sigmoid(s_pos)                                   # (B,)
    loss_pos = -np.log(sig_pos + 1e-7)                         # (B,)

    # ---- negative branch ----
    s_neg   = np.einsum('bd,bkd->bk', v_c, v_neg)             # (B, K)
    sig_neg = sigmoid(s_neg)                                   # (B, K)
    loss_neg = -np.log(1.0 - sig_neg + 1e-7)                  # (B, K)

    # ---- total loss ----
    loss = (loss_pos + loss_neg.sum(axis=1)).mean()

    # ---- gradients ----
    d_pos = (sig_pos - 1.0)[:, None]                          # (B, 1)
    d_neg = sig_neg[:, :, None]                                # (B, K, 1)

    grad_center = d_pos * v_pos + (d_neg * v_neg).sum(axis=1) # (B, D)
    grad_pos    = d_pos * v_c                                  # (B, D)
    grad_neg    = d_neg * v_c[:, None, :]                      # (B, K, D)

    # ---- SGD update — np.add.at handles repeated indices correctly ----
    np.add.at(W_in,  centers,   -lr * grad_center)
    np.add.at(W_out, contexts,  -lr * grad_pos)
    np.add.at(W_out, negatives, -lr * grad_neg)

    return float(loss)

def train(
    pairs:     np.ndarray,
    probs:     np.ndarray,
    W_in:      np.ndarray,
    W_out:     np.ndarray,
) -> list[float]:

    rng         = np.random.default_rng(SEED)
    N           = len(pairs)
    total_steps = EPOCHS * (N // BATCH_SIZE)
    step        = 0
    epoch_losses = []

    print(f"[train] {N:,} pairs  |  batch={BATCH_SIZE}  epochs={EPOCHS}  "
          f"K={K_NEGATIVES}  D={W_in.shape[1]}")
    print(f"        lr {LEARNING_RATE} -> {LR_MIN}  (linear decay)\n")

    for epoch in range(1, EPOCHS + 1):
        t0  = time.time()
        idx = rng.permutation(N)
        shuffled_pairs = pairs[idx]

        losses = []
        for start in range(0, N - BATCH_SIZE + 1, BATCH_SIZE):

            # linear LR decay
            lr_t = max(LEARNING_RATE * (1.0 - step / max(total_steps, 1)), LR_MIN)

            batch    = shuffled_pairs[start : start + BATCH_SIZE]
            centers  = batch[:, 0]                              # (B,)
            contexts = batch[:, 1]                              # (B,)

            # sample negatives using your negative_sampling function
            negatives = np.array(
                [negative_sampling(ctx, probs, k=K_NEGATIVES) for ctx in contexts],
                dtype=np.int32,
            )                                                   # (B, K)

            loss = sgns_step(centers, contexts, negatives, W_in, W_out, lr_t)
            losses.append(loss)
            step += 1

        mean_loss = float(np.mean(losses))
        epoch_losses.append(mean_loss)
        print(f"  epoch {epoch}/{EPOCHS}  loss={mean_loss:.4f}  "
              f"lr={lr_t:.5f}  time={time.time()-t0:.1f}s")

    return epoch_losses

def save_embeddings(W_in: np.ndarray, W_out: np.ndarray, data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "W_in.npy"),  W_in)
    np.save(os.path.join(data_dir, "W_out.npy"), W_out)
    print(f"\n[save]  data/W_in.npy   {W_in.shape}")
    print(f"        data/W_out.npy  {W_out.shape}")

# ---------------------------------------------------------------------------
# NEAREST NEIGHBOURS  (quick sanity check after training)
# ---------------------------------------------------------------------------

def nearest_neighbours(word: str, W_in: np.ndarray, top_n: int = 5):
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        word2idx = json.load(f)

    if word not in word2idx:
        print(f"  '{word}' not in vocabulary"); return

    idx    = word2idx[word]
    vec    = W_in[idx]
    norms  = np.linalg.norm(W_in, axis=1) + 1e-9
    scores = W_in @ vec / (norms * np.linalg.norm(vec))
    scores[idx] = -1

    top = np.argsort(scores)[::-1][:top_n]
    print(f"\n  Nearest neighbours of '{word}':")
    for rank, i in enumerate(top, 1):
        neighbour = idx2word.get(str(i), idx2word.get(i, "?"))
        print(f"    {rank}. {neighbour:<20} cosine={scores[i]:.3f}")


if __name__ == "__main__":
    print("=" * 55)
    print("  word2vec SGNS — Training")
    print("=" * 55)

    # build pairs and noise distribution from your script
    print("\n[prep]  Generating skip-gram pairs ...")
    pairs = np.array(generate_skipgram_pairs(tokens, window_size=WINDOW_SIZE),
                     dtype=np.int32)
    probs = build_negative_distribution(tokens, vocab_size)
    print(f"        {len(pairs):,} pairs ready")

    # initialise embeddings
    W_in, W_out = init_embeddings(vocab_size, EMBED_DIM)

    # train
    train(pairs, probs, W_in, W_out)

    # save
    save_embeddings(W_in, W_out)

    # spot-check embeddings
    print("\n[eval]  Nearest neighbour spot-checks:")
    for query in ["citizen", "speak", "good", "people"]:
        nearest_neighbours(query, W_in)

    print("\nTraining complete.")