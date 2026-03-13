import numpy as np
import json
import collections

tokens = np.load("data/tokens.npy")

with open("data/vocab_stats.json") as f:
    stats = json.load(f)
vocab_size = stats["vocab_size"]

with open("data/idx2word.json") as f:
    idx2word = json.load(f)


def w(idx: int) -> str:
    """Helper: index -> word, works with both int and str keys."""
    return idx2word.get(str(idx), idx2word.get(idx, "?"))


def generate_skipgram_pairs(token_ids: np.ndarray, window_size: int = 2) -> list[tuple[int, int]]:
    pairs = []
    for i, target in enumerate(token_ids):
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)
        for j in range(start, end):
            if i == j:
                continue
            context = token_ids[j]
            pairs.append((target, context))
    return pairs


def build_negative_distribution(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(token_ids, minlength=vocab_size)
    probs = counts ** 0.75
    probs = probs / probs.sum()
    return probs


def negative_sampling(context_word: int, probs: np.ndarray, k: int = 5) -> list[int]:
    negatives = []
    while len(negatives) < k:
        candidates = np.random.choice(len(probs), size=k * 2, p=probs)
        for neg in candidates:
            if neg != context_word and len(negatives) < k:
                negatives.append(int(neg))
    return negatives


# ---------------------------------------------------------------------------
# SANITY CHECKS
# ---------------------------------------------------------------------------

def check_pairs(pairs, window_size):
    print(f"\n{'='*55}")
    print("  CHECK 1 — Skip-gram pair generation")
    print(f"{'='*55}")

    # 1a. Basic counts
    print(f"  Total pairs generated : {len(pairs):,}")
    assert len(pairs) > 0, "No pairs generated!"

    # 1b. Total pairs ≤ T * 2 * window_size  (word index repeats across positions,
    #     so we check the global total, not per-word-index count)
    T = len(tokens)
    max_possible = T * 2 * window_size
    print(f"  Max possible pairs    : {max_possible:,}  "
          f"({T} tokens × {2*window_size} max context)")
    assert len(pairs) <= max_possible, \
        f"More pairs ({len(pairs)}) than theoretically possible ({max_possible})!"

    # 1c. Pairs where center == context (same word index, different positions)
    #     are technically allowed in word2vec — the model learns "the" predicts "the".
    #     Flag them as a warning rather than a hard failure.
    same_word_pairs = [(c, ctx) for c, ctx in pairs if c == ctx]
    pct = 100 * len(same_word_pairs) / len(pairs)
    print(f"  Same-word pairs       : {len(same_word_pairs)}  ({pct:.1f}%)  "
          f"<- allowed, but worth knowing")

    # 1d. Print first 6 pairs as human-readable words
    print(f"\n  First 6 pairs (center -> context):")
    for center, context in pairs[:6]:
        print(f"    '{w(center)}'  ->  '{w(context)}'")

    print("Pairs look correct")


def check_distribution(probs, vocab_size):
    print(f"\n{'='*55}")
    print("  CHECK 2 — Negative sampling distribution")
    print(f"{'='*55}")

    # 2a. Shape and dtype
    print(f"  probs.shape : {probs.shape}  (expected ({vocab_size},))")
    assert probs.shape == (vocab_size,), "Wrong shape!"

    # 2b. Sums to 1
    total = probs.sum()
    print(f"  probs.sum() : {total:.6f}  (expected ≈ 1.0)")
    assert abs(total - 1.0) < 1e-5, f"Probabilities don't sum to 1: {total}"

    # 2c. No negative values
    assert (probs >= 0).all(), "Negative probabilities found!"

    # 2d. Show top-5 most likely noise words — should be frequent words
    top5 = np.argsort(probs)[::-1][:5]
    print(f"\n  Top-5 noise words (should be frequent):")
    for idx in top5:
        print(f"    [{idx:4d}]  '{w(idx)}'   p={probs[idx]:.4f}")

    # 2e. Distribution is smoother than raw frequency (¾ power check)
    raw_counts = np.bincount(tokens, minlength=vocab_size).astype(np.float64)
    raw_probs  = raw_counts / raw_counts.sum()
    # The smoothed distribution should be less extreme: min prob should be higher
    assert probs[probs > 0].min() >= raw_probs[raw_probs > 0].min() - 1e-9, \
        "Smoothing did not raise minimum probability!"
    print(f"\n  Raw min prob    : {raw_probs[raw_probs>0].min():.6f}")
    print(f"  Smoothed min    : {probs[probs>0].min():.6f}  (should be ≥ raw)")
    print("  ✓  Distribution looks correct")


def check_negative_sampling(probs, pairs):
    print(f"\n{'='*55}")
    print("  CHECK 3 — Negative sampling")
    print(f"{'='*55}")

    K = 5
    # Pick a real context word from the pairs
    _, context_word = pairs[0]
    negatives = negative_sampling(context_word, probs, k=K)

    # 3a. Correct count
    print(f"  Context word : '{w(context_word)}'  (idx={context_word})")
    print(f"  k requested  : {K}")
    print(f"  Negatives    : {[w(n) for n in negatives]}")
    assert len(negatives) == K, f"Expected {K} negatives, got {len(negatives)}"

    # 3b. No negative equals the context word
    assert context_word not in negatives, \
        f"Context word '{w(context_word)}' appeared in its own negatives!"

    # 3c. All indices are valid vocab indices
    assert all(0 <= n < len(probs) for n in negatives), \
        "Negative index out of vocabulary range!"

    # 3d. Run many times to check empirical distribution is reasonable
    TRIALS = 2000
    all_negs = []
    for _, ctx in pairs[:TRIALS // K]:
        all_negs.extend(negative_sampling(ctx, probs, k=K))
    empirical = collections.Counter(all_negs)
    top3_empirical = [w(idx) for idx, _ in empirical.most_common(3)]
    print(f"\n  Top-3 sampled words over {TRIALS} draws: {top3_empirical}")
    print(f"  (should overlap with frequent words from CHECK 2)")
    print("Negative sampling looks correct")


def check_training_shapes(pairs, probs):
    """Simulate what the training loop will do with one mini-batch."""
    print(f"\n{'='*55}")
    print("  CHECK 4 — Training batch shapes")
    print(f"{'='*55}")

    B, K = 8, 5
    pairs_arr = np.array(pairs, dtype=np.int32)

    batch     = pairs_arr[:B]
    centers   = batch[:, 0]          # (B,)
    contexts  = batch[:, 1]          # (B,)
    negatives = np.array(
        [negative_sampling(ctx, probs, k=K) for ctx in contexts],
        dtype=np.int32
    )                                # (B, K)

    print(f"  centers   shape : {centers.shape}   dtype={centers.dtype}")
    print(f"  contexts  shape : {contexts.shape}   dtype={contexts.dtype}")
    print(f"  negatives shape : {negatives.shape}  dtype={negatives.dtype}")

    assert centers.shape   == (B,),    "centers shape wrong"
    assert contexts.shape  == (B,),    "contexts shape wrong"
    assert negatives.shape == (B, K),  "negatives shape wrong"

    print(f"\n  Sample batch (first 3 rows):")
    print(f"  {'center':<12} {'context':<12} {'negatives'}")
    for i in range(3):
        neg_words = [w(n) for n in negatives[i]]
        print(f"  {w(centers[i]):<12} {w(contexts[i]):<12} {neg_words}")

    print("Batch shapes are training-loop ready")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  word2vec — Step 2 Sanity Checks")
    print("=" * 55)
    print(f"\n  Corpus : {len(tokens):,} tokens  |  vocab: {vocab_size:,} words")

    WINDOW_SIZE = 2

    pairs = generate_skipgram_pairs(tokens, window_size=WINDOW_SIZE)
    probs = build_negative_distribution(tokens, vocab_size)

    check_pairs(pairs, window_size=WINDOW_SIZE)
    check_distribution(probs, vocab_size)
    check_negative_sampling(probs, pairs)
    check_training_shapes(pairs, probs)

    print(f"\n{'='*55}")
    print("  All checks passed ✓")
    print(f"  Ready for train.py")
    print(f"{'='*55}\n")