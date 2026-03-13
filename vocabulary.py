import numpy as np
import pandas as pd
import re
import collections
import json
import os

SPECIAL_TOKENS = ["<UNK>"]
FILE_PATH = "data/tiny-shakespeare.txt"

def load_text() -> str:
    if FILE_PATH and os.path.exists(FILE_PATH):
        print(f"[load]  Reading from file: {FILE_PATH}")
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"        {len(text):,} characters loaded.")
    else:
        print("[load]  File not found — no data loaded.")
        text = ""
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-z'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenise(text: str) -> list[str]:
    return text.split()

def build_vocab(
    tokens: list[str],
    min_freq: int = 2,
    max_vocab: int = 10000,
) -> tuple[dict[str, int], dict[int, str]]:
    freq = collections.Counter(tokens)
    print(f"\n[vocab] Raw vocabulary size : {len(freq):,} unique tokens")
    print(f"        Total tokens          : {len(tokens):,}")

    # Filter by minimum frequency
    freq = {w: c for w, c in freq.items() if c >= min_freq}
    print(f"        After min_freq={min_freq}    : {len(freq):,} unique tokens")

    # Keep top-N by frequency
    top_words = sorted(freq, key=freq.get, reverse=True)[:max_vocab - len(SPECIAL_TOKENS)]

    # Build mappings  (special tokens always get the lowest indices)
    word2idx: dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        word2idx[tok] = len(word2idx)       # <UNK> → 0
    for word in top_words:
        word2idx[word] = len(word2idx)

    idx2word: dict[int, str] = {i: w for w, i in word2idx.items()}

    print(f"        Final vocab size        : {len(word2idx):,} "
          f"(incl. {len(SPECIAL_TOKENS)} special token(s))")
    return word2idx, idx2word

def encode(tokens: list[str], word2idx: dict[str, int]) -> np.ndarray:
    unk_idx = word2idx["<UNK>"]
    ids = np.array([word2idx.get(t, unk_idx) for t in tokens], dtype=np.int32)
    unk_count = int((ids == unk_idx).sum())
    print(f"\n[encode] Encoded {len(ids):,} tokens  "
          f"({unk_count:,} → <UNK>,  {unk_count/len(ids)*100:.1f}%)")
    return ids

def save_artefacts(
    token_ids: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    out_dir: str = "data",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tokens_path   = os.path.join(out_dir, "tokens.npy")
    vocab_path    = os.path.join(out_dir, "vocab.json")
    idx2word_path = os.path.join(out_dir, "idx2word.json")
    stats_path    = os.path.join(out_dir, "vocab_stats.json")

    np.save(tokens_path, token_ids)

    with open(vocab_path, "w")    as f: json.dump(word2idx, f, indent=2)
    with open(idx2word_path, "w") as f: json.dump(idx2word, f, indent=2)

    stats = {
        "vocab_size"   : len(word2idx),
        "total_tokens" : int(len(token_ids)),
        "unk_index"    : 0,
        "special_tokens": SPECIAL_TOKENS,
        "min_freq"     : 2,
        "max_vocab"    : 10000,
    }
    with open(stats_path, "w") as f: json.dump(stats, f, indent=2)

    print(f"\n[save]  {tokens_path}   — integer token array")
    print(f"        {vocab_path}     — word → index")
    print(f"        {idx2word_path}  — index → word")
    print(f"        {stats_path}     — corpus statistics")


if __name__ == "__main__":
    text = load_text()
    cleaned_text = clean_text(text)
    tokens = tokenise(cleaned_text)
    word2idx, idx2word = build_vocab(tokens, min_freq=2, max_vocab=10000)
    ids = encode(tokens, word2idx)
    save_artefacts(ids, word2idx, idx2word, out_dir="data")