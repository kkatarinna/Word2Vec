# Word2Vec
Implementation of word2vec - skip gram - from scratch
## Core Modules

### vocabulary.py
This module handles the preprocessing of text data for the Word2Vec model. It includes functions to:
- Load text from a file (e.g., `data/tiny-shakespeare.txt`).
- Clean the text by converting to lowercase, removing punctuation, and normalizing whitespace.
- Tokenize the cleaned text into a list of words.
- Build a vocabulary by filtering words based on minimum frequency and maximum vocabulary size, creating mappings from words to indices (`word2idx`) and vice versa (`idx2word`).
- Encode tokens into integer IDs, handling unknown words with a special `<UNK>` token.
- Save the processed data, vocabulary, and statistics to the `data/` directory.

When run as a script, it processes the Shakespeare text and saves the artifacts for use in training.

### negative_sampleing.py
This module implements the negative sampling mechanism for efficient training of the Skip-Gram model. Key functionalities include:
- Generating skip-gram pairs from token sequences using a specified window size.
- Building a smoothed probability distribution for negative sampling based on word frequencies (raised to the power of 0.75).
- Performing negative sampling to select noise words for each context word.
- Comprehensive sanity checks to validate pair generation, distribution correctness, negative sampling, and training batch shapes.

It loads preprocessed data from `vocabulary.py` and ensures the data is ready for the training phase in `train.py`.

### train.py
This is the main training script for the Word2Vec Skip-Gram with Negative Sampling (SGNS) model. It includes:
- Initialization of input and output embedding matrices.
- The core training step (`sgns_step`) that computes the loss, gradients, and updates embeddings using stochastic gradient descent.
- A full training loop with mini-batching, learning rate decay, and epoch-based training.
- Saving the trained embeddings to `data/W_in.npy` and `data/W_out.npy`.
- A nearest neighbors function for quick evaluation of the learned embeddings.

The script trains on the pairs and distribution prepared by `negative_sampleing.py`, producing word embeddings that can be used for tasks like similarity queries.


## Idea
Word2Vec learns **vector representations (embeddings)** of words such that words appearing in similar contexts have similar vectors.
In the **skip-gram model**, the task is to predict surrounding context words given a center word.

Example sentence:

`the king loves the queen`

With a context window of 2, the training pairs are:

```
(center -> context)

king -> the
king -> loves
king -> the
king -> queen
```

The model learns embeddings so that words that appear in similar contexts (e.g. king and queen) obtain similar vector representations.

### Data Pipeline
The preprocessing pipeline consists of the following steps:

#### 1. Load dataset
The Tiny Shakespeare corpus is used as the training dataset.

#### 2. Text cleaning
- convert to lowercase
- remove non-alphabetic characters
- normalize whitespace

#### 3. Tokenization
The cleaned text is split into word tokens.

#### 4. Vocabulary construction
- count word frequencies
- remove rare words (`min_freq`)
- keep the most frequent words (`max_vocab`)
- add a special `<UNK>` token for unknown words

#### 5. Encoding
Tokens are converted into integer IDs using the vocabulary mapping.

---

### Skip-Gram Training Data

Training samples are generated using a **sliding context window**.

For each token position `i`:

- the **center word** is `tokens[i]`
- **context words** are tokens within the window around it

This produces pairs:
`(center_word, context_word)`

---

### Embeddings

Two embedding matrices are learned during training:
```
W_in : center word embeddings
W_out : context word embeddings
```

Each row corresponds to the embedding vector of one word in the vocabulary.
```
W_in shape = (vocab_size, embedding_dim)
W_out shape = (vocab_size, embedding_dim)
```

After training, **W_in is used as the final word embedding matrix**.

---

### Negative Sampling

Instead of computing a full softmax over the vocabulary, the model uses **negative sampling**.

For each positive pair `(center, context)`:

- the context word is treated as a **positive example**
- `K` random words are sampled as **negative examples**

The sampling distribution is based on word frequencies:
`P(w) ∝ frequency(w)^0.75`

This favors frequent words while still allowing rare words to appear as negatives.

---

### Training Objective

For a center word vector `v_c`, a positive context vector `v_pos`, and negative samples `v_neg`, the loss is:

```
L = -log σ(v_c · v_pos)
- Σ_k log σ(-v_c · v_neg_k)
```


where `σ` is the sigmoid function.

The model learns embeddings by **maximizing similarity between true word pairs** and **minimizing similarity with randomly sampled words**.

---

### Optimization

Training is performed using **mini-batch stochastic gradient descent (SGD)**:

1. Lookup embeddings for center, positive, and negative words
2. Compute scores via dot products
3. Compute the sigmoid loss
4. Calculate gradients for all embeddings
5. Update parameters using SGD

Updates are applied directly to the embedding matrices using NumPy.

---

### Evaluation

After training, embeddings can be inspected using:

#### Nearest Neighbours

For a given word, cosine similarity is used to find the most similar words in the embedding space:
