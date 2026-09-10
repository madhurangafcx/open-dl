# 08 — Sentiment Analysis RNN From Scratch

## Goal

Build and verify a pure-NumPy, word-level **Many-to-One** RNN for binary
sentiment classification. The implementation is in
[`src/sentiment_rnn.py`](src/sentiment_rnn.py); the detailed derivation is in
[`steps/step-01-mathematics.md`](steps/step-01-mathematics.md).

The model reads a variable-length review, preserves its final meaningful hidden
state, and produces one probability:

```text
review text → tokens → token IDs → embeddings → masked RNN → final hidden state
            → sentiment logit → sigmoid probability → positive / negative
```

## Implemented Model

For every real word at timestep `t`:

```math
e_t = E[w_t]
```

```math
z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h
```

```math
h_t = \tanh(z_t)
```

For each review, classification uses the hidden state at its actual length `L`:

```math
o = h_L W_{hy} + b_y
```

```math
\hat{y} = \sigma(o)
```

The loss is binary cross-entropy:

```math
L = -[y\ln(\hat{y}+\epsilon) + (1-y)\ln(1-\hat{y}+\epsilon)]
```

## Current Features

- lowercasing, punctuation removal, and whitespace tokenization
- fixed teaching vocabulary with `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>`
- token-ID encoding, post-padding, and actual-length tracking
- learnable `(V, D)` embedding matrix
- padding-safe forward and backward passes: `<PAD>` freezes the hidden state
- Many-to-One RNN with `tanh`, sigmoid, and binary cross-entropy
- BPTT into recurrent parameters and active embedding rows
- global-norm gradient clipping and gradient-descent updates
- batch training, threshold predictions, and training accuracy
- finite-difference verification of an embedding gradient

## Teaching Configuration

```text
Vocabulary size V       = 16
Embedding dimension D   = 4
Hidden-state size H     = 3
Binary output size      = 1
Total learnable values  = 92
```

| Parameter | Shape | Count |
| :--- | :--- | ---: |
| Embedding matrix `E` | `(16, 4)` | 64 |
| Embedding-to-hidden `W_xh` | `(4, 3)` | 12 |
| Recurrent `W_hh` | `(3, 3)` | 9 |
| Hidden bias `b_h` | `(1, 3)` | 3 |
| Hidden-to-sentiment `W_hy` | `(3, 1)` | 3 |
| Output bias `b_y` | `(1, 1)` | 1 |

## Vocabulary

```text
0 <PAD>   1 <UNK>   2 <BOS>   3 <EOS>
4 movie   5 great   6 terrible  7 the
8 really  9 bad    10 acting   11 was
12 film  13 boring 14 not      15 good
```

`<PAD>` is index `0` and its initial embedding is zero. The model still masks
padding because recurrent weights and biases would otherwise change hidden state
on padded timesteps.

## Training Dataset

The implementation trains on this eight-review educational batch:

| Review | Label |
| :--- | ---: |
| `acting was good` | 1 |
| `movie was great` | 1 |
| `the film was great` | 1 |
| `movie was not terrible` | 1 |
| `acting was terrible` | 0 |
| `movie was bad` | 0 |
| `the film was boring` | 0 |
| `acting was not good` | 0 |

It is deliberately tiny and exists to verify the mathematics and training loop;
it is not a real-world sentiment dataset.

## Verification Performed

The fixed walkthrough for `"acting was not good"` produces:

```text
final hidden state       [ 0.198918,  0.170715, -0.093722]
positive probability     0.525741
binary cross-entropy     0.746003
```

The training batch reaches `1.0` training accuracy. A finite-difference check
also agrees with BPTT for an embedding value:

```text
analytical dE[good, 0] ≈ 0.188722
numerical  dE[good, 0] ≈ 0.188722
```

Run the lesson:

```bash
python src/sentiment_rnn.py
```

## Project Structure

```text
08-sentiment-rnn/
├── README.md
├── src/
│   └── sentiment_rnn.py
└── steps/
    └── step-01-mathematics.md
```

## Not Implemented Yet

- corpus-derived vocabulary construction and vocabulary-frequency analysis
- confusion matrix, precision, recall, and F1
- learned-embedding similarity or hidden-state visualizations
- long-sequence vanishing/exploding-gradient experiment
- PyTorch implementation and performance comparison
