# 11 — Sequence Prediction & Autoregressive Modeling From Scratch

## Goal

Build and understand an **Autoregressive Neural Language Model** from first principles with pure NumPy. This module establishes the mathematical, architectural, and algorithmic foundation for all modern generative sequence prediction systems—from classical Recurrent Neural Networks (RNNs) to state-of-the-art Large Language Models (LLMs).

Under the probabilistic chain rule, the joint probability of any discrete text sequence $w = (w_1, w_2, \dots, w_T)$ factorizes into an ordered product of conditional next-token probabilities:

```math
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \dots, w_{t-1})
```

The network implements the canonical autoregressive sequence forward pipeline:

```text
e_t       = embedding_forward(token_t, E)
z_{h, t}  = e_t @ W_xh + h_{t-1} @ W_hh + b_h
h_t       = tanh(z_{h, t})
logits_t  = h_t @ W_hy + b_y
probs_t   = softmax(logits_t)
loss_t    = -ln(probs_{t, target_t})
```

Where:

- `token_t` ($w_{t-1}$) is the input context token ID at step `t`.
- `E` is the learnable vocabulary embedding matrix (shape `(V, D)`).
- `e_t` is the continuous dense representation coordinates of token `token_t` (shape `(B, D)`).
- `h_{t-1}` is the previous recurrent context vector summarizing all historical tokens $w_{<t}$ (shape `(B, H)`).
- `W_xh` and `W_hh` are input-to-hidden and recurrent hidden-to-hidden transition matrices.
- `b_h` is the recurrent hidden state bias vector (shape `(1, H)`).
- `h_t` is the updated context state vector compressed via hyperbolic tangent bounding.
- `W_hy` and `b_y` project the hidden representation into the full vocabulary space (shape `(H, V)` and `(1, V)`).
- `logits_t` and `probs_t` are unnormalized scores and normalized probabilities across all $V$ candidate words.
- `loss_t` is the negative log-likelihood (NLL) incurred for the true ground-truth target token.

```text
Input Sequence X:   [ <BOS>,       crypto,        blocks     ]  (Tokens at t = 0, 1, 2)
                         │              │              │
                         ▼              ▼              ▼
Embedding Table:      [ e_1 ]        [ e_2 ]        [ e_3 ]
                         │              │              │
                         ▼              ▼              ▼
Recurrent State:      h_0 ──►[ Cell ] ──►[ Cell ] ──►[ Cell ] ──► h_3
                                │              │              │
                                ▼              ▼              ▼
Output Projection:            [ o_1 ]        [ o_2 ]        [ o_3 ]
                                │              │              │
                                ▼              ▼              ▼
Softmax Distribution:         y_hat_1        y_hat_2        y_hat_3
                                │              │              │
Target Sequence Y:  [ crypto,        blocks,        verify    ]  (Tokens at t = 1, 2, 3)
```

## Project structure

```text
11-sequence-prediction/
├── README.md
├── src/
│   └── sequence_model.py         # Pure NumPy autoregressive language model & sampling suite
└── steps/
    └── step-01-mathematics.md    # Complete mathematical foundations (Q1–Q12)
```

## Learning workflow

Every project in this repository adheres to the standard 10-step sequence:

| Step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Mathematics | Complete | [`steps/step-01-mathematics.md`](steps/step-01-mathematics.md) (Q1–Q12: probabilistic chain rule, causal sequence shift topology, dense embedding matrix, recurrent state compression, categorical cross-entropy, perplexity proof, decoding algorithms, 3-step forward trace, BPTT calculus with sparse embedding updates, and code mapping) |
| 2. NumPy implementation | Complete | [`src/sequence_model.py`](src/sequence_model.py): 3-tier hierarchy (`recurrent_cell_forward`, `sequence_model_forward`, `train_sequence_model`), sparse embedding updates via `np.add.at`, batch-mean scaling, and full decoding suite (Greedy, Temperature, Top-K, Top-P) |
| 3. Understand forward pass | Complete | Verified exact hidden states, vocabulary logits, and per-token losses matching manual derivation down to 6 decimal places ($L_1 = 1.722698, L_2 = 1.665319, L_3 = 1.515687, L_{\text{total}} = 4.903704, \text{PPL} = 5.127242$) |
| 4. Understand loss | Complete | Negative log-likelihood with numerical floor, sequence cross-entropy, and Perplexity $\text{PPL} = \exp(\bar{L})$ evaluating effective vocabulary branching factor |
| 5. Derive gradients | Complete | Derived analytical gradients for readout projection ($dW_{hy}, db_y$), recurrent transition ($dW_{xh}, dW_{hh}, db_h$), and sparse embedding row accumulation ($dE$) |
| 6. Implement backpropagation | Complete | Implemented full BPTT in `sequence_model_backward` with exact sparse row updates into embedding table $E$ |
| 7. Train model | Complete | Trained sequence model on `<BOS> crypto blocks verify <EOS>`, reducing loss to $0.0135$ and perplexity to $1.00$, and verified autonomous free-running text generation |
| 8. Debug & analyze | Complete | Validated all parameter tensors against symmetric finite differences ($< 10^{-10}$ rel error) and verified $B = 2$ batched gradient scaling |
| 9. PyTorch implementation | Pending | Equivalent pipeline using `torch.nn.Embedding`, `torch.nn.RNN`, and `torch.nn.Linear` |
| 10. Compare results | Pending | Compare NumPy vs. PyTorch forward distributions, embedding gradient accumulation, and decoding trajectories |

## Parameters and tensor shapes

For the baseline verification example in `src/sequence_model.py` ($V = 5, D = 3, H = 4, T = 3, B = 1$):

| Layer / Parameter | Tensor Key | Shape | Parameters | Mathematical Role |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding Table** | `E` | (5, 3) | 15 | Maps discrete token IDs to continuous coordinates |
| **Input-to-Hidden Weights** | `W_xh` | (3, 4) | 12 | Projects token embeddings into recurrent hidden space |
| **Recurrent Transition Weights**| `W_hh` | (4, 4) | 16 | Projects previous context $h_{t-1}$ to new hidden state |
| **Hidden Bias** | `b_h` | (1, 4) | 4 | Additive bias vector for hidden state |
| **Vocabulary Projection Weights**| `W_hy` | (4, 5) | 20 | Projects hidden context to vocabulary logits |
| **Vocabulary Readout Bias** | `b_y` | (1, 5) | 5 | Additive bias vector for vocabulary logits |
| **Total Parameters** | — | — | **72** | $V \times D + D \times H + H^2 + H + H \times V + V$ |

### General Parameter Budget Formula:

```math
P_{\text{total}} = V \times D + D \times H + H^2 + H + H \times V + V
```

For large-scale models, the embedding table $E$ and vocabulary projection $W_{hy}$ dominate the parameter budget:
- When $V = 50,000$, $D = 768$, $H = 768$: $V \times D \approx 38.4\text{M}$ weights, $H \times V \approx 38.4\text{M}$ weights (together comprising over 85% of total parameters).

## NumPy implementation details

`src/sequence_model.py` follows the repository's 3-tier hierarchy and 9-part modular layout:

### 1. Token Embedding Forward & Sparse BPTT Accumulation

```python
# Forward: Extract row vector corresponding to token index
def embedding_forward(token_indices, embedding_matrix):
    return embedding_matrix[token_indices]

# Backward: Sparse index accumulation into embedding table
delta_e_t = delta_z_h @ parameters["W_xh"].T  # (B, D)
np.add.at(param_gradients["E"], token_sequence[:, timestep], delta_e_t)
```

If a word appears multiple times in a sequence, `np.add.at` safely sums all upstream gradient contributions into that word's embedding coordinates without race conditions.

### 2. Recurrent Context Compression (Micro-Flow)

```python
z_h_pre = (
    current_embedding @ parameters["W_xh"]
    + previous_hidden @ parameters["W_hh"]
    + parameters["b_h"]
)
current_hidden = np.tanh(z_h_pre)
```

### 3. Batch-Mean Logit Error Scaling

To ensure that analytical gradients differentiate the exact batch-mean loss reported by `sequence_cross_entropy`, `output_score_gradients` divides by batch size $B$:

```python
def output_score_gradients(output_probabilities, target_tokens):
    batch_size, sequence_length, vocab_size = output_probabilities.shape
    logit_errors = output_probabilities.copy()
    for t in range(sequence_length):
        target_ids = target_tokens[:, t]
        logit_errors[np.arange(batch_size), t, target_ids] -= 1.0
    return logit_errors / batch_size
```

### 4. The Autoregressive Decoding Suite

`src/sequence_model.py` implements the four canonical token selection strategies:

1. **Greedy Decoding**: `np.argmax(logits)` (deterministic, zero variance).
2. **Temperature Scaling**: Divides logits by $T_{\text{temp}} > 0$ before Softmax ($T < 1.0$ sharpens, $T > 1.0$ flattens).
3. **Top-K Truncation**: Masks all logits outside the top $K$ values to $-\infty$.
4. **Top-P (Nucleus) Sampling**: Dynamically filters the smallest set of tokens whose cumulative probability exceeds threshold $p \in (0, 1)$.

```python
def sample_next_token(logits, method="greedy", temperature=1.0, top_k=None, top_p=None):
    # Handles Greedy, Temperature, Top-K, and Top-P cumulative filtering
```

## Numerical verification & results

Running `src/sequence_model.py` verifies the exact numerical derivations from `steps/step-01-mathematics.md`:

### Checkpoint 1: Step-by-Step Forward Loss & Perplexity Parity

Vocabulary: `0: <BOS>`, `1: crypto`, `2: blocks`, `3: verify`, `4: <EOS>`.  
Input: `X = [0, 1, 2]` (`"<BOS> crypto blocks"`), Target: `Y = [1, 2, 3]` (`"crypto blocks verify"`).

| Timestep | Input Token | Target Token | Top Logit / Class | Target Prob $P(w_t^*)$ | Token Loss $L_t$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$t = 1$** | `0` (`<BOS>`) | `1` (`crypto`) | `2` (`blocks`, $o = 0.0865$) | $0.178584$ | **1.722698** |
| **$t = 2$** | `1` (`crypto`) | `2` (`blocks`) | `1` (`crypto`, $o = 0.1030$) | $0.189130$ | **1.665319** |
| **$t = 3$** | `2` (`blocks`) | `3` (`verify`) | `2` (`blocks`, $o = 0.1145$) | $0.219657$ | **1.515687** |
| **Total** | — | — | — | — | **$L_{\text{total}} = 4.903704$** |
| **Mean** | — | — | — | — | **$\bar{L} = 1.634568$** |
| **Perplexity** | — | — | — | — | **$\text{PPL} = \mathbf{5.127242}$** |

*All forward losses and perplexity match theoretical derivations down to 6 decimal places (tolerance: $10^{-4}$).*

### Checkpoint 2: Finite-Difference Gradient Checking

Analytical BPTT gradients (including sparse embedding updates) verified against central symmetric numerical differentiation ($\epsilon = 10^{-5}$):

| Parameter Tensor | Coordinate | Analytical Gradient | Numerical Gradient | Relative Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `E` (`<BOS>` Embed) | `[0, 1]` | `-0.3825505` | `-0.3825505` | $1.78 \times 10^{-11}$ | **PASS** |
| `E` (`crypto` Embed) | `[1, 0]` | `+0.1631298` | `+0.1631298` | $2.72 \times 10^{-11}$ | **PASS** |
| `E` (`blocks` Embed) | `[2, 2]` | `-0.0976560` | `-0.0976560` | $3.52 \times 10^{-11}$ | **PASS** |
| `W_xh` (Input-to-Hidden)| `[0, 2]` | `+0.0669980` | `+0.0669980` | $3.64 \times 10^{-11}$ | **PASS** |
| `W_hh` (Recurrent) | `[1, 1]` | `-0.0256816` | `-0.0256816` | $8.55 \times 10^{-11}$ | **PASS** |
| `b_h` (Hidden Bias) | `[0, 1]` | `-1.0643136` | `-1.0643136` | $2.20 \times 10^{-11}$ | **PASS** |
| `W_hy` (Readout Weights) | `[0, 0]` | `+0.0498870` | `+0.0498870` | $4.93 \times 10^{-11}$ | **PASS** |
| `b_y` (Readout Bias) | `[0, 1]` | `-0.4148526` | `-0.4148526` | $5.85 \times 10^{-11}$ | **PASS** |

*Batched scaling verified: duplicating the sequence into a $B = 2$ batch produces analytical gradients that match single-example values within $10^{-10}$ and numerical gradients within $10^{-6}$.*

### Checkpoint 3: Decoding & Sampling Verification on Step 3 Logits

Given step 3 logits $o_3 = [-0.021794, -0.067240, +0.114507, +0.105949, -0.088976]$:

| Decoding Strategy | Selected Token / Distribution | Theoretical Behavior |
| :--- | :--- | :--- |
| **Greedy Argmax** | Token `2` (`"blocks"`) | Deterministic peak logit selection |
| **Cold Temp ($T=0.5$)** | `[0.1855, 0.1694, 0.2436, 0.2395, 0.1621]` | Probability mass sharpens around top tokens |
| **Standard Temp ($T=1.0$)**| `[0.1933, 0.1847, 0.2215, 0.2197, 0.1808]` | Unscaled categorical distribution |
| **Hot Temp ($T=2.0$)** | `[0.1968, 0.1924, 0.2107, 0.2098, 0.1903]` | Flattens towards uniform distribution ($0.20$) |
| **Top-K ($K = 2$)** | `[0.0, 0.0, 0.502139, 0.497861, 0.0]` | Probability restricted strictly to `"blocks"` and `"verify"` |
| **Top-P ($p = 0.70$)** | `[0.235968, 0.225484, 0.270426, 0.268122, 0.0]` | Nucleus set `{blocks, verify, <BOS>, crypto}`, `<EOS>` pruned |

### Checkpoint 4: Training Convergence & Autoregressive Text Generation

Training the sequence model on `<BOS> crypto blocks verify <EOS>`:

```text
Epoch   1/150 | Loss: 6.4463 | Perplexity: 5.01 (Near random guessing over V = 5)
Epoch  40/150 | Loss: 0.0893 | Perplexity: 1.02
Epoch 150/150 | Loss: 0.0135 | Perplexity: 1.00 (Near-absolute certainty)
```

Free-running autonomous text generation from prompt `"<BOS>"`:
- **Prompt**: `"<BOS>"`
- **Generated**: `"<BOS> crypto blocks verify <EOS>"`

## Run the project

Run the complete validation suite from the repository root:

```bash
python3 11-sequence-prediction/src/sequence_model.py
```

Or execute directly from the `11-sequence-prediction/src` directory:

```bash
python3 sequence_model.py
```

## Key takeaways

1. **The Causal Shift Topology**: Next-token prediction requires structuring training pairs with an offset of $+1$: input $X = w_{<T}$ and target $Y = w_{1:T}$. At every step $t$, the loss directly enforces the conditional probability $P(w_t \mid w_{<t})$.
2. **Dense Embedding Matrix & Sparse Updates**: Instead of multiplying massive one-hot vectors, embedding lookups extract continuous coordinate slices $E[w_t, :]$. During BPTT, gradients accumulate sparsely into embedding rows using `np.add.at`.
3. **Physical Meaning of Perplexity**: Perplexity is the exponentiated mean cross-entropy $\text{PPL} = \exp(\bar{L})$, representing the effective branching factor. A model with $\text{PPL} = 5.0$ on vocabulary $V = 5$ is guessing randomly; a model with $\text{PPL} = 1.00$ predicts the next word with complete certainty.
4. **Decoding Trade-offs**: Greedy search is fast but prone to repetitive loops; temperature scaling modulates distribution entropy; Top-K guarantees tail truncation; and Top-P (Nucleus) sampling dynamically expands or contracts candidate sets based on model confidence.

## Next steps

1. Build `12-attention` introducing Bahdanau additive and Luong multiplicative attention to overcome recurrent memory bottlenecks.
2. Build `13-mini-transformer` replacing recurrence entirely with Multi-Head Self-Attention and causal triangular masking.
3. Build `14-mini-llm` training a full decoder-only language model on multi-sentence text corpora.
