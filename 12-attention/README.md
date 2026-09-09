# 12 — Attention Mechanism From Scratch

## Goal

Understand the **Attention Mechanism** and **Multi-Head Attention (MHA)** from first principles with pure NumPy, and master the foundational mathematical engine that powers modern Transformers, Large Language Models (LLMs), Vision Transformers (ViTs), and multimodal AI systems.

While a Recurrent Neural Network (RNN, `07-rnn-from-scratch`, `09-lstm`, `10-gru`) compresses an entire sequence into a single recurrent hidden state vector `h_t` via sequential recurrence:

```math
h_t = f(x_t, h_{t-1})
```

the **Attention Mechanism** (Vaswani et al., 2017: *"Attention Is All You Need"*) completely discards recurrence in favor of **direct, pairwise, parallelized interaction** between all tokens in a sequence:

```math
\mathrm{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

```text
====================================================================================================
THE PARADIGM SHIFT: RECURRENT COMPRESSION VS. ATTENTION HIGHWAYS
====================================================================================================

1. RECURRENT NEURAL NETWORK (Sequential Bottleneck & Information Loss):
   x₁ ──► [Cell] ──► h₁ ──► [Cell] ──► h₂ ──► [Cell] ──► h₃ ──► [Cell] ──► h₄ (Bottleneck!)
           ▲                 ▲                 ▲                 ▲
           x₁                x₂                x₃                x₄
   * Path length between x₁ and x₄ is O(T) sequential steps!
   * Early context is compressed, degraded, and overwritten.

2. SELF-ATTENTION MECHANISM (Direct O(1) Path-Length Shortcuts):
   x₁ ───────────┬──────────────┬──────────────┬──────────────► Context Vector c₁
   x₂ ───────────┼──────────────┼──────────────┼──────────────► Context Vector c₂
   x₃ ───────────┼──────────────┼──────────────┼──────────────► Context Vector c₃
   x₄ ───────────┴──────────────┴──────────────┴──────────────► Context Vector c₄
         All tokens can attend directly to all other tokens, giving O(1) maximum path length.
====================================================================================================
```

---

## The Core Mathematical Equations of Attention

Given an input matrix `X \in \mathbb{R}^{T \times d_{\mathrm{model}}}` representing a sequence of `T` tokens with feature dimension `d_{\mathrm{model}}`:

### 1. Linear Projections into Query, Key, and Value Subspaces
The input representations are projected into three distinct learned functional spaces:

```math
Q = X W^Q \in \mathbb{R}^{T \times d_k} \quad (\text{Queries: What each token is looking for})
```

```math
K = X W^K \in \mathbb{R}^{T \times d_k} \quad (\text{Keys: What each token offers as index})
```

```math
V = X W^V \in \mathbb{R}^{T \times d_v} \quad (\text{Values: The substantive content to be retrieved})
```

Where:
- `W^Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}`: Query projection weight matrix.
- `W^K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}`: Key projection weight matrix.
- `W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}`: Value projection weight matrix.
- `d_k`: Query/Key dimension per head.
- `d_v`: Value dimension per head.
- In this implementation, `d_k = d_v = d_{\mathrm{model}} / h` (where `h` is the number of attention heads).

---

### 2. Scaled Dot-Product Similarity Scores
Every Query vector computes a dot product with every Key vector to measure relevance:

```math
S = Q K^T \in \mathbb{R}^{T \times T}
```

To prevent gradient vanishing in the Softmax function when `d_k` is large, the raw scores are scaled by `\frac{1}{\sqrt{d_k}}`:

```math
S_{\mathrm{scaled}} = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
```

---

### 3. Normalized Attention Weights (Softmax)
The scaled scores are normalized row-by-row into non-negative probabilities summing to `1.0`:

```math
A = \operatorname{softmax}\left(S_{\mathrm{scaled}}\right) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in (0, 1)^{T \times T}
```

For each query token `i`:

```math
A_{i, j} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^{T} \exp\left(\frac{q_i \cdot k_m}{\sqrt{d_k}}\right)}, \quad \sum_{j=1}^{T} A_{i, j} = 1.0
```

Where `A_{i, j}` represents the exact **attention weight** (fraction of focus) that token `i` allocates to token `j`.

---

### 4. Context Output Representation
The final output for each token is a convex linear combination of all Value vectors weighted by the attention distribution:

```math
\mathrm{Attention}(Q, K, V) = A V \in \mathbb{R}^{T \times d_v}
```

For token `i`:

```math
\mathrm{context}_i = \sum_{j=1}^{T} A_{i, j} v_j \in \mathbb{R}^{1 \times d_v}
```

---

### 5. Multi-Head Attention (MHA)
Instead of performing a single attention function, Multi-Head Attention linearly projects Queries, Keys, and Values `h` times with distinct, learnable projection matrices:

```math
\mathrm{MultiHead}(Q, K, V) = \operatorname{Concat}(\mathrm{head}_1, \mathrm{head}_2, \dots, \mathrm{head}_h) W^O
```

Where each individual head computes:

```math
\mathrm{head}_i = \mathrm{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right) \in \mathbb{R}^{T \times d_v}
```

And:
- `W_i^Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}`
- `W_i^K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}`
- `W_i^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}`
- `W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\mathrm{model}}}`: Output projection mixing representations across all heads.

---

## Why Did We Need Attention? The Collapse of Recurrence

Before the Transformer architecture, state-of-the-art Natural Language Processing relied on Recurrent Neural Networks (RNNs, LSTMs, and GRUs). While capable of handling sequential data, recurrent models suffered from **two fatal architectural flaws**:

```text
====================================================================================================
THE TWO FATAL FLAWS OF RECURRENT MODELS
====================================================================================================

FLAW 1: THE SEQUENTIAL COMPUTATION BOTTLENECK (Limited Time-Parallelization)
   Timestep 1 ──► Timestep 2 ──► Timestep 3 ──► ... ──► Timestep 1000
   To compute h₁₀₀₀, the computer MUST sequentially execute steps 1 through 999.
   * GPU parallelism is limited across timesteps because each timestep depends on the previous hidden state.
   * The recurrent dependency creates O(T) sequential steps, preventing full parallelization across time (with computational complexity around O(T · d²)).

FLAW 2: THE FIXED-DIMENSIONAL INFORMATION BOTTLENECK
   "A 1000-word contract" ──────────► Compressed into ──► Vector h₁₀₀₀ (e.g., 512 numbers)
   Compressing long sequences into a fixed-dimensional recurrent state can make it
   difficult to preserve and access fine-grained information from early positions.
====================================================================================================
```

### 1. The Sequential Bottleneck: Why GPUs Hate RNNs
Modern hardware accelerators (NVIDIA GPUs, Google TPUs) achieve extreme floating-point throughput by executing tens of thousands of matrix operations in **parallel**.
- In an RNN, step `t` strictly requires the output `h_{t-1}` from the preceding step.
- As sequence length `T` grows from `50` to `4,096`, the sequential execution graph grows linearly `O(T)`.
- **Self-attention eliminates recurrence**: all `T` token representations can be projected and their pairwise similarities computed in parallel, with `O(1)` sequential depth. The actual computational work remains approximately `O(T^2 \cdot d)`.

---

### 2. The Information Bottleneck & Path Length

Consider a sequence of length `T`:

```text
Token 1 ("The") ................................................. Token T ("was")
```

- In an RNN, information from Token 1 must travel through `T - 1` non-linear recurrent matrix multiplications (`W_{hh}`) to reach Token `T`.
- The maximum path length that a signal must traverse between any two tokens is:
  ```math
  \text{Path Length}_{\mathrm{RNN}} = O(T)
  ```
- Even with LSTM's Constant Error Carousel or GRU's state interpolation, gradients and semantic signals degrade over dozens of sequential transitions.

- In Self-Attention, **every token directly attends to every other token**:
  ```math
  \text{Path Length}_{\mathrm{Attention}} = O(1)
  ```
  Token 1 directly computes a dot-product with Token `T` in a single step! There is no additional sequential path-length growth with token distance: the maximum attention path length is `O(1)`.

---

### Table 1 — Theoretical Comparison: Recurrent vs. Convolutional vs. Self-Attention
*(Reproduced from Vaswani et al., 2017)*

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Self-Attention** | `O(T^2 \cdot d)` | **`O(1)`** (Embarrassingly Parallel) | **`O(1)`** (Direct Shortcut) | `O(T^2 + T \cdot d)` |
| **Recurrent (RNN/LSTM)** | `O(T \cdot d^2)` | **`O(T)`** (Strictly Sequential) | **`O(T)`** (Information Decay) | `O(T \cdot d)` |
| **Convolutional (1D CNN)** | `O(k \cdot T \cdot d^2)` | **`O(1)`** (Parallel receptive fields) | **`O(\log_k(T))`** (Tree height) | `O(T \cdot d)` |

> [!IMPORTANT]
> Computational complexity is fundamentally distinct from sequential dependency depth. When `T` is not too large relative to `d` (e.g., standard sentence lengths where `T \cdot d < d^2`), self-attention can be competitive with or more efficient than recurrent layers, while offering much greater parallelism due to `O(1)` sequential operations.

---

## Conceptual Analogy: The Database Information Retrieval Intuition

The terms **Query (Q)**, **Key (K)**, and **Value (V)** originate from classical database search and information retrieval:

```text
====================================================================================================
DATABASE INFORMATION RETRIEVAL VS. SOFT ATTENTION
====================================================================================================

CLASSICAL HARD DATABASE LOOKUP (Python Dict / SQL):
   Database Keys:    { "movie_1": "Titanic",   "movie_2": "Inception",   "movie_3": "The Matrix" }
   Search Query:     "movie_2"
   Exact Match:      Key "movie_2" matches 100% (Binary: 1 or 0)
   Retrieved Value:  "Inception"

NEURAL SOFT ATTENTION LOOKUP (Continuous, Differentiable):
   Database Keys:    [ Vector k₁,  Vector k₂,  Vector k₃ ]  (Semantic tags of tokens)
   Database Values:  [ Vector v₁,  Vector v₂,  Vector v₃ ]  (Content carried by tokens)
   Input Query:      Vector q₁                              (What token 1 is searching for)
   
   Relevance Scores: Dot products:  q₁ · k₁ = 0.1,   q₁ · k₂ = 4.2,   q₁ · k₃ = 0.8
   Softmax Weights:  Probabilities: A₁,₁ = 0.02,     A₁,₂ = 0.94,     A₁,₃ = 0.04
   
   Retrieved Output: Context = 0.02 · v₁  +  0.94 · v₂  +  0.04 · v₃
                     (A soft, differentiable blend dominated by the most relevant Value!)
====================================================================================================
```

### Why Separate Projections for Q, K, and V?
If we directly computed `\operatorname{softmax}(X X^T) X`:
- Using `X` directly for `Q`, `K`, and `V` couples the search representation, matching representation, and retrieved content.
- It can also produce strong self-similarity because `x_i \cdot x_i = \|x_i\|^2`, which can bias attention toward the diagonal.
- By introducing separate learned weight matrices `W^Q`, `W^K`, and `W^V`:
  1. A token can **search** for one concept (via `Q`).
  2. It can **advertise** another concept to other tokens (via `K`).
  3. It can **transmit** a third distinct semantic content (via `V`).

---

## The Mathematical Proof: Why Scale by 1 / sqrt(d_k)?

A defining hallmark of the Vaswani et al. paper is the **scaling factor** `\frac{1}{\sqrt{d_k}}`. Why did Vaswani et al. introduce the scaling factor `\frac{1}{\sqrt{d_k}}`?

```math
S = Q K^T \quad \text{vs.} \quad S_{\mathrm{scaled}} = \frac{Q K^T}{\sqrt{d_k}}
```

---

### The Mathematical Variance Proof:

Let `q = (q_1, q_2, \dots, q_{d_k})` and `k = (k_1, k_2, \dots, k_{d_k})` be independent random feature vectors in `\mathbb{R}^{d_k}`.
Assume each component has zero mean and unit variance:

```math
\mathbb{E}[q_i] = 0, \quad \operatorname{Var}(q_i) = 1
```

```math
\mathbb{E}[k_i] = 0, \quad \operatorname{Var}(k_i) = 1
```

The dot product is the sum of `d_k` independent random variables:

```math
S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i
```

#### 1. Expectation of the Dot Product:
By independence:

```math
\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0
```

#### 2. Variance of the Dot Product:
Using the product variance identity for independent zero-mean variables `\operatorname{Var}(X Y) = \operatorname{Var}(X) \operatorname{Var}(Y)`:

```math
\operatorname{Var}(q_i k_i) = \operatorname{Var}(q_i) \cdot \operatorname{Var}(k_i) = 1 \cdot 1 = 1
```

Summing over all `d_k` dimensions:

```math
\operatorname{Var}(S) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k
```

Therefore, the **standard deviation** of the unscaled dot product is:

```math
\operatorname{std}(S) = \sqrt{\operatorname{Var}(S)} = \sqrt{d_k}
```

> [!NOTE]
> This derivation represents an idealized statistical argument under standard independent zero-mean and unit-variance initialization assumptions (e.g., standard normal or Xavier initialization). It provides the theoretical rationale for introducing the scaling factor at initialization, rather than guaranteeing that trained Query and Key vectors maintain these exact distributions throughout optimization.

---

### Softmax Saturation from Large Dot-Product Variance:

As representation dimension `d_k` grows large (e.g., `d_k = 64` or `d_k = 128`):
- The variance of the dot products grows proportionally to `d_k`, meaning standard deviations scale as `\sqrt{64} = 8` or `\sqrt{128} \approx 11.3`.
- When extremely large numbers enter the Softmax function `\operatorname{softmax}(z)`:
  ```math
  \operatorname{softmax}([+12.0, -11.0, -9.0]) \approx [0.99999, 0.00000, 0.00000]
  ```
- The distribution collapses toward a saturated one-hot distribution.
- **The Jacobian of Softmax** is:
  ```math
  \frac{\partial \operatorname{softmax}(z)_i}{\partial z_j} = \hat{y}_i (\delta_{ij} - \hat{y}_j)
  ```
  For the diagonal terms where `i = j`, this reduces to `\hat{y}_i (1 - \hat{y}_i)`. When probabilities saturate (`\hat{y}_i \approx 1.0` or `\hat{y}_i \approx 0.0`), both diagonal and off-diagonal Jacobian terms approach zero.
- When the logits become highly separated, Softmax can saturate and the resulting gradients can become very small, making optimization more difficult.

#### The Scaling Solution:
Dividing by `\sqrt{d_k}` scales the variance back to unit variance under the stated initialization assumptions:

```math
\operatorname{Var}\left(\frac{S}{\sqrt{d_k}}\right) = \frac{1}{d_k} \operatorname{Var}(S) = \frac{d_k}{d_k} = 1.0
```

This scaling keeps the logits in a more favorable range under the stated variance assumptions, reducing the risk of early Softmax saturation.

---

## Masked Self-Attention: Preserving Autoregressive Causality

In auto-regressive generation (such as GPT-style decoders or `11-sequence-prediction`), when predicting token `w_{t+1}`, the model **must not be allowed to look into future tokens** `w_{t+2}, \dots, w_T`.

```text
====================================================================================================
CAUSAL ATTENTION MASKING (Preventing Cheating / Looking into the Future)
====================================================================================================

Unmasked Attention (Bidirectional / Encoder):
   "YOUR"   can look at:  "YOUR", "CAT", "IS", "LOVELY"  (All tokens)
   "CAT"    can look at:  "YOUR", "CAT", "IS", "LOVELY"  (All tokens)

Masked Attention (Causal / Decoder):
   "YOUR"   can ONLY look at:  "YOUR"
   "CAT"    can ONLY look at:  "YOUR", "CAT"
   "IS"     can ONLY look at:  "YOUR", "CAT", "IS"
   "LOVELY" can ONLY look at:  "YOUR", "CAT", "IS", "LOVELY"
====================================================================================================
```

### Mathematical Formulation of Causal Masking:

A causal mask `M \in \{0, -\infty\}^{T \times T}` is an upper-triangular matrix:

```math
M_{i, j} = \begin{cases} 0 & \text{if } j \le i \quad (\text{Past or Present}) \\ -\infty & \text{if } j > i \quad (\text{Future - Forbidden!}) \end{cases}
```

```math
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
```

The mask is added to the scaled scores **before** applying Softmax:

```math
A_{\mathrm{causal}} = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right)
```

Because `\lim_{x \to -\infty} \exp(x) = 0`, all future positions become **mathematically zero** in the attention matrix when using `-np.inf`:

```math
A_{\mathrm{causal}} = \begin{bmatrix}
1.00 & 0.00 & 0.00 & 0.00 \\
0.42 & 0.58 & 0.00 & 0.00 \\
0.18 & 0.22 & 0.60 & 0.00 \\
0.12 & 0.15 & 0.33 & 0.40
\end{bmatrix}
```

> [!NOTE]
> In our pure NumPy implementation using literal `-np.inf`, masked entries evaluate to exact mathematical zeros (`exp(-inf) == 0.0`). In deep learning frameworks and mixed-precision routines (e.g., FP16), large finite negative constants such as `-1e9` or `-1e4` are commonly substituted to prevent NaN artifacts, yielding numerically negligible weights rather than exact zeros.

---

## Multi-Head Attention: Subspace Diversity

### Why use multiple attention heads?

A single head has one learned attention pattern per query position. Multiple heads allow the model to represent different relationships and feature subspaces in parallel.

```text
====================================================================================================
WHY MULTI-HEAD ATTENTION? (Joint Subspace Focus)
====================================================================================================
Illustrative example: different heads may learn to emphasize different relationships;
specific specialization is learned and is not guaranteed to follow these categories.

Consider the sentence: "The animal didn't cross the street because it was too tired."

Head 1 (Grammar / Syntax):        "it" ──────── attend ────────► "animal" (Resolves coreference pronoun)
Head 2 (State / Adjective):       "it" ──────── attend ────────► "tired"  (Connects subject to condition)
Head 3 (Spatial / Relationship):  "cross" ───── attend ────────► "street" (Connects action to target)

* A single attention head produces a single attention distribution per position.
* Multi-Head Attention allows the model to attend to information from different representation subspaces simultaneously.
====================================================================================================
```

### Tensor Reshaping Magic (Vectorized NumPy Execution)

Rather than running `h` separate attention loops, high-performance implementations project the input once into `d_{\mathrm{model}}` and **reshape into parallel head tensors**:

```math
X \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
```

1. Project with unified weight matrices `W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}`:
   ```math
   Q_{\mathrm{all}} = X W^Q \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
   ```
2. Reshape and transpose into multi-head tensor format:
   ```python
   # Split d_model into (h, d_k)
   # Shape: (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
   Q = Q_all.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
   K = K_all.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
   V = V_all.reshape(B, T, h, d_v).transpose(0, 2, 1, 3)
   ```
3. Batched matrix multiplication over heads in parallel:
   ```python
   # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
   scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
   weights = softmax(scores, axis=-1)
   # (B, h, T, T) @ (B, h, T, d_v) -> (B, h, T, d_v)
   head_outputs = weights @ V
   ```
4. Recombine and project:
   ```python
   # (B, h, T, d_v) -> (B, T, h, d_v) -> (B, T, d_model)
   out = head_outputs.transpose(0, 2, 1, 3).reshape(B, T, d_model)
   final_output = out @ W_O
   ```

---

## The Positional Problem & Positional Encoding

### Pure Self-Attention is Permutation Equivariant (Order-Agnostic)

Without positional information, self-attention does not inherently encode the order of tokens; permuting the input sequence correspondingly permutes the output representations.

Consider two sequences with identical words in completely different orders:
```text
Sequence A: "The dog bit the man"
Sequence B: "The man bit the dog"
```

In pure self-attention without positional encodings:
- The dot product between word vectors `x_i \cdot x_j` depends **only on their feature values**, not their sequence position index.
- Permuting the input rows permutes the output rows identically (permutation equivariance):
  ```math
  \mathrm{Attention}(P X) = P \mathrm{Attention}(X)
  ```
- **The model has zero knowledge of sequential token order!**

---

### The Sinusoidal Positional Encoding Solution (Vaswani et al., 2017)

To give the model awareness of token order without adding recurrent steps, a fixed **Positional Encoding matrix (`PE`)** is added directly to the input embeddings:

```math
X_{\mathrm{encoder\_input}} = X_{\mathrm{embedding}} + PE
```

Where the `(pos, 2i)` and `(pos, 2i+1)` elements of `PE` are computed using sinusoidal waves of varying frequencies:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\mathrm{model}}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\mathrm{model}}}}\right)
```

Where:
- `pos \in \{0, 1, \dots, T-1\}`: The temporal position of the token in the sequence.
- `i \in \{0, 1, \dots, d_{\mathrm{model}}/2 - 1\}`: The feature dimension index.
- `10000^{2i / d_{\mathrm{model}}}`: The wave period scaling across dimensions (from `2\pi` to `10000 \cdot 2\pi`).

#### Why Sinusoids? The Linear Relative Translation Property:
By trigonometric angle-addition identities:
```math
\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)
```
```math
\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)
```
For any fixed temporal distance `k`, the positional encoding at `pos + k` can be expressed as a **pure linear transformation** of the positional encoding at `pos`:
```math
PE_{(pos + k, :)} = PE_{(pos, :)} M_k
```
This structure provides a mechanism for representing relative positional offsets and can facilitate extrapolation to sequence positions beyond those seen during training.

---

## The Three Flavors of Attention in the Transformer Architecture

```text
====================================================================================================
THE 3 ATTENTION FLAVORS IN THE TRANSFORMER (Vaswani et al. Architecture)
====================================================================================================

1. ENCODER SELF-ATTENTION:
   * Query source: Encoder Layer Input
   * Key source:   Encoder Layer Input
   * Value source: Encoder Layer Input
   * Mask:         None (Bidirectional: every word sees every other word)

2. MASKED DECODER SELF-ATTENTION:
   * Query source: Decoder Layer Input
   * Key source:   Decoder Layer Input
   * Value source: Decoder Layer Input
   * Mask:         Causal Lower-Triangular Mask (Prevents attending to future words)

3. CROSS-ATTENTION (ENCODER-DECODER ATTENTION):
   * Query source: Previous Decoder Layer (What target word needs to be translated)
   * Key source:   Final Encoder Output   (Source language context tags)
   * Value source: Final Encoder Output   (Source language content)
   * Mask:         None / Source Padding Mask
====================================================================================================
```

---

## Parameters and Tensor Shapes Reference

Let:
- `B`: Batch size.
- `T`: Sequence length.
- `d_{\mathrm{model}}`: Total model embedding dimension.
- `h`: Number of parallel attention heads.
- `d_k`: Query/Key dimension per head.
- `d_v`: Value dimension per head.
- In this implementation, `d_k = d_v = d_{\mathrm{model}} / h`.

| Tensor / Parameter | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `X` | `(B, T, d_{\mathrm{model}})` | `X \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}` | Input sequence batch |
| `W^Q` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `W^Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}` | Unified query projection weight matrix |
| `W^K` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `W^K \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}` | Unified key projection weight matrix |
| `W^V` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}` | Unified value projection weight matrix |
| `W^O` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `W^O \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}` | Output multi-head projection weight matrix |
| `Q` | `(B, h, T, d_k)` | `Q \in \mathbb{R}^{B \times h \times T \times d_k}` | Multi-head query tensor |
| `K` | `(B, h, T, d_k)` | `K \in \mathbb{R}^{B \times h \times T \times d_k}` | Multi-head key tensor |
| `V` | `(B, h, T, d_v)` | `V \in \mathbb{R}^{B \times h \times T \times d_v}` | Multi-head value tensor |
| `scores` | `(B, h, T, T)` | `S \in \mathbb{R}^{B \times h \times T \times T}` | Scaled pairwise similarity scores `Q K^T / \sqrt{d_k}` |
| `attn_weights` | `(B, h, T, T)` | `A \in (0, 1)^{B \times h \times T \times T}` | Softmax attention distribution summing to 1 across rows |
| `context` | `(B, T, d_{\mathrm{model}})` | `C \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}` | Multi-head attention output after `W^O` projection |

---

### Concrete Parameter Budget for Educational Setup (`d_{\mathrm{model}} = 4, h = 2, d_k = 2`)

```text
W^Q : (4, 4) -> 16 weights
W^K : (4, 4) -> 16 weights
W^V : (4, 4) -> 16 weights
W^O : (4, 4) -> 16 weights
------------------------------------
Total Attention Parameters = 64 weights
```

---

## Project Structure

```text
12-attention/
├── README.md                      # Foundational theory, MHA mechanics, and 101-topic curriculum
├── src/
│   └── attention.py               # Pure NumPy Scaled Dot-Product Attention & MultiHeadAttention
└── steps/
    └── step-01-mathematics.md      # Matrix calculus derivations, Jacobian proofs, and manual trace
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Pending | `steps/step-01-mathematics.md`: Dot-product variance proof, Softmax gradient, MHA calculus |
| **2. NumPy Implementation** | Pending | `src/attention.py`: Vectorized `scaled_dot_product_attention`, `MultiHeadAttention` |
| **3. Understand Forward Pass** | Pending | Tracing Queries, Keys, Values, attention maps, and weighted Value combinations |
| **4. Understand Loss & Masking** | Pending | Causal lower-triangular masking (`-\infty`), padding mask, and sequence cross-entropy |
| **5. Derive Gradients** | Pending | Matrix calculus for `dQ, dK, dV`, Softmax Jacobian product, and projection gradients |
| **6. Implement Backpropagation** | Pending | Reverse-mode automatic differentiation in NumPy verified against finite differences |
| **7. Train Model** | Pending | Learning token-to-token associative retrieval and syntax binding |
| **8. Debug & Analyze** | Pending | Visualizing attention heatmaps, diagonal dominance, and multi-head specialization |
| **9. PyTorch Implementation** | Pending | Parity validation against `torch.nn.MultiheadAttention` down to 7 decimal places |
| **10. Compare Results** | Pending | Runtime benchmark, memory efficiency, and numerical agreement validation |

---

## Complete 101-Topic Foundational Curriculum

This project systematically covers the following 101 core deep learning topics across 16 structured modules:

```text
THE ATTENTION REVOLUTION
│
├── 01. Why attention exists: The death of sequential recurrence
├── 02. The information bottleneck problem in RNNs/LSTMs
├── 03. O(T) sequential dependency depth vs O(1) parallel depth
├── 04. Maximum path length: O(T) vs O(1) direct shortcuts
├── 05. The Vaswani et al. (2017) breakthrough
├── 06. Permutation equivariance of pure self-attention (order-agnostic)
│
▼
THE QUERY, KEY, VALUE PARADIGM
│
├── 07. Database information retrieval analogy (Dict search)
├── 08. What is a Query vector? (The search probe)
├── 09. What is a Key vector? (The index tag)
├── 10. What is a Value vector? (The content payload)
├── 11. Why separate Q, K, V projections instead of raw X?
├── 12. Linear projection matrices (W_Q, W_K, W_V)
├── 13. Query dimension (d_k) vs Value dimension (d_v)
│
▼
SCALED DOT-PRODUCT ATTENTION
│
├── 14. Dot product as directional similarity metric
├── 15. Vector dot product: q · k = sum(q_i * k_i)
├── 16. Matrix multiplication Q @ K.T -> (T, T) similarity matrix
├── 17. Variance of dot product: Var(q · k) = d_k
├── 18. Why unscaled dot products explode with high dimensions
├── 19. Softmax gradient saturation risk from large dot-product variance
├── 20. The scaling factor: 1 / sqrt(d_k)
├── 21. Preserving unit variance: Var(S / sqrt(d_k)) = 1
│
▼
THE ATTENTION MATRIX & SOFTMAX
│
├── 22. Row-wise Softmax normalization
├── 23. Attention weights: A_ij = P(token_j | token_i)
├── 24. Numerical stability trick: Subtracting row-max
├── 25. Self-attention diagonal: Why tokens attend to themselves
├── 26. Off-diagonal elements: Syntactic and semantic dependencies
├── 27. Weighted sum of Values: A @ V -> Context Matrix
├── 28. Interpreting attention as a dynamic routing filter
│
▼
CAUSAL & PADDING MASKING
│
├── 29. Autoregressive language modeling requirement (Causality)
├── 30. The Upper-Triangular Causal Mask
├── 31. Replacing future values with -inf
├── 32. Why exp(-inf) = 0 in Softmax
├── 33. Zeroing out future token attention weights
├── 34. Padding mask: Ignoring <PAD> tokens in variable-length batches
├── 35. Combining Causal Mask and Padding Mask
│
▼
MULTI-HEAD ATTENTION (MHA)
│
├── 36. Motivation: Why multiple attention heads are useful
├── 37. Representational subspaces (Syntax, Coreference, Logic)
├── 38. Splitting d_model into h heads: d_k = d_model / h
├── 39. Parallel multi-head computation
├── 40. Tensor transpositions: (B, T, d_model) -> (B, h, T, d_k)
├── 41. Batched matrix multiplication over heads
├── 42. Concatenating head outputs: (B, T, h * d_v)
├── 43. Output projection matrix W_O
├── 44. Mixing information across heads
│
▼
POSITIONAL ENCODING
│
├── 45. Why Transformers cannot understand word order natively
├── 46. Adding position vectors to word embeddings
├── 47. Sinusoidal positional encodings (Vaswani et al.)
├── 48. Sine for even dimensions, Cosine for odd dimensions
├── 49. Wavelength progression: 2pi to 10000 * 2pi
├── 50. Linear relative shift property: PE_{pos+k} = PE_pos * M_k
├── 51. Learned positional embeddings (BERT / GPT style)
├── 52. Rotary Position Embeddings (RoPE) preview
│
▼
ATTENTION TOPOLOGIES
│
├── 53. Self-Attention (Q, K, V from same source)
├── 54. Cross-Attention (Q from decoder, K & V from encoder)
├── 55. Encoder Self-Attention (Bidirectional)
├── 56. Decoder Masked Self-Attention (Causal)
├── 57. Bidirectional vs Autoregressive attention
│
▼
MANUAL NUMERICAL FORWARD TRACE
│
├── 58. Setup: B=1, T=3, d_model=4, h=2, d_k=2
├── 59. Input tokens & embedding matrix
├── 60. Compute Q, K, V projections by hand
├── 61. Split into Head 1 and Head 2
├── 62. Calculate raw dot-product similarity matrices
├── 63. Divide by sqrt(d_k) = sqrt(2)
├── 64. Apply row-wise Softmax
├── 65. Multiply attention weights by Value vectors
├── 66. Concatenate Head 1 and Head 2
├── 67. Multiply by W_O for final attention output
│
▼
BACKPROPAGATION CALCULUS FOR ATTENTION
│
├── 68. Upstream gradient dL/d(Context)
├── 69. Gradient of Output Projection: dW_O and d(Concat)
├── 70. Splitting gradients back into individual heads
├── 71. Gradient of A @ V: dV = A.T @ d(Head)
├── 72. Gradient of A @ V: dA = d(Head) @ V.T
├── 73. Matrix calculus of Softmax backward pass
├── 74. Softmax Jacobian product: dS = A * (dA - sum(dA * A, axis=-1))
├── 75. Gradient of scaling factor: dS_raw = dS / sqrt(d_k)
├── 76. Gradient of Q @ K.T: dQ = dS_raw @ K
├── 77. Gradient of Q @ K.T: dK = dS_raw.T @ Q
├── 78. Accumulating parameter gradients: dW_Q, dW_K, dW_V
├── 79. Input gradient: dX = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T
│
▼
NUMERICAL GRADIENT VERIFICATION
│
├── 80. Finite-difference gradient checking script
├── 81. Verifying dW_Q, dW_K, dW_V, dW_O (< 1e-7 relative error)
├── 82. Verifying input gradient dX
├── 83. Causal mask gradient routing (Zero gradient to masked entries)
│
▼
ATTENTION HEATMAP VISUALIZATION
│
├── 84. Plotting (T, T) attention matrices with Matplotlib / Seaborn
├── 85. Identifying diagonal self-attention dominance
├── 86. Visualizing syntactic dependencies (Adjective -> Noun)
├── 87. Analyzing cross-head diversity
├── 88. Entropy of attention distributions: Sharp vs Diffuse
│
▼
NUMPY OPTIMIZATIONS
│
├── 89. Fused QKV linear projection: W_qkv in R^(d_model x 3*d_model)
├── 90. Einsum notation for multi-head attention: np.einsum('bhtd,bhkd->bhtk')
├── 91. In-place mask operations
├── 92. Memory-efficient tensor restriding
│
▼
ADVANCED ATTENTION VARIANTS
│
├── 93. Scaled Dot-Product vs Additive (Bahdanau) Attention
├── 94. Multi-Query Attention (MQA: Shared K, V across heads)
├── 95. Grouped-Query Attention (GQA: LLaMA 2 / 3 standard)
├── 96. FlashAttention overview: IO-aware tiling & online Softmax
├── 97. Linear Attention & Kernelized Approximations
│
▼
PYTORCH PARITY & VALIDATION
│
├── 98. torch.nn.MultiheadAttention implementation
├── 99. Exact numerical agreement down to 7 decimal places
├── 100. Benchmarking throughput: NumPy vs PyTorch CUDA
└── 101. Final reflections: Why Attention conquered Modern AI
```

---

## Conclusion & Architectural Next Steps

Mastering the Attention Mechanism is the definitive bridge from the **Sequential Recurrent Era** (`07-rnn`, `08-sentiment-rnn`, `09-lstm`, `10-gru`, `11-sequence-prediction`) to the modern **Transformer & Generative AI Era** (`13-mini-transformer`, `14-mini-llm`).

The next immediate step is deriving and verifying every equation, Jacobian matrix, and finite-difference check in [steps/step-01-mathematics.md](file:///Users/pasan/Documents/Personal/deep-learning-foundations/12-attention/steps/step-01-mathematics.md), followed by the pure NumPy implementation in [src/attention.py](file:///Users/pasan/Documents/Personal/deep-learning-foundations/12-attention/src/attention.py).
