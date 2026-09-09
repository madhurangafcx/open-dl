# Step 1 — Mathematics for Sentiment Analysis with Recurrent Neural Networks (RNN)

Goal: understand the complete mathematics of Natural Language Processing (NLP) and sequence classification using a **Many-to-One Recurrent Neural Network** from first principles.
This document provides the rigorous theoretical foundations, word embedding geometry, sequence padding dynamics, tensor dimensions, forward calculations, and Backpropagation Through Time (BPTT) matrix calculus that will be implemented with pure NumPy from scratch.

---

## Q1 — The Sentiment Analysis Problem: Why Feedforward ANNs & Bag-of-Words Fail

Sentiment analysis aims to determine the emotional valence or polarity (positive vs. negative) of human natural language.
In traditional machine learning (and the Multi-Layer Perceptrons in `02-ann-from-scratch`), text is converted into fixed-length vectors using **Bag-of-Words (BoW)** or **Term Frequency-Inverse Document Frequency (TF-IDF)**:

```math
x_{\mathrm{bow}} = \begin{bmatrix} c_0 & c_1 & \dots & c_{V-1} \end{bmatrix} \in \mathbb{R}^{1 \times V}
```

Where `c_j` is the count or frequency of word `j` in the document, and `V` is the dictionary size.

While simple and computationally cheap, Bag-of-Words fundamentally breaks down on human language for three mathematical reasons:

### 1. Temporal Amnesia & Word Order Inversion
Bag-of-Words is commutative: it sums word counts without regard to position.
Consider these two opposing reviews:

```text
Review A: "The film was not good, it was bad."  ──► True Polarity: NEGATIVE (0)
Review B: "The film was not bad, it was good."  ──► True Polarity: POSITIVE (1)
```

Both reviews contain the exact same multiset of words:
`{'the': 1, 'film': 1, 'was': 2, 'not': 1, 'good': 1, 'it': 1, 'bad': 1}`.

Under Bag-of-Words:

```math
x_{\mathrm{bow}}(\text{Review A}) \equiv x_{\mathrm{bow}}(\text{Review B})
```

Because their input vectors are identical, any feedforward ANN or linear classifier is mathematically forced to produce the **exact same output probability**:

```math
P(\text{Positive} \mid \text{Review A}) = P(\text{Positive} \mid \text{Review B})
```

The model is incapable of distinguishing between opposite sentiments!

### 2. Negation Flipping
A single negation word (`"not"`, `"never"`, `"hardly"`) reverses the semantic meaning of downstream words:

```text
"I enjoyed the performance."       ──► Positive (+)
"I never enjoyed the performance." ──► Negative (-)
```

In a feedforward BoW model, `"enjoyed"` and `"performance"` contribute massive positive weight to the logit sum, frequently overwhelming the single `"never"` feature and yielding a false positive.

### 3. Contrastive Conjunctions ("but", "however")
Reviews frequently open with concessive praise before delivering the concluding judgment:

```text
"The actors were gifted and the visuals were magnificent, BUT the plot was boring and ruined the movie."
```

In human cognition, the turning conjunction `"BUT"` invalidates the preceding clauses and places dominant emotional weight on the trailing clause. An RNN's sequential hidden state `h_t` reads from left to right, allowing the concluding negative tokens to overwrite earlier positive activations.

---

## Q2 — One-Hot Word Vectors vs. Dense Continuous Word Embeddings

To feed words into neural networks, strings must be mapped to numbers.

### The Three Fatal Flaws of One-Hot Word Vectors

If we assign each word a standard one-hot basis vector in `\mathbb{R}^V`:

```text
Vocabulary size V = 20,000 words

'great'   = [0, 0, 0, 1, 0, 0, ..., 0]  in R^(20000)
'awesome' = [0, 1, 0, 0, 0, 0, ..., 0]  in R^(20000)
'terrible'= [0, 0, 0, 0, 1, 0, ..., 0]  in R^(20000)
```

1. **Extreme Sparsity & Memory Exhaustion**:
   For a modest review of 30 words, one-hot encoding requires `30 \times 20,000 = 600,000` floats, of which 99.995% are zero.
2. **Orthogonality (Zero Semantic Relatedness)**:
   Any two distinct one-hot vectors are mutually orthogonal standard basis vectors:
   ```math
   x_{\text{great}} \cdot x_{\text{awesome}} = 0.0, \quad x_{\text{great}} \cdot x_{\text{terrible}} = 0.0
   ```
   The Euclidean distance between every single pair of words is identical:
   ```math
   \|x_i - x_j\|_2 = \sqrt{2} \quad \forall i \ne j
   ```
   The model cannot know that `"great"` is synonymous with `"awesome"`, nor that it is the antonym of `"terrible"`.
3. **Severe Parameter Explosion**:
   Connecting a 20,000-dimensional one-hot input to a hidden layer of 128 neurons requires `20,000 \times 128 = 2,560,000` weights, leading to rapid overfitting on small training datasets.

---

### The Solution: Dense Word Embeddings Layer

Instead of sparse high-dimensional vectors, each word is mapped to a continuous, dense low-dimensional vector:

```math
e_t = E[w_t] \in \mathbb{R}^{1 \times D} \quad (\text{e.g., } D = 4, 16, \text{ or } 64)
```

Where `E \in \mathbb{R}^{V \times D}` is the learnable **Embedding Matrix**.

```text
Embedding Matrix E: (V = 16, D = 4)

             Dim 0   Dim 1   Dim 2   Dim 3
Index 0 <PAD> [ 0.00,   0.00,   0.00,   0.00 ]  <── Zero vector for padding
Index 1 <UNK> [ 0.10,  -0.10,   0.20,  -0.20 ]  <── Out-of-vocabulary representation
Index 4 'great'[ 0.80,   0.70,  -0.10,   0.50 ]  <── Positive valence
Index 5 'loved'[ 0.75,   0.65,  -0.05,   0.45 ]  <── Proximity to 'great'
Index 6 'terrible'[-0.70,-0.60, 0.20,  -0.40 ]  <── Opposite direction!
Index 10'acting'[ 0.30, -0.20,  0.40,   0.10 ]
Index 14'not'  [-0.50,  -0.30,  0.10,  -0.60 ]  <── Negation modifier
Index 15'good' [ 0.60,   0.50, -0.20,   0.40 ]  <── Positive valence
```

### Mathematical Equivalence: Matrix Lookup vs. One-Hot Multiplication

The embedding retrieval can be viewed from two mathematically equivalent perspectives:

```math
e_t = E[w_t] \equiv \text{one\_hot}(w_t) \cdot E
```

- **Theoretical View**: Matrix multiplication between a sparse one-hot row vector `(1, V)` and the embedding table `(V, D)`.
- **Computational View**: Direct indexed row retrieval `E[w_t]` in `O(1)` time complexity, bypassing expensive matrix multiplications entirely.

---

## Q3 — Sequence Padding, Variable Lengths, & The Trailing Padding Corruption Problem

Natural language sentences vary dynamically in length:
- Sentence 1: `"Loved it!"` (Length = 2)
- Sentence 2: `"The movie was completely awful."` (Length = 5)

To construct batch tensors of shape `(B, T)` for vectorized NumPy computation, shorter sequences are padded with a dedicated `<PAD>` token (assigned index `0`) to match the maximum sequence length `T`:

```text
Raw Sequences:
A: [5, 12]              (Length 2)
B: [2, 3, 11, 13, 6]    (Length 5)

Padded Batch (T_max = 5, Post-Padding with <PAD> = 0):
A: [ 5, 12,  0,  0,  0 ]  <── 3 trailing pad tokens
B: [ 2,  3, 11, 13,  6 ]  <── Full sentence
```

---

### The Trailing Padding Corruption Proof

What happens if an RNN naively processes trailing `<PAD>` tokens (`e_t = \mathbf{0}`)?

Recall the standard RNN hidden state update:

```math
h_t = \tanh(e_t W_{xh} + h_{t-1} W_{hh} + b_h)
```

At a padding timestep where `e_t = \mathbf{0}`:

```math
h_{\mathrm{pad}} = \tanh(\mathbf{0} \cdot W_{xh} + h_{t-1} W_{hh} + b_h) = \tanh(h_{t-1} W_{hh} + b_h)
```

Notice that:

```math
h_{\mathrm{pad}} \ne h_{t-1}
```

Because of the non-zero bias vector `b_h` and recurrent weights `W_{hh}`, **the hidden state continues to mutate on empty padding tokens**!
If a sentence has length 2 and is padded to length 10, the meaningful semantic representation accumulated at `t = 2` will be transformed through 8 consecutive non-linear bias shifts, destroying the true sentence representation before reaching the classification head!

---

### Two Mathematical Solutions to Padding Corruption

#### Solution 1: Dynamic Sequence Length Indexing
Instead of taking the terminal hidden state at `t = T`, extract the hidden state at the sample's **exact actual sentence length**:

```math
h_{\mathrm{sentence}}^{(i)} = h_{L_i}^{(i)} \quad \text{where } L_i = \operatorname{len}(\text{Sentence } i)
```

In NumPy:
```python
# Extract the true final hidden state per sample in the batch
h_final = H[np.arange(batch_size), actual_lengths - 1]  # Shape: (B, H)
```

#### Solution 2: Hidden State Masking
Introduce a binary mask `m_t \in \{0, 1\}`:
- `m_t = 1` for real words (`token != <PAD>`)
- `m_t = 0` for padding tokens (`token == <PAD>`)

The hidden state update is modified into an explicit conditional pass-through:

```math
h_t = m_t \odot \tanh(e_t W_{xh} + h_{t-1} W_{hh} + b_h) + (1 - m_t) \odot h_{t-1}
```

When `m_t = 0`, `h_t = h_{t-1}`: the hidden state freezes completely and preserves historical context across all trailing padding steps!

---

## Q4 — Architecture & Sequence Topology: The Many-to-One Network

In Project 07 (`07-rnn-from-scratch`), we evaluated loss at every single timestep (Many-to-Many Synchronous).
In Sentiment Analysis, we map an entire sequential document into a single scalar classification prediction (**Many-to-One Topology**):

```text
                                  Predicted Probability y_hat
                                              ▲
                                              │
                                         [ Sigmoid ]
                                              ▲
                                              │
                                         Logit o
                                              ▲
                                              │
                                     [ Output Layer ]
                                        (W_hy, b_y)
                                              ▲
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │               Sentence Summary Vector h_T         │
                    └─────────────────────────┬─────────────────────────┘
                                              ▲
                                              │
         e_1                   e_2            │            e_T
          │                     │             │             │
          ▼                     ▼             │             ▼
       ┌────┐                ┌────┐        ┌────┐        ┌────┐
h_0 ──►│ H  │───────────────►│ H  │───────►│ H  │ ... ──►│ H  │
 = 0   └────┘                └────┘        └────┘        └────┘
          ▲                     ▲             ▲             ▲
          │                     │             │             │
       Word 1                Word 2        Word 3        Word T
```

---

## Q5 — Tensor Shapes and Parameter Budget

To make every calculation verifiable on paper, we specify a clean educational configuration:

### Problem Dimensions:
- Batch size: `B = 1`
- Vocabulary size: `V = 16`
- Sequence length: `T = 4` words (`['acting', 'was', 'not', 'good']`)
- Embedding dimension: `D = 4`
- Hidden recurrent units: `H = 3`
- Output dimension: `K = 1` (binary scalar logit for sentiment)

### Complete Tensor Dimension Table:

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `tokens` | `(1, 4)` | `w \in \mathbb{N}^{1 \times T}` | Sequence of integer word IDs |
| `E` | `(16, 4)` | `E \in \mathbb{R}^{V \times D}` | Word embedding matrix |
| `e_t` | `(1, 4)` | `e_t \in \mathbb{R}^{1 \times D}` | Embedded word vector at timestep `t` |
| `W_xh` | `(4, 3)` | `W_{xh} \in \mathbb{R}^{D \times H}` | Embedding-to-hidden weight matrix |
| `W_hh` | `(3, 3)` | `W_{hh} \in \mathbb{R}^{H \times H}` | Hidden-to-hidden recurrent weight matrix |
| `b_h` | `(1, 3)` | `b_h \in \mathbb{R}^{1 \times H}` | Hidden state bias offset |
| `h_t` | `(1, 3)` | `h_t \in \mathbb{R}^{1 \times H}` | Recurrent hidden state vector |
| `W_hy` | `(3, 1)` | `W_{hy} \in \mathbb{R}^{H \times 1}` | Hidden-to-sentiment projection matrix |
| `b_y` | `(1, 1)` | `b_y \in \mathbb{R}^{1 \times 1}` | Output classification bias |
| `o` | `(1, 1)` | `o \in \mathbb{R}` | Unnormalized scalar sentiment logit |
| `\hat{y}` | `(1, 1)` | `\hat{y} \in (0, 1)` | Predicted probability of positive sentiment |
| `y` | `(1, 1)` | `y \in \{0, 1\}` | Ground-truth binary sentiment label |

### Parameter Budget:

```math
N_{\mathrm{params}} = \underbrace{(16 \cdot 4)}_{E} + \underbrace{(4 \cdot 3)}_{W_{xh}} + \underbrace{(3 \cdot 3)}_{W_{hh}} + \underbrace{3}_{b_h} + \underbrace{(3 \cdot 1)}_{W_{hy}} + \underbrace{1}_{b_y} = 64 + 12 + 9 + 3 + 3 + 1 = 92 \text{ parameters}
```

A compact budget of **92 scalars** allows us to hand-calculate every forward transformation, logit projection, and BPTT matrix update down to 6 decimal places.

---

## Q6 — Concrete Numerical Setup (Matching verify_sentiment_rnn.py)

We establish concrete numerical values for all parameters, precisely matching [`verify_sentiment_rnn.py`](file:///Users/pasan/.gemini/antigravity-ide/brain/d3b699e8-4410-4b8c-b2cb-728045b5ca32/scratch/verify_sentiment_rnn.py).

### 1. Training Sample
We analyze the review:

```text
Sentence: "acting was not good"
Ground-truth label: y = 0.0 (Negative)
Sequence length: T = 4
```

Vocabulary tokens:
- `w_1 = 'acting' = 10`
- `w_2 = 'was'    = 11`
- `w_3 = 'not'    = 14`
- `w_4 = 'good'   = 15`

### 2. Initial State
The initial hidden state at `t = 0` is the zero vector:

```math
h_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{1 \times 3}
```

### 3. Word Embedding Vectors for the Sequence:
Retrieved directly from the embedding table `E`:

```math
e_1 = E[10] = \begin{bmatrix} 0.3 & -0.2 & 0.4 & 0.1 \end{bmatrix} \quad (\text{'acting'})
```
```math
e_2 = E[11] = \begin{bmatrix} 0.1 & 0.05 & -0.1 & 0.15 \end{bmatrix} \quad (\text{'was'})
```
```math
e_3 = E[14] = \begin{bmatrix} -0.5 & -0.3 & 0.1 & -0.6 \end{bmatrix} \quad (\text{'not'})
```
```math
e_4 = E[15] = \begin{bmatrix} 0.6 & 0.5 & -0.2 & 0.4 \end{bmatrix} \quad (\text{'good'})
```

### 4. Parameter Matrices:

```math
W_{xh} = \begin{bmatrix}
 0.4 & -0.2 &  0.3 \\
-0.3 &  0.5 & -0.1 \\
 0.2 & -0.1 &  0.4 \\
 0.1 &  0.3 & -0.2
\end{bmatrix} \in \mathbb{R}^{4 \times 3}
```

```math
W_{hh} = \begin{bmatrix}
 0.2 &  0.3 & -0.1 \\
-0.1 &  0.2 &  0.4 \\
 0.3 & -0.2 &  0.1
\end{bmatrix} \in \mathbb{R}^{3 \times 3}
```

```math
b_h = \begin{bmatrix} 0.1 & -0.05 & 0.0 \end{bmatrix} \in \mathbb{R}^{1 \times 3}
```

```math
W_{hy} = \begin{bmatrix}
 0.5 \\
-0.4 \\
 0.3
\end{bmatrix} \in \mathbb{R}^{3 \times 1}, \quad b_y = \begin{bmatrix} 0.1 \end{bmatrix} \in \mathbb{R}^{1 \times 1}
```

---

## Q7 — Forward Pass: Step-by-Step Numerical Walkthrough

We now trace every matrix multiplication and non-linear activation across all 4 timesteps.

### Timestep 1 (t = 1): Word "acting" (Token 10)

#### 1. Input and Prior State:
```text
e_1 = [0.3, -0.2, 0.4, 0.1]
h_0 = [0.0, 0.0, 0.0]
```

#### 2. Pre-activation `z_1`:
```math
e_1 W_{xh} = [0.3(0.4) + (-0.2)(-0.3) + 0.4(0.2) + 0.1(0.1), \;
              0.3(-0.2) + (-0.2)(0.5) + 0.4(-0.1) + 0.1(0.3), \;
              0.3(0.3) + (-0.2)(-0.1) + 0.4(0.4) + 0.1(-0.2)]
```
```text
e_1 @ W_xh = [0.12 + 0.06 + 0.08 + 0.01, -0.06 - 0.10 - 0.04 + 0.03, 0.09 + 0.02 + 0.16 - 0.02]
           = [0.27, -0.17, 0.25]
```

```text
h_0 @ W_hh = [0.0, 0.0, 0.0]
b_h        = [0.10, -0.05, 0.00]
```

Summing terms:
```math
z_1 = [0.27 + 0.0 + 0.10, \; -0.17 + 0.0 - 0.05, \; 0.25 + 0.0 + 0.0] = [0.37, -0.22, 0.25]
```

#### 3. Hidden State `h_1 = tanh(z_1)`:
```math
h_{1, 0} = \tanh(0.37) = 0.353992
```
```math
h_{1, 1} = \tanh(-0.22) = -0.216518
```
```math
h_{1, 2} = \tanh(0.25) = 0.244919
```
```text
h_1 = [0.353992, -0.216518, 0.244919]
```

---

### Timestep 2 (t = 2): Word "was" (Token 11)

#### 1. Input and Prior State:
```text
e_2 = [0.1, 0.05, -0.1, 0.15]
h_1 = [0.353992, -0.216518, 0.244919]
```

#### 2. Pre-activation `z_2`:
```text
e_2 @ W_xh = [0.02, 0.06, -0.045]
```
```math
h_1 W_{hh} = [0.353992(0.2) - 0.216518(-0.1) + 0.244919(0.3), \;
              0.353992(0.3) - 0.216518(0.2) + 0.244919(-0.2), \;
              0.353992(-0.1) - 0.216518(0.4) + 0.244919(0.1)]
           = [0.165926, 0.013910, -0.097515]
```

Summing with `b_h = [0.10, -0.05, 0.00]`:
```math
z_2 = [0.02 + 0.165926 + 0.10, \; 0.06 + 0.013910 - 0.05, \; -0.045 - 0.097515 + 0.0]
    = [0.285926, 0.023910, -0.142515]
```

#### 3. Hidden State `h_2 = tanh(z_2)`:
```text
h_2 = [tanh(0.285926), tanh(0.023910), tanh(-0.142515)]
    = [0.278381, 0.023906, -0.141557]
```

---

### Timestep 3 (t = 3): Word "not" (Token 14 - Negation Operator!)

#### 1. Input and Prior State:
```text
e_3 = [-0.5, -0.3, 0.1, -0.6]
h_2 = [0.278381, 0.023906, -0.141557]
```

#### 2. Pre-activation `z_3`:
```text
e_3 @ W_xh = [-0.15, -0.24, 0.04]
h_2 @ W_hh = [0.010818, 0.116607, -0.032432]
b_h        = [0.10, -0.05, 0.00]
```

Summing terms:
```math
z_3 = [-0.15 + 0.010818 + 0.10, \; -0.24 + 0.116607 - 0.05, \; 0.04 - 0.032432 + 0.0]
    = [-0.039182, -0.173393, 0.007568]
```

#### 3. Hidden State `h_3 = tanh(z_3)`:
```text
h_3 = [tanh(-0.039182), tanh(-0.173393), tanh(0.007568)]
    = [-0.039162, -0.171676, 0.007568]
```

> [!NOTE]
> Observe how the negation word `"not"` (`e_3`) flips the first hidden dimension `h_{t, 0}` from positive `+0.278381` to negative `-0.039162`, while driving `h_{t, 1}` to `-0.171676`.
> The recurrent memory is actively storing the negation state!

---

### Timestep 4 (t = 4): Word "good" (Token 15 - Final Timestep)

#### 1. Input and Prior State:
```text
e_4 = [0.6, 0.5, -0.2, 0.4]
h_3 = [-0.039162, -0.171676, 0.007568]
```

#### 2. Pre-activation `z_4`:
```text
e_4 @ W_xh = [0.09, 0.27, -0.03]
h_3 @ W_hh = [0.011606, -0.047597, -0.063997]
b_h        = [0.10, -0.05, 0.00]
```

Summing terms:
```math
z_4 = [0.09 + 0.011606 + 0.10, \; 0.27 - 0.047597 - 0.05, \; -0.03 - 0.063997 + 0.0]
    = [0.201606, 0.172403, -0.093997]
```

#### 3. Final Hidden State `h_4 = tanh(z_4)`:
```text
h_4 = [tanh(0.201606), tanh(0.172403), tanh(-0.093997)]
    = [0.198918, 0.170715, -0.093722]
```

The terminal state summarizing the entire 4-word review is:

```math
h_T = h_4 = \begin{bmatrix} 0.198918 & 0.170715 & -0.093722 \end{bmatrix} \in \mathbb{R}^{1 \times 3}
```

---

### Output Layer: Logit and Sentiment Probability

The classification projection evaluates exclusively on the terminal hidden state `h_T`:

#### 1. Sentiment Logit `o`:
```math
o = h_T W_{hy} + b_y = \begin{bmatrix} 0.198918 & 0.170715 & -0.093722 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.4 \\ 0.3 \end{bmatrix} + 0.1
```
```math
h_T W_{hy} = 0.198918(0.5) + 0.170715(-0.4) - 0.093722(0.3) = 0.099459 - 0.068286 - 0.028117 = 0.003057
```
```math
o = 0.003057 + 0.1 = 0.103057
```

#### 2. Sigmoid Activation:
```math
\hat{y} = \sigma(o) = \frac{1}{1 + e^{-0.103057}} = \frac{1}{1 + 0.902078} = \frac{1}{1.902078} = 0.525741
```

The model predicts a positive probability of `0.525741` (52.6%).

---

## Q8 — Binary Cross-Entropy (BCE) Loss

Because sentiment analysis is a binary classification task (`y \in \{0, 1\}`), we optimize the **Binary Cross-Entropy (BCE)** loss function:

```math
L = - \left[ y \ln(\hat{y} + \epsilon) + (1 - y) \ln(1 - \hat{y} + \epsilon) \right]
```

Where `\epsilon = 10^{-12}` is a numerical stabilization safeguard preventing `\ln(0) \to -\infty`.

Substituting our ground-truth target `y = 0.0` (Negative) and prediction `\hat{y} = 0.525741`:

```math
L = - \left[ 0 \cdot \ln(\hat{y}) + (1 - 0) \ln(1 - 0.525741) \right] = - \ln(0.474259)
```
```math
L = - (-0.746003) = 0.746003
```

The initial scalar loss for this review sequence is **0.746003**.

---

## Q9 — Backpropagation Through Time (BPTT) for Many-to-One: First-Principles Matrix Calculus

We derive every parameter gradient step-by-step using matrix calculus.

### 1. Output Layer Logit Error (`∂L / ∂o`)
Using the chain rule on Binary Cross-Entropy and Sigmoid:

```math
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}
```

The derivative of the Sigmoid function is:

```math
\frac{\partial \hat{y}}{\partial o} = \sigma(o)(1 - \sigma(o)) = \hat{y}(1 - \hat{y})
```

Multiplying by the chain rule:

```math
\delta_o = \frac{\partial L}{\partial o} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial o} = \left(\frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}\right) \left(\hat{y}(1 - \hat{y})\right) = \hat{y} - y \in \mathbb{R}^{1 \times 1}
```

Substituting `\hat{y} = 0.525741` and `y = 0.0`:

```math
\delta_o = 0.525741 - 0.0 = 0.525741
```

---

### 2. Output Projection Gradients (`W_hy`, `b_y`)
Because `o = h_T W_{hy} + b_y`:

```math
\frac{\partial L}{\partial W_{hy}} = h_T^T \delta_o \in \mathbb{R}^{H \times 1}
```
```math
\frac{\partial L}{\partial b_y} = \delta_o \in \mathbb{R}^{1 \times 1}
```

Substituting `h_T = [0.198918, 0.170715, -0.093722]` and `\delta_o = 0.525741`:

```math
\frac{\partial L}{\partial W_{hy}} = \begin{bmatrix} 0.198918 \\ 0.170715 \\ -0.093722 \end{bmatrix} (0.525741) = \begin{bmatrix} 0.104579 \\ 0.089752 \\ -0.049273 \end{bmatrix}
```
```math
\frac{\partial L}{\partial b_y} = 0.525741
```

---

### 3. Terminal Hidden State Gradient (`∂L / ∂h_T`)
At the final timestep `t = T = 4`, the error enters exclusively from the classification head:

```math
\delta_{h_T} = \frac{\partial L}{\partial h_T} = \delta_o W_{hy}^T \in \mathbb{R}^{1 \times H}
```

```math
\delta_{h_4} = (0.525741) \begin{bmatrix} 0.5 & -0.4 & 0.3 \end{bmatrix} = \begin{bmatrix} 0.262871 & -0.210297 & 0.157722 \end{bmatrix}
```

---

### 4. Recurrent Error Propagation for Intermediate Steps (`t < T`)

For any intermediate timestep `t \in \{T-1, ..., 1\}`, **there is no direct output loss**!
The error arrives solely from the subsequent timestep `t+1` through the recurrent weights `W_{hh}`:

```math
\delta_{h_t} = \delta_{z_{t+1}} W_{hh}^T \in \mathbb{R}^{1 \times H}
```

---

### 5. Backpropagating Through tanh (`∂L / ∂z_t`)
At any timestep `t`, the gradient through the hyperbolic tangent activation is:

```math
\delta_{z_t} = \frac{\partial L}{\partial z_t} = \delta_{h_t} \odot (1 - h_t^2) \in \mathbb{R}^{1 \times H}
```

Where `\odot` is the element-wise Hadamard product.

---

### 6. Gradients with Respect to Shared Recurrent Weights
Recall `z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h`.
Taking partial derivatives and accumulating across all `T` timesteps:

```math
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} e_t^T \delta_{z_t} \in \mathbb{R}^{D \times H}
```
```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} h_{t-1}^T \delta_{z_t} \in \mathbb{R}^{H \times H}
```
```math
\frac{\partial L}{\partial b_h} = \sum_{t=1}^{T} \delta_{z_t} \in \mathbb{R}^{1 \times H}
```

---

### 7. Gradients Flowing Into the Embedding Table `E`
For each word token `w_t` at timestep `t`:

```math
\frac{\partial L}{\partial e_t} = \delta_{z_t} W_{xh}^T \in \mathbb{R}^{1 \times D}
```

Because `e_t = E[w_t]`, this error gradient directly updates row `w_t` of the embedding matrix:

```math
E[w_t] \gets E[w_t] - \eta \left( \delta_{z_t} W_{xh}^T \right)
```

> [!TIP]
> If a word appears multiple times in the same review, its embedding gradients accumulate:
> `dE[k] = \sum_{t: w_t = k} \delta_{z_t} W_{xh}^T`.

---

## Q10 — Step-by-Step Numerical BPTT Walkthrough

We now run the backward equations in reverse chronological order from `t = 4` down to `t = 1`.

---

### Backward Step t = 4: Word "good" (Token 15)

#### 1. Incoming Gradient `\delta_{h_4}`:
From Step 3:
```text
delta_h_4 = [0.262871, -0.210297, 0.157722]
```

#### 2. Through tanh `\delta_{z_4}`:
```text
1 - h_4^2 = 1 - [0.198918^2, 0.170715^2, (-0.093722)^2]
          = 1 - [0.039568, 0.029143, 0.008784]
          = [0.960432, 0.970857, 0.991216]
```
```math
\delta_{z_4} = \delta_{h_4} \odot (1 - h_4^2) = [0.262871(0.960432), \; -0.210297(0.970857), \; 0.157722(0.991216)]
             = [0.252469, -0.204168, 0.156337]
```

#### 3. Embedding Gradient for "good" (`dE[15]`):
```math
\delta_{e_4} = \delta_{z_4} W_{xh}^T = \begin{bmatrix} 0.252469 & -0.204168 & 0.156337 \end{bmatrix} \begin{bmatrix} 0.4 & -0.3 & 0.2 & 0.1 \\ -0.2 & 0.5 & -0.1 & 0.3 \\ 0.3 & -0.1 & 0.4 & -0.2 \end{bmatrix}
```
```text
dE[15] = [0.188722, -0.193458, 0.133445, -0.067271]
```

#### 4. Propagate Gradient to `h_3`:
```math
\delta_{h_3} = \delta_{z_4} W_{hh}^T = \begin{bmatrix} 0.252469 & -0.204168 & 0.156337 \end{bmatrix} \begin{bmatrix} 0.2 & -0.1 & 0.3 \\ 0.3 & 0.2 & -0.2 \\ -0.1 & 0.4 & 0.1 \end{bmatrix}
             = [-0.026390, -0.003546, 0.132208]
```

---

### Backward Step t = 3: Word "not" (Token 14)

#### 1. Incoming Gradient `\delta_{h_3}`:
```text
delta_h_3 = [-0.026390, -0.003546, 0.132208]
```

#### 2. Through tanh `\delta_{z_3}`:
```text
1 - h_3^2 = 1 - [(-0.039162)^2, (-0.171676)^2, 0.007568^2] = [0.998466, 0.970527, 0.999943]
```
```math
\delta_{z_3} = \delta_{h_3} \odot (1 - h_3^2) = [-0.026350, -0.003441, 0.132200]
```

#### 3. Embedding Gradient for "not" (`dE[14]`):
```math
\delta_{e_3} = \delta_{z_3} W_{xh}^T = [0.029809, -0.007036, 0.047954, -0.030107]
```

#### 4. Propagate Gradient to `h_2`:
```math
\delta_{h_2} = \delta_{z_3} W_{hh}^T = [-0.019522, 0.054827, 0.006003]
```

---

### Backward Step t = 2: Word "was" (Token 11)

#### 1. Incoming Gradient `\delta_{h_2}`:
```text
delta_h_2 = [-0.019522, 0.054827, 0.006003]
```

#### 2. Through tanh `\delta_{z_2}`:
```text
1 - h_2^2 = 1 - [0.278381^2, 0.023906^2, (-0.141557)^2] = [0.922504, 0.999429, 0.979962]
```
```math
\delta_{z_2} = \delta_{h_2} \odot (1 - h_2^2) = [-0.018009, 0.054796, 0.005883]
```

#### 3. Embedding Gradient for "was" (`dE[11]`):
```math
\delta_{e_2} = \delta_{z_2} W_{xh}^T = [-0.016398, 0.032212, -0.006728, 0.013461]
```

#### 4. Propagate Gradient to `h_1`:
```math
\delta_{h_1} = \delta_{z_2} W_{hh}^T = [0.012248, 0.015113, -0.015774]
```

---

### Backward Step t = 1: Word "acting" (Token 10)

#### 1. Incoming Gradient `\delta_{h_1}`:
```text
delta_h_1 = [0.012248, 0.015113, -0.015774]
```

#### 2. Through tanh `\delta_{z_1}`:
```text
1 - h_1^2 = 1 - [0.353992^2, (-0.216518)^2, 0.244919^2] = [0.874690, 0.953120, 0.940015]
```
```math
\delta_{z_1} = \delta_{h_1} \odot (1 - h_1^2) = [0.010714, 0.014405, -0.014827]
```

#### 3. Embedding Gradient for "acting" (`dE[10]`):
```math
\delta_{e_1} = \delta_{z_1} W_{xh}^T = [-0.003044, 0.005471, -0.005229, 0.008358]
```

#### 4. Propagate Gradient to `h_0`:
```math
\delta_{h_0} = \delta_{z_1} W_{hh}^T = [0.007947, -0.004121, -0.001150]
```
*(Terminates BPTT; `h_0` has no prior learnable weights).*

---

### Final Accumulated Gradients

Summing parameter gradients across all 4 timesteps:

#### 1. Hidden Bias Gradient `db_h`:
```math
\frac{\partial L}{\partial b_h} = \sum_{t=1}^{4} \delta_{z_t} = \delta_{z_1} + \delta_{z_2} + \delta_{z_3} + \delta_{z_4}
```
```text
db_h = [ 0.010714 + (-0.018009) + (-0.026350) + 0.252469,
         0.014405 + 0.054796 + (-0.003441) + (-0.204168),
        -0.014827 + 0.005883 + 0.132200 + 0.156337 ]
     = [0.218824, -0.138409, 0.279593]
```

#### 2. Input Weight Gradient `dW_xh`:
```math
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{4} e_t^T \delta_{z_t} = \begin{bmatrix}
 0.166070 & -0.110979 &  0.023842 \\
 0.131096 & -0.101193 &  0.041768 \\
-0.047042 &  0.040772 & -0.024567 \\
 0.115167 & -0.069943 & -0.017386
\end{bmatrix}
```

#### 3. Recurrent Weight Gradient `dW_hh`:
```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{4} h_{t-1}^T \delta_{z_t} = \begin{bmatrix}
-0.023598 &  0.026435 &  0.032762 \\
-0.040074 &  0.023104 & -0.024953 \\
 0.001230 &  0.012362 & -0.016090
\end{bmatrix}
```

#### 4. Embedding Matrix Updates `dE`:
Non-zero gradient updates are committed exclusively to the rows active in this review:
```text
dE[10] ('acting'): [-0.003044,  0.005471, -0.005229,  0.008358]
dE[11] ('was'):    [-0.016398,  0.032212, -0.006728,  0.013461]
dE[14] ('not'):    [ 0.029809, -0.007036,  0.047954, -0.030107]
dE[15] ('good'):   [ 0.188722, -0.193458,  0.133445, -0.067271]
```
All other 12 vocabulary rows in `E` receive zero gradient (`dE[k] = \mathbf{0}`).

---

## Q11 — Gradient Pathologies in Many-to-One Sentiment Classification

Why is Many-to-One sentiment classification particularly susceptible to the **vanishing gradient problem**?

### The Asymmetric Gradient Horizon

In synchronous Many-to-Many models (`07-rnn-from-scratch`), an error signal is injected at every single timestep:
`\delta_{h_t} = \delta_{o_t} W_{hy}^T + \delta_{z_{t+1}} W_{hh}^T`.
Even if the historical gradient vanishes, every word receives an immediate local gradient `\delta_{o_t} W_{hy}^T`.

In Many-to-One sentiment classification:
- Error is injected **exclusively at the final timestep `T`**:
  ```math
  \delta_{h_T} = \delta_o W_{hy}^T
  ```
- For an early word at timestep `t = 1` in a 50-word review, the gradient must survive **49 consecutive matrix multiplications through `W_{hh}^T` and `(1 - h_j^2)`**:
  ```math
  \delta_{h_1} = \delta_o W_{hy}^T \prod_{j=2}^{T} \left( \operatorname{diag}(1 - h_j^2) W_{hh}^T \right)
  ```

### The Physical Manifestation:
1. **Trailing Word Recency Bias**:
   The network easily updates weights for words near the end of the sentence (`dE[15]` for `"good"` has magnitude `0.188`), but words near the beginning receive faded gradients (`dE[10]` for `"acting"` has magnitude `0.008`, over 23 times smaller!).
2. **Failure on Opening Modifiers**:
   If a critical negation appears at the very beginning of a long review (`"Not once during this 3-hour film did I feel entertained..."`), a vanilla Many-to-One RNN will suffer complete gradient vanishing and fail to learn the significance of the opening `"Not"`.

This directly motivates gated memory architectures with linear gradient channels:
- **LSTM** in Project 09 (`09-lstm`)
- **GRU** in Project 10 (`10-gru`)

---

## Q12 — Complete Mathematical Mapping to sentiment_rnn.py Implementation

This cross-reference connects every mathematical formula derived above directly to its implementation in [`src/sentiment_rnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/08-sentiment-rnn/src/sentiment_rnn.py):

| Component / Function | Mathematical Equation | NumPy Variables & Tensor Dimensions |
| :--- | :--- | :--- |
| `Embedding.forward(tokens)` | `e_t = E[w_t]` | `tokens: (B, T)` -> `X_embed: (B, T, D)` |
| `Embedding.backward(dX_embed)` | `dE[w_t] += delta_e_t` | Sparse accumulation into `dE: (V, D)` |
| `RNNCell.forward(e_t, h_prev)` | `z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h`, `h_t = \tanh(z_t)` | `e_t: (B, D)`, `h_prev: (B, H)` -> `h_t: (B, H)` |
| `ManyToOneRNN.forward(X_embed, lengths)` | Unrolls cell over `T` steps; extracts `h_T = H[lengths-1]` | `X_embed: (B, T, D)` -> `h_final: (B, H)` |
| `DenseOutput.forward(h_final)` | `o = h_T W_{hy} + b_y`, `\hat{y} = \sigma(o)` | `h_final: (B, H)` -> `prob: (B, 1)` |
| `binary_cross_entropy(y_hat, y)` | `L = -[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]` | `y_hat: (B, 1)`, `y: (B, 1)` -> scalar `L` |
| `DenseOutput.backward(prob, y)` | `\delta_o = \hat{y} - y`, `dW_{hy} = h_T^T \delta_o`, `db_y = \delta_o` | `dW_hy: (H, 1)`, `db_y: (1, 1)`, `delta_h_T: (B, H)` |
| `ManyToOneRNN.backward(delta_h_T)` | BPTT: `\delta_z = \delta_h \odot (1-h^2)`, `\delta_h \gets \delta_z W_{hh}^T` | Accumulates `dW_xh: (D, H), dW_hh: (H, H), db_h: (1, H)` |

---

## Summary of the Complete Mathematical Pipeline

```text
========================================================================================================================
FORWARD OPERATION (Text ──► Sentiment Probability)
========================================================================================================================
1. Vocabulary Lookup & Embedding:      e_t = E[w_t]                                     e_t in R^(1 x D)
2. Recurrent Pre-activation:           z_t = e_t @ W_xh + h_{t-1} @ W_hh + b_h          z_t in R^(1 x H)
3. Non-linear Memory State:            h_t = tanh(z_t)                                  h_t in R^(1 x H)
4. Terminal Sentence Summary:          h_T = h_{actual_len}                             h_T in R^(1 x H)
5. Linear Output Projection:           o = h_T @ W_hy + b_y                             o in R^(1 x 1)
6. Sigmoid Activation:                 y_hat = 1 / (1 + exp(-o))                        y_hat in (0, 1)
7. Binary Cross-Entropy Loss:          L = -[y * ln(y_hat) + (1 - y) * ln(1 - y_hat)]   L in R^+
========================================================================================================================
BACKWARD PROPAGATION THROUGH TIME (BPTT)
========================================================================================================================
1. Output Logit Error:                 delta_o = y_hat - y                              delta_o in R^(1 x 1)
2. Output Parameter Gradients:         dW_hy = h_T.T @ delta_o                          dW_hy in R^(H x 1)
                                       db_y  = delta_o                                  db_y in R^(1 x 1)
3. Terminal Hidden Error (t = T):      delta_h_T = delta_o @ W_hy.T                     delta_h_T in R^(1 x H)
4. Through tanh Activation:            delta_z_t = delta_h_t * (1 - h_t^2)              delta_z_t in R^(1 x H)
5. Prior Recurrent State (t < T):      delta_h_{t-1} = delta_z_t @ W_hh.T               delta_h_{t-1} in R^(1 x H)
6. Shared Weight Accumulation:         dW_xh = sum(e_t.T @ delta_z_t)                   dW_xh in R^(D x H)
                                       dW_hh = sum(h_{t-1}.T @ delta_z_t)               dW_hh in R^(H x H)
                                       db_h  = sum(delta_z_t)                           db_h in R^(1 x H)
7. Embedding Table Gradients:          dE[w_t] += delta_z_t @ W_xh.T                    dE in R^(V x D)
========================================================================================================================
```

This completes the foundational mathematics for Project 08. The next step is validating
each of these formulas in executable NumPy code in [`src/sentiment_rnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/08-sentiment-rnn/src/sentiment_rnn.py).
