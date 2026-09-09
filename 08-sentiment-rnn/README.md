# 08 — Sentiment Analysis with Recurrent Neural Networks (RNN)

## Goal

Apply Recurrent Neural Networks to **Natural Language Processing (NLP)** and **Sequence Classification** from first principles with pure NumPy.

While Project 07 (`07-rnn-from-scratch`) introduced the recurrent engine on character-level transitions using a synchronous **Many-to-Many** topology, Project 08 tackles real-world natural language text using the **Many-to-One** sequence topology:

```text
Raw Text Sentence ──► Tokenization ──► Vocabulary Lookup ──► Dense Word Embeddings ──► Many-to-One RNN ──► Final State h_T ──► Sentiment Logits ──► Probability
```

The network maps a variable-length sequence of word tokens `(w_1, w_2, ..., w_T)` into a single sentiment classification prediction `\hat{y} \in [0, 1]` (e.g., `0 = Negative`, `1 = Positive`):

```math
e_t = E[w_t] = \text{Embedding}(w_t) \in \mathbb{R}^{1 \times D}
```

```math
z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h
```

```math
h_t = \tanh(z_t)
```

```math
o = h_T W_{hy} + b_y
```

```math
\hat{y} = \sigma(o) = \frac{1}{1 + e^{-o}}
```

Where:
- `w_t`: Integer token ID for the word at timestep `t`.
- `E`: Learnable word embedding matrix of shape `(V, D)`.
- `e_t`: Dense embedding vector for word `w_t`.
- `h_{t-1}`: Recurrent hidden state carrying sentence context from previous words.
- `W_{xh}`: Embedding-to-hidden weight matrix.
- `W_{hh}`: Hidden-to-hidden recurrent weight matrix (shared across all timesteps).
- `b_h`: Hidden state bias offset.
- `h_t`: Updated hidden state memory at timestep `t`.
- `h_T`: Final hidden state summarizing the complete semantic meaning of the sentence.
- `W_{hy}`: Hidden-to-output classification projection matrix.
- `b_y`: Output bias offset.
- `o`: Unnormalized scalar logit for sentiment classification.
- `\hat{y}`: Predicted probability of positive sentiment.

---

## Why Sentiment Analysis? (The NLP & Sequence Classification Problem)

In digital communication, sentiment analysis extracts subjective opinion, emotional valence, and attitude from natural language text (reviews, customer feedback, tweets, financial news).

Before recurrent networks, traditional machine learning models processed text using **Bag-of-Words (BoW)** or **TF-IDF** fed into Logistic Regression or feedforward Artificial Neural Networks (`02-ann-from-scratch`). These traditional approaches fundamentally fail on human language.

---

### Problem 1 — Bag-of-Words Ignores Word Order & Temporal Flow

Bag-of-Words counts word frequencies while completely discarding syntax, word order, and temporal narrative.

#### Example A: Word Order Inversion
Consider two sentences containing identical words:

```text
Sentence 1: "The movie was not good, it was bad."  ──► Sentiment: Highly Negative
Sentence 2: "The movie was not bad, it was good."  ──► Sentiment: Highly Positive
```

Under Bag-of-Words, both sentences produce the **exact same feature frequency vector**:

```text
{'movie': 1, 'not': 1, 'good': 1, 'bad': 1, 'was': 2, 'it': 1, 'the': 1}
```

A static feedforward ANN or linear classifier receives identical inputs and is mathematically incapable of distinguishing between the two opposite sentiments!

#### Example B: Negation Flipping
A single negation word completely reverses the semantic polarity of downstream words:

```text
"I like this restaurant."     ──► Positive (+)
"I do not like this restaurant." ──► Negative (-)
```

In a feedforward BoW model, `"like"` and `"restaurant"` still register strong positive weights, often causing false positive misclassifications despite the presence of `"not"`.

#### Example C: Long-Range Contrastive Clauses ("but", "however")
Human reviews frequently contain contrastive sentiment shifts:

```text
"The cinematography was breathtaking, the musical score was majestic,
 and the actors gave brilliant performances, BUT the plot was completely
 nonsensical, boring, and ruined the entire experience."
```

In an RNN, the recurrent hidden state reads the sentence sequentially from left to right. When it processes the turning conjunction `"BUT"`, the hidden state transitions to prioritize the concluding negative judgment, overriding the initial praise.

---

### Problem 2 — One-Hot Word Vectors Suffer from Extreme Sparsity & Semantic Blindness

To feed words into a neural network, one might consider one-hot encoding every word in the dictionary:

```text
Vocabulary size V = 20,000 words

'great'   = [0, 0, 0, 1, 0, 0, ..., 0]  in R^(20000)
'awesome' = [0, 1, 0, 0, 0, 0, ..., 0]  in R^(20000)
'awful'   = [0, 0, 0, 0, 1, 0, ..., 0]  in R^(20000)
```

This causes three mathematical breakdowns:
1. **Curse of Dimensionality & Sparsity**: A sentence of 20 words requires `20 × 20,000 = 400,000` numbers, of which 99.99% are zeros.
2. **Orthogonal Vectors (Zero Semantic Relatedness)**: The dot product between any two distinct one-hot vectors is identically zero:
   ```math
   \text{great} \cdot \text{awesome} = 0, \quad \text{great} \cdot \text{awful} = 0
   ```
   The model cannot know that `"great"` and `"awesome"` are synonymous, nor that `"great"` and `"awful"` are antonyms.
3. **Severe Overfitting**: An input layer connected to one-hot vectors requires millions of weights, failing to generalize to unseen synonyms.

#### The Solution: Dense Word Embeddings
Instead of sparse 20,000-dimensional one-hot vectors, each word is mapped to a continuous, dense low-dimensional vector:

```math
e_w \in \mathbb{R}^{D} \quad (\text{e.g., } D = 16 \text{ or } D = 64)
```

In this dense geometric space, geometric proximity reflects semantic meaning:
- `\text{cosine\_similarity}(e_{\text{great}}, e_{\text{awesome}}) \approx +0.89`
- `\text{cosine\_similarity}(e_{\text{great}}, e_{\text{terrible}}) \approx -0.42`

---

## The Complete End-to-End Sentiment RNN Pipeline

The transformation from raw human text to a trained sentiment prediction proceeds through 7 explicit stages:

```text
 ┌────────────────────────────────────────────────────────┐
 │ 1. Raw Text String: "The movie was really great!"      │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Text Normalization & Lowercasing:                   │
 │    "the movie was really great"                        │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Tokenization (Word Splitting):                      │
 │    ['the', 'movie', 'was', 'really', 'great']          │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Numerical Encoding (Vocabulary Lookup):             │
 │    [4, 12, 7, 19, 5]                                   │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 5. Sequence Padding & Truncation (Fixed Length T = 6): │
 │    [4, 12, 7, 19, 5, <PAD>]  ──► [4, 12, 7, 19, 5, 0]  │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 6. Embedding Lookup Layer (E in R^(V x D)):            │
 │    e_1, e_2, e_3, e_4, e_5, e_6   (each in R^D)        │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 7. Many-to-One Recurrent Neural Network:               │
 │    e_1 ──► [ RNN Cell ] ──► h_1                        │
 │    e_2 ──► [ RNN Cell ] ──► h_2                        │
 │    e_3 ──► [ RNN Cell ] ──► h_3                        │
 │    e_4 ──► [ RNN Cell ] ──► h_4                        │
 │    e_5 ──► [ RNN Cell ] ──► h_5                        │
 │    e_6 ──► [ RNN Cell ] ──► h_6 (or masked h_actual)   │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 8. Classification Head & Probability:                  │
 │    o = h_final @ W_hy + b_y                            │
 │    y_hat = sigmoid(o) = 0.942  ──► Sentiment: POSITIVE │
 └────────────────────────────────────────────────────────┘
```

---

## Core NLP Components Explained

### 1. Tokenization and Normalization
Computers cannot directly compute mathematical gradients on string characters. Raw text must first be cleaned and tokenized:
- **Lowercasing**: Converting `"Great"`, `"GREAT"`, and `"great"` into a unified canonical form.
- **Punctuation Stripping**: Removing periods, commas, exclamation marks, or treating them as distinct tokens.
- **Word Tokenization**: Splitting sentences on whitespace into discrete word units.

### 2. Vocabulary and Special Tokens
A vocabulary dictionary maps every unique token to a unique non-negative integer index:

```text
Vocabulary: {
  '<PAD>': 0,   # Special token for sequence length padding
  '<UNK>': 1,   # Out-Of-Vocabulary (OOV) token for unseen words
  '<BOS>': 2,   # Beginning-Of-Sequence marker (optional)
  '<EOS>': 3,   # End-Of-Sequence marker (optional)
  'movie': 4,
  'great': 5,
  'bad':   6,
  'was':   7,
  ...
}
```

> [!IMPORTANT]
> The `<PAD>` token is assigned index `0`. This allows efficient array masking and zero-padding in batch matrix processing.
> Unseen words at test time are mapped to index `1` (`<UNK>`), preventing runtime crashes.

### 3. Sequence Padding and Batching
Natural language sentences naturally vary in length:
- Sentence A: `"Loved it!"` (Length = 2 words)
- Sentence B: `"The acting was utterly terrible and completely boring."` (Length = 8 words)

To process multiple sentences in parallel within a vectorized NumPy batch tensor `(B, T)`, all sequences must be aligned to a uniform maximum sequence length `T`:

```text
Raw Sequences:
A: [15, 22]
B: [4, 8, 7, 31, 14, 28, 19, 35]

Padded to Length T = 6 (Post-Padding with <PAD> = 0):
A: [15, 22,  0,  0,  0,  0]  (Padded with 4 zeros)
B: [ 4,  8,  7, 31, 14, 28]  (Truncated from 8 down to 6)
```

#### The Trailing Padding Corruption Problem & Masking
When an RNN processes trailing `<PAD>` tokens (`e_t = \mathbf{0}`), the hidden state continues to update:

```math
h_{\mathrm{pad}} = \tanh(\mathbf{0} W_{xh} + h_{t-1} W_{hh} + b_h) \ne h_{t-1}
```

Because of `b_h` and `W_{hh}`, trailing zeros **corrupt** the meaningful sentence context accumulated at the true final word!
In Project 08, we explore two mathematical solutions:
1. **Dynamic Length Indexing**: Extracting `h_t` at the exact index `t = \text{actual\_len}` for each sentence in the batch.
2. **Hidden State Masking**: Freezing the hidden state (`h_t = h_{t-1}`) whenever `token == <PAD>`.

---

### 4. Dense Word Embeddings Layer

The Embedding layer can be formalized as an indexed matrix lookup or a linear projection from one-hot vectors:

```math
e_t = E[w_t] = \text{one\_hot}(w_t) \cdot E
```

Where:
- `E \in \mathbb{R}^{V \times D}`: Embedding table storing `V` vocabulary words as `D`-dimensional row vectors.
- `w_t \in \{0, 1, ..., V-1\}`: Integer word index.
- `e_t \in \mathbb{R}^{1 \times D}`: Resulting word embedding vector.

```text
Embedding Matrix E: (Vocab Size V = 8, Embedding Dim D = 4)

             Dim 0   Dim 1   Dim 2   Dim 3
Index 0 <PAD> [ 0.00,   0.00,   0.00,   0.00 ]  <── Zero vector for padding
Index 1 <UNK> [ 0.12,  -0.05,   0.31,  -0.18 ]
Index 2 'great'[ 0.85,   0.72,  -0.11,   0.45 ]  <── High positive valence
Index 3 'good' [ 0.78,   0.65,  -0.08,   0.40 ]  <── Close to 'great'
Index 4 'bad'  [-0.81,  -0.69,   0.25,  -0.38 ]  <── Opposite direction!
Index 5 'film' [ 0.05,   0.10,   0.92,   0.15 ]
Index 6 'not'  [-0.40,  -0.20,   0.05,  -0.55 ]
Index 7 'the'  [ 0.01,  -0.02,   0.04,   0.01 ]
```

During backpropagation, error gradients flow directly into the embedding matrix:

```math
\frac{\partial L}{\partial E[w_t]} = \delta_{e_t} = \delta_{z_t} W_{xh}^T
```

Words that appear in positive reviews (`"great"`, `"loved"`, `"superb"`) automatically cluster together in embedding space through gradient descent!

---

## Sequence Topology: Many-to-One Architecture

In Project 07, we utilized **Many-to-Many (Synchronous)** to predict characters at every single timestep.
In Project 08, we configure the RNN in the **Many-to-One** topology:

```text
                         Sentiment Probability y_hat
                                    ▲
                                    │
                              [ Sigmoid ]
                                    ▲
                                    │
                               Logit o
                                    ▲
                                    │
                            [ Output Dense ]
                               (W_hy, b_y)
                                    ▲
                                    │
   ┌────────────────────────────────┴────────────────────────────────┐
   │                    Final Hidden State h_T                       │
   │               (Fixed-Size Sentence Representation)              │
   └────────────────────────────────┬────────────────────────────────┘
                                    ▲
                                    │
         e₁            e₂           │           e_T
         │             │            │            │
         ▼             ▼            │            ▼
      ┌────┐        ┌────┐        ┌────┐       ┌────┐
h₀ ──►│ H₁ │───────►│ H₂ │───────►│ H₃ │ ... ─►│ H_T│
      └────┘        └────┘        └────┘       └────┘
         ▲             ▲            ▲            ▲
         │             │            │            │
       Word 1        Word 2       Word 3       Word T
```

### Contrast: Many-to-Many vs Many-to-One

| Architectural Feature | Project 07 (`07-rnn-from-scratch`) | Project 08 (`08-sentiment-rnn`) |
| :--- | :--- | :--- |
| **Sequence Topology** | Many-to-Many (Synchronous) | **Many-to-One** |
| **Input Type** | Character one-hot vectors `(1, 4)` | **Word token embeddings** `(1, D)` |
| **Output Timesteps** | Every timestep `t = 1 ... T` | **Only at final step** `t = T` |
| **Classification Head** | Dense layer at each timestep | **Single Dense layer at sentence end** |
| **Loss Evaluation** | Sum across all timesteps: `L = sum_t L_t` | **Single scalar loss at step T**: `L = L_T` |
| **Gradient Injection** | `\delta_{o_t}` injected at every timestep | `\delta_o` injected **ONLY** at step `T` |
| **BPTT Flow** | `\delta_{h_t} = \delta_{o_t} W_{hy}^T + \delta_{z_{t+1}} W_{hh}^T` | For `t < T`: `\delta_{h_t} = \delta_{z_{t+1}} W_{hh}^T` (pure time flow!) |
| **Application** | Autoregressive character generation | **Document & sentence sentiment classification** |

---

## The Mathematics of Many-to-One Sentiment Classification

### 1. Forward Pass Equations

Given a sentence of length `T` with word tokens `[w_1, w_2, ..., w_T]`:

1. **Embedding Lookup**:
   ```math
   e_t = E[w_t] \in \mathbb{R}^{1 \times D} \quad \text{for } t = 1, 2, ..., T
   ```
2. **Recurrent Hidden Updates**:
   ```math
   z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h \in \mathbb{R}^{1 \times H}
   ```
   ```math
   h_t = \tanh(z_t) \in \mathbb{R}^{1 \times H}
   ```
   *(with initial state `h_0 = \mathbf{0} \in \mathbb{R}^{1 \times H}`)*
3. **Sentence Summary State**:
   The final hidden vector `h_T` encapsulates the complete sequential meaning of the review:
   ```math
   h_{\mathrm{sentence}} = h_T \in \mathbb{R}^{1 \times H}
   ```
4. **Classification Projection**:
   ```math
   o = h_T W_{hy} + b_y \in \mathbb{R}^{1 \times 1}
   ```
5. **Binary Sentiment Probability**:
   ```math
   \hat{y} = \sigma(o) = \frac{1}{1 + e^{-o}} \in (0, 1)
   ```

---

### 2. Binary Cross-Entropy (BCE) Loss

For binary sentiment labels `y \in \{0, 1\}` (where `0 = Negative`, `1 = Positive`):

```math
L = - \left[ y \ln(\hat{y} + \epsilon) + (1 - y) \ln(1 - \hat{y} + \epsilon) \right]
```

Where `\epsilon = 10^{-12}` prevents numerical log-zero overflow.

---

### 3. Backpropagation Through Time (BPTT) for Many-to-One

Because loss is evaluated exclusively at the final timestep `T`, backpropagation through time exhibits an elegant structure:

#### Step 1: Output Gradient
The derivative of Binary Cross-Entropy combined with the Sigmoid activation simplifies directly to:

```math
\delta_o = \frac{\partial L}{\partial o} = \hat{y} - y \in \mathbb{R}^{1 \times 1}
```

#### Step 2: Output Weights & Bias Gradients
```math
\frac{\partial L}{\partial W_{hy}} = h_T^T \delta_o \in \mathbb{R}^{H \times 1}
```
```math
\frac{\partial L}{\partial b_y} = \delta_o \in \mathbb{R}^{1 \times 1}
```

#### Step 3: Gradient Arriving at Final Hidden State `h_T`
At timestep `T`, error enters exclusively from the classification head:

```math
\delta_{h_T} = \frac{\partial L}{\partial h_T} = \delta_o W_{hy}^T \in \mathbb{R}^{1 \times H}
```

#### Step 4: Recurrent Propagation Backward Through Time (`t = T-1, ..., 1`)
For all earlier timesteps `t < T`, there is **no direct output loss**:

```math
\delta_{z_t} = \delta_{h_t} \odot (1 - h_t^2) \in \mathbb{R}^{1 \times H}
```
```math
\delta_{h_{t-1}} = \delta_{z_t} W_{hh}^T \in \mathbb{R}^{1 \times H}
```

> [!NOTE]
> In Many-to-One RNNs, early word embeddings receive gradients that have traveled **entirely backward through time** along the recurrent chain `\delta_{z_T} \to \delta_{z_{T-1}} \to ... \to \delta_{z_1}`.
> This makes Many-to-One RNNs an ideal laboratory for studying the **vanishing gradient problem**!

#### Step 5: Shared Parameter Gradients
Accumulated across all unrolled timesteps:

```math
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} e_t^T \delta_{z_t} \in \mathbb{R}^{D \times H}
```
```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} h_{t-1}^T \delta_{z_t} \in \mathbb{R}^{H \times H}
```
```math
\frac{\partial L}{\partial b_h} = \sum_{t=1}^{T} \delta_{z_t} \in \mathbb{R}^{1 \times H}
```
```math
\frac{\partial L}{\partial E[w_t]} = \delta_{z_t} W_{xh}^T \in \mathbb{R}^{1 \times D}
```

---

## Concrete Toy Dataset for Educational Walkthrough

To verify every mathematical step and gradient assertion by hand, we design a transparent 6-sentence movie sentiment dataset:

```text
====================================================================================================
SENTENCE                                                  WORDS               LABEL     POLARITY
====================================================================================================
1. "great movie loved it"                                 ['great', 'movie', 'loved', 'it']     1       Positive
2. "terrible film hated it"                               ['terrible', 'film', 'hated', 'it']   0       Negative
3. "acting was really superb"                             ['acting', 'was', 'really', 'superb'] 1       Positive
4. "plot was completely awful"                            ['plot', 'was', 'completely', 'awful']0       Negative
5. "acting was not good"                                  ['acting', 'was', 'not', 'good']      0       Negative (Negation!)
6. "not bad really loved"                                 ['not', 'bad', 'really', 'loved']     1       Positive (Double-Twist!)
====================================================================================================
```

### Vocabulary Setup (`V = 16`, `D = 4`, `H = 3`)

```text
Vocabulary Mapping (V = 16):
  0: <PAD>       4: great        8: superb      12: plot
  1: <UNK>       5: loved        9: bad         13: completely
  2: movie       6: terrible    10: acting      14: not
  3: film        7: hated       11: was         15: good
```

### Parameter Budget

| Parameter | Shape | Values | Total Scalars |
| :--- | :--- | :--- | :--- |
| **Embedding Table `E`** | `(V, D) = (16, 4)` | Dense word representations | 64 weights |
| **Input Weights `W_xh`** | `(D, H) = (4, 3)` | Projection from embedding to hidden | 12 weights |
| **Recurrent Weights `W_hh`** | `(H, H) = (3, 3)` | Temporal memory transitions | 9 weights |
| **Hidden Bias `b_h`** | `(1, H) = (1, 3)` | Hidden neuron offset | 3 biases |
| **Output Weights `W_hy`** | `(H, 1) = (3, 1)` | Linear projection to sentiment logit | 3 weights |
| **Output Bias `b_y`** | `(1, 1) = (1, 1)` | Sentiment baseline bias | 1 bias |
| **TOTAL** | — | — | **92 learnable parameters** |

---

## Parameters and Tensor Shapes Reference

| Tensor | NumPy Shape | Mathematical Notation | Description |
| :--- | :--- | :--- | :--- |
| `X_tokens` | `(B, T)` | `X \in \mathbb{N}^{B \times T}` | Batch of integer token IDs |
| `E` | `(V, D)` | `E \in \mathbb{R}^{V \times D}` | Word embedding matrix |
| `X_embed` | `(B, T, D)` | `X_e \in \mathbb{R}^{B \times T \times D}` | Dense embedded sequence batch |
| `e_t` | `(B, D)` | `e_t \in \mathbb{R}^{B \times D}` | Word embedding slice at timestep `t` |
| `W_xh` | `(D, H)` | `W_{xh} \in \mathbb{R}^{D \times H}` | Embedding-to-hidden weight matrix |
| `W_hh` | `(H, H)` | `W_{hh} \in \mathbb{R}^{H \times H}` | Recurrent hidden-to-hidden weight matrix |
| `b_h` | `(1, H)` | `b_h \in \mathbb{R}^{1 \times H}` | Hidden state bias offset |
| `h_prev` | `(B, H)` | `h_{t-1} \in \mathbb{R}^{B \times H}` | Hidden state memory from step `t-1` |
| `h_t` | `(B, H)` | `h_t \in \mathbb{R}^{B \times H}` | Updated hidden state at timestep `t` |
| `H_seq` | `(B, T, H)` | `H \in \mathbb{R}^{B \times T \times H}` | Complete sequence of hidden states |
| `h_T` | `(B, H)` | `h_T \in \mathbb{R}^{B \times H}` | Final summary state for classification |
| `W_hy` | `(H, 1)` | `W_{hy} \in \mathbb{R}^{H \times 1}` | Hidden-to-sentiment projection weights |
| `b_y` | `(1, 1)` | `b_y \in \mathbb{R}^{1 \times 1}` | Output sentiment bias |
| `logit` | `(B, 1)` | `o \in \mathbb{R}^{B \times 1}` | Raw unnormalized sentiment logit |
| `prob` | `(B, 1)` | `\hat{y} \in (0, 1)^{B \times 1}` | Predicted positive sentiment probability |
| `Y` | `(B, 1)` | `Y \in \{0, 1\}^{B \times 1}` | Ground-truth binary sentiment label |

---

## Project Structure

```text
08-sentiment-rnn/
├── README.md                      # Complete architecture, NLP pipeline, and 101-topic syllabus
├── src/
│   └── sentiment_rnn.py           # Pure NumPy Tokenizer, Embedding, Many-to-One RNN, & Trainer
└── steps/
    └── step-01-mathematics.md      # Detailed mathematical proofs, BPTT matrix calculus, and walkthrough
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Pending | `steps/step-01-mathematics.md`: Word embeddings, Many-to-One unrolling, BPTT matrix calculus |
| **2. NumPy Implementation** | Pending | `src/sentiment_rnn.py`: `Tokenizer`, `Embedding`, `RNNCell`, `ManyToOneRNN`, `BCE` |
| **3. Understand Forward Pass** | Pending | Tracing token vectors through embedding matrix and unrolled RNN steps to `h_T` |
| **4. Understand Loss** | Pending | Binary Cross-Entropy formulation, numerical stability clipping, decision thresholds |
| **5. Derive Gradients** | Pending | Backpropagation Through Time starting strictly from final step `T` down to `t=1` |
| **6. Implement Backpropagation** | Pending | Vectorized gradient accumulation across time and sparse embedding table updates |
| **7. Train Model** | Pending | Training on toy sentiment dataset until 100% training accuracy on negations |
| **8. Debug & Analyze** | Pending | Investigating trailing padding corruption, sequence masking, and gradient vanishing |
| **9. PyTorch Implementation** | Pending | Equivalent model using `nn.Embedding`, `nn.RNN`, and `nn.BCEWithLogitsLoss` |
| **10. Compare Results** | Pending | Numerical comparison of weights, embeddings, loss curves, and evaluation metrics |

---

## Complete 101-Topic Foundational Curriculum

This project systematically builds through the following 101 core deep learning topics across 16 structured modules:

```text
NLP & TEXT PREPROCESSING
│
├── 01. Raw natural language text
├── 02. Text cleaning and lowercasing
├── 03. Punctuation handling
├── 04. Whitespace normalization
├── 05. Word-level tokenization
├── 06. Character-level vs word-level tradeoffs
├── 07. Subword tokenization intuition (BPE overview)
│
▼
VOCABULARY MANAGEMENT
│
├── 08. Building a vocabulary from a corpus
├── 09. Word frequency distributions (Zipf's law)
├── 10. Vocabulary truncation & cutoff thresholds
├── 11. Word-to-index mapping (word2idx)
├── 12. Index-to-word mapping (idx2word)
├── 13. Special token: <PAD> (Index 0)
├── 14. Special token: <UNK> (Out-Of-Vocabulary handling)
├── 15. Special tokens: <BOS> and <EOS>
├── 16. Vocabulary size constraints (V)
│
▼
INTEGER ENCODING & SEQUENCES
│
├── 17. Converting text tokens to integer vectors
├── 18. Handling variable-length sequences
├── 19. Maximum sequence length selection (T_max)
├── 20. Pre-padding vs post-padding
├── 21. Pre-truncation vs post-truncation
├── 22. Sequence length tensor (actual lengths per sample)
├── 23. Binary sequence attention masks
│
▼
WORD EMBEDDINGS (DENSE VECTOR SPACES)
│
├── 24. Limitations of one-hot word vectors
├── 25. High-dimensionality and sparsity
├── 26. Orthogonality problem (zero semantic similarity)
├── 27. The embedding hypothesis (distributional semantics)
├── 28. Dense embedding matrix representation E in R^(V x D)
├── 29. Embedding dimension selection (D)
├── 30. Embedding lookup as indexed row retrieval
├── 31. Embedding lookup as one-hot matrix multiplication
├── 32. Cosine similarity and Euclidean distance in embedding space
├── 33. Word clustering and semantic trajectories
│
▼
RECURRENT CELL REFRESHER
│
├── 34. The simple RNN cell recurrence
├── 35. Input-to-hidden transformation (e_t @ W_xh)
├── 36. Hidden-to-hidden transformation (h_{t-1} @ W_hh)
├── 37. Hidden bias offset (b_h)
├── 38. Pre-activation summation (z_t)
├── 39. Hyperbolic tangent non-linearity (tanh)
├── 40. Bounded memory states h_t in (-1, 1)
│
▼
MANY-TO-ONE TOPOLOGY FOR TEXT
│
├── 41. Sequence topologies recap (1:1, 1:M, M:1, M:M)
├── 42. Why sentiment analysis is Many-to-One
├── 43. Unrolling the sequence over T timesteps
├── 44. Initial hidden state initialization (h_0 = 0)
├── 45. Sequential state evolution across words
├── 46. Context accumulation over long sentences
├── 47. Negation flipping inside recurrent states
├── 48. Extracting the sentence summary vector h_T
│
▼
PADDING DYNAMICS & MASKING
│
├── 49. The trailing padding corruption problem
├── 50. Non-zero tanh response to zero inputs
├── 51. Dynamic indexing (extracting h_actual)
├── 52. Hidden state masking logic
├── 53. Forward pass with padding masks
│
▼
CLASSIFICATION HEAD & PROBABILITY
│
├── 54. Linear classification projection (W_hy in R^(H x 1))
├── 55. Sentiment classification bias (b_y)
├── 56. Unnormalized sentiment logit (o)
├── 57. The Sigmoid activation function sigma(o)
├── 58. Properties of Sigmoid: range (0, 1)
├── 59. Interpreting output as positive probability P(y=1|X)
├── 60. Decision boundary thresholding (0.5 cutoff)
│
▼
LOSS FUNCTION (BINARY CROSS-ENTROPY)
│
├── 61. Maximum likelihood estimation for binary labels
├── 62. Binary Cross-Entropy formula
├── 63. Negative log-likelihood penalties
├── 64. Numerical stability with epsilon clipping (1e-12)
├── 65. BCE loss vs Mean Squared Error for classification
├── 66. Batch-averaged sequence loss
│
▼
BACKPROPAGATION THROUGH TIME (MANY-TO-ONE)
│
├── 67. Loss computational graph for Many-to-One
├── 68. Output logit gradient: delta_o = y_hat - y
├── 69. Output weight gradient: dW_hy = h_T.T @ delta_o
├── 70. Output bias gradient: db_y = delta_o
├── 71. Error injection at final step: delta_h_T = delta_o @ W_hy.T
├── 72. Zero output error for intermediate steps (t < T)
├── 73. Pure temporal error propagation: delta_h_t = delta_z_{t+1} @ W_hh.T
├── 74. Backpropagating through tanh: delta_z_t = delta_h_t * (1 - h_t^2)
├── 75. Masked backward pass through <PAD> tokens
│
▼
ACCUMULATING PARAMETER GRADIENTS
│
├── 76. Input weight gradient: dW_xh = sum(e_t.T @ delta_z_t)
├── 77. Recurrent weight gradient: dW_hh = sum(h_{t-1}.T @ delta_z_t)
├── 78. Hidden bias gradient: db_h = sum(delta_z_t)
├── 79. Embedding gradient formulation: dE[w_t] = delta_z_t @ W_xh.T
├── 80. Sparse vs dense embedding gradient updates
├── 81. Accumulating duplicate word gradients in the same sentence
│
▼
OPTIMIZATION & GRADIENT DYNAMICS
│
├── 82. Stochastic Gradient Descent (SGD) for NLP
├── 83. Learning rate tuning
├── 84. Vanishing gradient analysis over long reviews
├── 85. Why early words lose influence on h_T
├── 86. Exploding gradients in recurrent weights
├── 87. Global gradient norm clipping
├── 88. Embedding weight regularization
│
▼
EVALUATION & INFERENCE
│
├── 89. Prediction pipeline for raw custom strings
├── 90. Accuracy metric computation
├── 91. Confusion matrix (TP, FP, TN, FN)
├── 92. Precision, Recall, and F1-score
├── 93. Analyzing false positives on sarcasm
├── 94. Analyzing false negatives on complex negations
│
▼
MODEL DIAGNOSTICS & COMPARISON
│
├── 95. Inspecting learned word embeddings via cosine similarity
├── 96. Visualizing hidden state trajectories h_t across words
├── 97. Numerical gradient checking with finite differences
├── 98. Building equivalent model in PyTorch
├── 99. PyTorch nn.Embedding, nn.RNN, nn.Linear alignment
├── 100. Benchmarking NumPy vs PyTorch training dynamics
│
▼
MASTERY MILESTONE
│
└── 101. Complete end-to-end Sentiment RNN from scratch
```

---

## Key Takeaways

1. **Many-to-One Sequence Topology**:
   Unlike character-level next-step models that predict at every timestep, sentiment analysis compresses an entire variable-length sequence of word embeddings into a single terminal hidden state `h_T` that summarizes the overall semantic opinion.
2. **Dense Embeddings Solve Orthogonality**:
   One-hot encoding words wastes memory and destroys semantic relations. Learnable dense word embeddings map words into a continuous geometric vector space where semantic similarity corresponds to vector proximity.
3. **Word Order Matters**:
   Bag-of-Words models cannot distinguish `"not bad"` from `"not good"`. The sequential recurrent hidden state `h_t` retains temporal order, successfully resolving negations and contrastive conjunctions.
4. **Padding Requires Careful Handling**:
   Trailing `<PAD>` tokens pass non-zero bias signals through `\tanh(\dots)`, risking corruption of the true sentence summary `h_{\mathrm{actual}}` unless masked or explicitly indexed.
5. **Pure Temporal BPTT**:
   Because output loss exists only at step `T`, gradients at earlier timesteps flow purely through recurrent transitions `W_{hh}^T`, clearly highlighting vanishing gradient degradation over long sequences.

---

## Next Steps

1. Create [`steps/step-01-mathematics.md`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/08-sentiment-rnn/steps/step-01-mathematics.md) detailing the complete theoretical proofs, embedding calculus, and reverse-time BPTT derivations for Many-to-One classification.
2. Implement the complete pure NumPy sentiment pipeline in [`src/sentiment_rnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/08-sentiment-rnn/src/sentiment_rnn.py), including `Tokenizer`, `Embedding`, `ManyToOneRNN`, and automated numerical assertion tests.
