# 07 — RNN From Scratch

## Goal

Understand neural networks for sequential data from first principles with NumPy.
While an Artificial Neural Network (ANN, `02-ann-from-scratch`) assumes inputs are independent,
and a Convolutional Neural Network (CNN, `05-cnn-from-scratch`) specializes in 2D spatial grid topology,
a **Recurrent Neural Network (RNN)** introduces internal memory via recurrent hidden states to model
temporal dependencies where the order of observations matters.

The network implements the canonical simple RNN recurrence across timesteps:

```math
z_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h
```

```math
h_t = \tanh(z_t)
```

```math
o_t = h_t W_{hy} + b_y
```

```math
\hat{y}_t = \mathrm{softmax}(o_t)
```

Where:
- `x_t`: Input vector at timestep `t`.
- `h_{t-1}`: Hidden state vector from previous timestep `t-1` (memory context).
- `W_{xh}`: Input-to-hidden weight matrix.
- `W_{hh}`: Hidden-to-hidden recurrent weight matrix (shared across all timesteps).
- `b_h`: Hidden bias vector.
- `h_t`: Updated hidden state vector at timestep `t`.
- `W_{hy}`: Hidden-to-output projection weight matrix.
- `b_y`: Output bias vector.
- `o_t`: Unnormalized output logits at timestep `t`.
- `\hat{y}_t`: Normalized categorical probability distribution over classes.

---

## Why Do We Need RNNs?

### Traditional ANNs Cannot Handle Sequential Data

A traditional feedforward ANN processes inputs independently:

```text
x₁ ──► ANN ──► y₁
x₂ ──► ANN ──► y₂
x₃ ──► ANN ──► y₃
```

There is no memory or internal state connecting `x₁ ──► y₁` to `x₂ ──► y₂`. Each input is treated in isolation.

#### What Is Sequential Data?
In sequential data, **order matters**. The current observation `x_t` depends on, or is conditioned by, historical context:

```text
x₁ ──► x₂ ──► x₃ ──► x₄ ──► ... ──► xₜ
```

#### Real-World Sequential Examples

1. **Time-Series Business Forecasting**:
   Predicting Year 11 revenue requires learning trends from Years 1 through 10:
   ```text
   Year 1 ──► Year 2 ──► ... ──► Year 10 ──► [ Model ] ──► Year 11 Prediction
   ```
2. **Financial Market Sequences**:
   Observations over consecutive days provide temporal momentum context:
   ```text
   Day 1 ──► Day 2 ──► ... ──► Day 10 ──► [ Model ] ──► Next State Prediction
   ```
   *(Note: While financial markets are inherently noisy, the underlying data structure is strictly sequential.)*
3. **Natural Language Processing (NLP)**:
   Word order dictates semantic meaning:
   ```text
   "I love machine learning"  ≠  "Learning machine love I"
   ```
4. **Machine Translation**:
   Translating sentences word by word requires sentence-level context:
   ```text
   x₁("I") ──► x₂("love") ──► x₃("programming") ──► [ Translation Head ] ──► Target Language
   ```
   *(Modern production translation utilizes Transformers, but RNNs are the foundational basis for sequence-to-sequence learning.)*
5. **Speech Recognition**:
   Continuous acoustic waveforms vary dynamically across time:
   ```text
   audio₁ ──► audio₂ ──► audio₃ ──► ... ──► audioₜ ──► Transcribed Text
   ```

---

## Core RNN Concepts

### The Hidden State: Internal Temporal Memory

An RNN introduces a recurrent **hidden state** `h_t` that carries information forward from past timesteps:

```text
x₁ ──► RNN ──► h₁
                │
x₂ ──► RNN ──► h₂
                │
x₃ ──► RNN ──► h₃ ──► Output Prediction
```

At timestep `t`, the RNN cell receives two inputs:
1. **Current input**: `x_t`
2. **Previous hidden state**: `h_{t-1}`

And produces:
1. **New hidden state**: `h_t`

```text
               Previous Hidden State
                      hₜ₋₁
                        │
                        ▼
Current Input ────► [ RNN Cell ] ────► New Hidden State
     xₜ                 │                    hₜ
                        ▼
                   Output yₜ (optional)
```

General recurrence relation:

```math
h_t = f(x_t, h_{t-1})
```

For a standard Simple RNN:

```math
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
```

---

### Unrolling an RNN Through Time

A single recurrent cell with a feedback loop can be unrolled into a chain of timesteps:

```text
x_t        x_{t+1}        x_{t+2}
 │            │              │
 ▼            ▼              ▼
┌────┐      ┌────┐         ┌────┐
│ H  │─────►│ H  │────────►│ H  │
└────┘      └────┘         └────┘
 │            │              │
 ▼            ▼              ▼
y_t         y_{t+1}        y_{t+2}
```

> [!IMPORTANT]
> **Parameter Sharing Across Time**: The matrices `W_xh`, `W_hh`, `W_hy` and biases `b_h`, `b_y` are **identical and shared** at every single timestep. The unrolled diagram illustrates computation flow across time, not separate layers!

---

### Complete Simple RNN Pipeline Flow

```text
                    Sequence Input (x₁, x₂, ..., x_T)
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   x₁  (Timestep 1)  │
                        └──────────┬──────────┘
                                   ↓
                         [ Simple RNN Cell ] ◄── Initial State h₀ = 0
                                   ↓
                                  h₁
                                   ↓
                        ┌─────────────────────┐
                        │   x₂  (Timestep 2)  │
                        └──────────┬──────────┘
                                   ↓
                         [ Simple RNN Cell ] ◄── h₁
                                   ↓
                                  h₂
                                   ↓
                        ┌─────────────────────┐
                        │   x₃  (Timestep 3)  │
                        └──────────┬──────────┘
                                   ↓
                         [ Simple RNN Cell ] ◄── h₂
                                   ↓
                                  h₃
                                   ↓
                        [ Dense Output Layer ]
                                   ↓
                              Logits o₃
                                   ↓
                               Softmax
                                   ↓
                        Predicted Probabilities ŷ
                                   ↓
                        Categorical Cross-Entropy
                                   ↓
                               Scalar Loss L
                                   ↓
                     Backpropagation Through Time (BPTT)
                                   ↓
                          Parameter Gradients
                   (∂L/∂W_xh, ∂L/∂W_hh, ∂L/∂W_hy, ∂L/∂b)
                                   ↓
                            Gradient Descent
                                   ↓
                            Updated Weights
```

---

## Sequence Topologies: The 5 Fundamental RNN Patterns

In traditional feedforward neural networks (ANNs), models map a single fixed-size input vector to a single fixed-size output vector (One-to-One).
In recurrent neural networks, inputs and outputs can be variable-length sequences, giving rise to **5 distinct sequence topologies**:

```text
1. One-to-One          2. One-to-Many         3. Many-to-One
     ┌───┐                  ┌───┐               ┌───┐   ┌───┐   ┌───┐
     │ y │                  │y₁ │               │   │   │   │   │ y │
     └───┘                  └───┘               └───┘   └───┘   └───┘
       ▲                      ▲                   ▲       ▲       ▲
       │                      │                   │       │       │
     ┌───┐                  ┌───┐─►┌───┐        ┌───┐─►┌───┐─►┌───┐
     │ H │                  │ H₁│  │ H₂│        │ H₁│  │ H₂│  │ H₃│
     └───┘                  └───┘  └───┘        └───┘  └───┘  └───┘
       ▲                      ▲                   ▲       ▲       ▲
       │                      │                   │       │       │
     ┌───┐                  ┌───┐               ┌───┐   ┌───┐   ┌───┐
     │ x │                  │ x │               │x₁ │   │x₂ │   │x₃ │
     └───┘                  └───┘               └───┘   └───┘   └───┘

4. Many-to-Many (Synchronous)           5. Many-to-Many (Encoder-Decoder / Seq2Seq)
     ┌───┐   ┌───┐   ┌───┐                            ┌───┐   ┌───┐
     │y₁ │   │y₂ │   │y₃ │                            │y₁ │   │y₂ │
     └───┘   └───┘   └───┘                            └───┘   └───┘
       ▲       ▲       ▲                                ▲       ▲
       │       │       │                                │       │
     ┌───┐─►┌───┐─►┌───┐                    ┌───┐─►┌───┐─►┌───┐─►┌───┐
     │ H₁│  │ H₂│  │ H₃│                    │ H₁│  │ H₂│  │ H₃│  │ H₄│
     └───┘  └───┘  └───┘                    └───┘  └───┘  └───┘  └───┘
       ▲       ▲       ▲                      ▲       ▲
       │       │       │                      │       │
     ┌───┐   ┌───┐   ┌───┐                  ┌───┐   ┌───┐
     │x₁ │   │x₂ │   │x₃ │                  │x₁ │   │x₂ │
     └───┘   └───┘   └───┘                  └───┘   └───┘
                                            [ Encoder ]     [ Decoder ]
```

### Detailed Comparison Table

| Topology | Input Size | Output Size | Loss Computation | Real-World Application | Used in Project 07? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. One-to-One** | Single `(1, D)` | Single `(1, K)` | Standard loss: `L = loss(y, ŷ)` | Traditional tabular or vector classification (ANN baseline) | No (Baseline) |
| **2. One-to-Many** | Single `(1, D)` | Sequence `(T_y, K)` | Sum over output steps: `L = sum_t loss(y_t, ŷ_t)` | Image Captioning (Image -> sequence of words), Music generation | No |
| **3. Many-to-One** | Sequence `(T_x, D)` | Single `(1, K)` | Computed only at final step: `L = loss(y_T, ŷ_T)` | Sentiment Analysis (Sentence -> Positive/Negative), Action recognition from video | Dedicated to `08-sentiment-rnn` |
| **4. Many-to-Many (Synchronous)** | Sequence `(T, D)` | Sequence `(T, K)` (Same Length `T`) | Sum over all timesteps: `L = sum_t loss(y_t, ŷ_t)` | **Character-Level Language Modeling**, Part-of-Speech tagging, Video frame classification | **Yes (Chosen for Project 07)** |
| **5. Many-to-Many (Seq2Seq)** | Sequence `(T_x, D)` | Sequence `(T_y, K)` (Different Lengths) | Sum over decoder steps: `L = sum_t loss(y_t, ŷ_t)` | Machine Translation (English -> French), Text Summarization, Chatbots | Advanced (Encoder-Decoder) |

> [!IMPORTANT]
> **Why Many-to-Many (Synchronous) is the Perfect Foundation for Project 07**:
> 1. It exercises the **complete recurrent chain** at every timestep: every step computes pre-activations `z_t`, non-linear hidden activations `h_t`, logits `o_t`, softmax probabilities `y_t`, and cross-entropy loss `L_t`.
> 2. Backpropagation Through Time (BPTT) flows gradients from **every output step** simultaneously, accumulating shared weight gradients across time.

---

## Toy Implementation Use Case: Character-Level Next-Step Prediction ("hello")

To prove that the RNN works and to make every calculation verifiable on paper, we implement the classic **Character-Level Language Model on the word `"hello"`**.

### 1. The Core Task
Given a prefix sequence of characters, the network must predict the immediate next character at every timestep:
- Input sequence: `"hell"`
- Target sequence: `"ello"`

```text
Timestep 1: Input 'h' ──► Target 'e'
Timestep 2: Input 'e' ──► Target 'l'
Timestep 3: Input 'l' ──► Target 'l'
Timestep 4: Input 'l' ──► Target 'o'
```

### 2. Vocabulary & One-Hot Encoding
The unique character vocabulary extracted from `"hello"` is:

```text
Vocabulary: {'h', 'e', 'l', 'o'}  (Vocab size V = 4)
```

Mapping characters to unique indices and one-hot vectors (`D = 4`):

```text
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
idx_to_char = {0: 'h', 1: 'e', 2: 'l', 3: 'o'}

'h' -> [1, 0, 0, 0]
'e' -> [0, 1, 0, 0]
'l' -> [0, 0, 1, 0]
'o' -> [0, 0, 0, 1]
```

### 3. The Collision Paradox: Why Feedforward ANNs Fundamentally Fail

Consider timesteps 3 and 4:
- At Timestep 3: Input is `'l'` (`[0, 0, 1, 0]`) -> Expected Output is `'l'` (`[0, 0, 1, 0]`).
- At Timestep 4: Input is `'l'` (`[0, 0, 1, 0]`) -> Expected Output is `'o'` (`[0, 0, 0, 1]`).

In a standard feedforward ANN (or MLP):

```math
P = \mathrm{softmax}(x W + b)
```

Because the input vector `x = [0, 0, 1, 0]` is **identical** at both timesteps:
- The feedforward ANN is forced to compute the **exact same logits** and **exact same probabilities** for both steps!
- An ANN has zero temporal context; it cannot distinguish whether `'l'` was preceded by `'e'` (producing next letter `'l'`) or preceded by another `'l'` (producing next letter `'o'`).
- This creates an unsolvable mathematical collision (a 1-to-many mapping on identical inputs).

#### How the RNN Resolves the Collision with Hidden State Memory:

In an RNN, the hidden state at timestep 3 incorporates memory of `'e'`:

```math
h_3 = \tanh(x_3 W_{xh} + h_2 W_{hh} + b_h)
```
*(contains context of `h_2` which processed `'e'`)*

Whereas the hidden state at timestep 4 incorporates memory of `'l'`:

```math
h_4 = \tanh(x_4 W_{xh} + h_3 W_{hh} + b_h)
```
*(contains context of `h_3` which processed `'l'`)*

Even though inputs `x_3` and `x_4` are identical (`[0, 0, 1, 0]`), the prior states differ (`h_2 != h_3`), so:

```math
h_3 \ne h_4 \implies o_3 \ne o_4 \implies \hat{y}_3 \ne \hat{y}_4
```

The RNN successfully predicts `'l'` at step 3 and `'o'` at step 4!

---

### 4. Concrete Tensor Dimensions for the Toy Problem

For the `"hello"` character model:
- Batch size: `B = 1`
- Sequence length: `T = 4` (timesteps: `['h', 'e', 'l', 'l']`)
- Input feature dimension: `D = 4` (one-hot vector size)
- Hidden state dimension: `H = 3` (number of recurrent memory neurons)
- Output classes: `K = 4` (vocabulary size)

#### Learnable Parameter Budget:

```text
W_xh : (D, H) = (4, 3) -> 12 weights
W_hh : (H, H) = (3, 3) ->  9 weights
b_h  : (1, H) = (1, 3) ->  3 biases
W_hy : (H, K) = (3, 4) -> 12 weights
b_y  : (1, K) = (1, 4) ->  4 biases
----------------------------------------
Total learnable parameters = 40 parameters
```

> [!TIP]
> A parameter budget of just **40 scalars** allows us to calculate and verify every single weight update, matrix dot product, and BPTT gradient step by hand down to 6 decimal places.

---

### 5. Sequence Cross-Entropy Loss

At each timestep `t in {1, 2, 3, 4}`, the categorical cross-entropy loss is evaluated against true one-hot target `Y_t`:

```math
L_t = - \sum_{k=0}^{K-1} Y_{t, k} \ln(\hat{y}_{t, k} + \epsilon)
```

The total sequence loss across the entire word is the sum across all 4 timesteps:

```math
L_{\mathrm{seq}} = \sum_{t=1}^{T} L_t = L_1 + L_2 + L_3 + L_4
```

---

## Parameters and Tensor Shapes

For a batch of sequences processed by a Simple RNN:

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `X` | `(B, T, D)` | `X in R^(B x T x D)` | Input sequence batch (`B`: batch size, `T`: timesteps, `D`: input features) |
| `x_t` | `(B, D)` | `x_t in R^(B x D)` | Input slice at timestep `t` |
| `h_prev` | `(B, H)` | `h_{t-1} in R^(B x H)` | Prior hidden state (`H`: hidden units / state capacity) |
| `W_xh` | `(D, H)` | `W_xh in R^(D x H)` | Input-to-hidden projection weights |
| `W_hh` | `(H, H)` | `W_hh in R^(H x H)` | Hidden-to-hidden recurrent weights |
| `b_h` | `(1, H)` | `b_h in R^(1 x H)` | Hidden state bias offset |
| `h_t` | `(B, H)` | `h_t in R^(B x H)` | Current hidden state after tanh activation |
| `W_hy` | `(H, K)` | `W_hy in R^(H x K)` | Hidden-to-output projection weights (`K`: output classes) |
| `b_y` | `(1, K)` | `b_y in R^(1 x K)` | Output classification bias offset |
| `logits` | `(B, K)` | `o_t in R^(B x K)` | Raw class logits before probability normalization |
| `probs` | `(B, K)` | `y_t in R^(B x K)` | Normalized probabilities summing to 1.0 across classes |

---

## Project Structure

```text
07-rnn-from-scratch/
├── README.md                 # Architecture overview, intuition, and implementation guide
├── src/
│   └── rnn.py                # Pure NumPy RNN cell, forward loop, BPTT, and training
└── steps/
    └── step-01-mathematics.md # Theoretical proofs, matrix calculus, and BPTT derivation
```

---

## Learning Workflow

Following the repository's 10-step sequence:

| Step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Mathematics | Implemented | `steps/step-01-mathematics.md`: recurrence, tensor shapes, loss, BPTT derivation, and gradient pathology |
| 2. NumPy implementation | Implemented | `src/rnn.py`: forward pass, BPTT, gradient clipping, SGD, prediction, generation, and gradient checking |
| 3. Understand forward pass | Implemented | Deterministic `"hell"` walkthrough and cached hidden states across all `T` timesteps |
| 4. Understand loss | Implemented | Many-to-many sequence cross-entropy for `"hell" -> "ello"` |
| 5. Derive gradients | Implemented | Output-layer and recurrent-parameter gradients derived in the mathematics step |
| 6. Implement backpropagation | Implemented | Full BPTT with gradients accumulated over shared parameters |
| 7. Train model | Implemented | Full-batch training reduces loss and learns the `"hello"` transition sequence |
| 8. Debug & analyze | Partially implemented | Global-norm clipping and analytical-vs-numerical gradient checking; no separate long-sequence pathology experiment yet |
| 9. PyTorch implementation | Pending | Equivalent model using `torch.nn.RNN` and `torch.nn.Linear` |
| 10. Compare results | Pending | Validation of weights, hidden states, loss curves, and runtime |

---

## Implemented in This Project

The current project implements and verifies a vanilla many-to-many RNN with
pure NumPy. It includes:

- vectors, matrices, matrix multiplication, transposes, tensor shapes, and bias broadcasting
- input-to-hidden, hidden-to-hidden, and hidden-to-output dense projections
- `tanh` hidden activation and its derivative
- one-hot character inputs, batch dimensions, timesteps, and shared parameters across time
- a complete forward pass: hidden states, logits, stable softmax, predictions, and sequence cross-entropy
- BPTT: output gradients, gradients through `tanh`, recurrent gradient flow, and accumulated shared-parameter gradients
- global-norm gradient clipping, gradient-descent updates, and a repeatable training loop
- teacher-forced prediction, autoregressive generation, and numerical gradient checking with finite differences

## Planned Extensions

These are intentionally outside the current Simple-RNN implementation:

- ReLU and sigmoid activation lessons and their derivatives
- many-to-one/final-timestep-only loss configuration
- explicit gradients with respect to the input sequence, `dL/dx_t`
- a numerical long-sequence experiment for vanishing and exploding gradients
- a PyTorch implementation using `torch.nn.RNN`
- a NumPy-versus-PyTorch comparison of results and runtime

---

## Key Takeaways

1. **Temporal Context**: Feedforward networks have zero memory of previous inputs. RNNs maintain an internal hidden state `h_t` that persists context across sequential timesteps.
2. **Shared Recurrence**: The recurrent weights `W_hh` and input weights `W_xh` are tied across time, enabling the model to generalize across variable-length sequences.
3. **Backpropagation Through Time (BPTT)**: Gradients flow backward across the unrolled sequence chain, accumulating parameter updates across every timestep where the weights were reused.
4. **Fundamental Bottleneck**: Long sequences suffer from vanishing/exploding gradients due to repeated multiplications by `W_hh^T`, directly motivating modern gated architectures like **LSTM** (`09-lstm`) and **GRU** (`10-gru`).

---

## Next Steps

1. Add the planned activation-function and final-timestep-loss lessons.
2. Implement and verify explicit input gradients, `dL/dx_t`.
3. Create a long-sequence experiment that measures vanishing and exploding gradients.
4. Build the equivalent PyTorch RNN and compare outputs, training curves, and runtime.
