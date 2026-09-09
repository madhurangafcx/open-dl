# Step 1 — Mathematics for a Recurrent Neural Network (RNN)

Goal: understand the complete mathematics of a Recurrent Neural Network (RNN) from first principles.
This document provides the rigorous theoretical foundations, sequence topologies, temporal geometry,
tensor dimensions, forward calculations, and Backpropagation Through Time (BPTT) matrix calculus
that will be implemented with pure NumPy from scratch.

---

## Q1 — The Sequential Problem & Why Feedforward Networks Fail

In traditional feedforward networks (such as the Artificial Neural Network in `02-ann-from-scratch`),
the model processes inputs independently:

```math
y = \mathrm{softmax}(x W + b)
```

This works for tabular records and static datasets where each sample is self-contained.
However, it fundamentally fails on **sequential data** where **temporal order dictates meaning**.

### The Concrete Toy Problem: Character-Level Next-Step Prediction

To understand the core limitation of feedforward models and how an RNN resolves it, consider training a model on the word:

```text
"hello"
```

We configure the task as predicting the immediate next character at every step:
- **Input sequence**: `'h'` ──► `'e'` ──► `'l'` ──► `'l'`
- **Target sequence**: `'e'` ──► `'l'` ──► `'l'` ──► `'o'`

The unique character vocabulary is:

```text
Vocabulary: {'h', 'e', 'l', 'o'}  (Vocab size V = 4)
```

Each character is encoded as a 4-dimensional one-hot indicator vector:

```text
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}

'h' = [1, 0, 0, 0]
'e' = [0, 1, 0, 0]
'l' = [0, 0, 1, 0]
'o' = [0, 0, 0, 1]
```

---

### The Collision Paradox That Breaks Feedforward ANNs

Examine the training sequence at timesteps 3 and 4:

```text
Timestep 3: Input is 'l' [0, 0, 1, 0] ──► Expected Output is 'l' [0, 0, 1, 0]
Timestep 4: Input is 'l' [0, 0, 1, 0] ──► Expected Output is 'o' [0, 0, 0, 1]
```

In any feedforward architecture (ANN, MLP):

```math
P_t = \mathrm{softmax}(x_t W + b)
```

Because the input vector `x_3` and `x_4` are mathematically identical (`[0, 0, 1, 0]`):

```math
x_3 = x_4 \implies x_3 W + b = x_4 W + b \implies P_3 = P_4
```

A feedforward network is mathematically forced to output the **exact same probability distribution** at both timesteps!
It cannot simultaneously predict `'l'` at step 3 and `'o'` at step 4 because it has **zero internal memory** of what preceded the input.
It cannot distinguish whether `'l'` was preceded by `'e'` (meaning the next letter is `'l'`) or preceded by another `'l'` (meaning the next letter is `'o'`).

---

### The RNN Solution: The Recurrent Hidden State

A Recurrent Neural Network resolves this mathematical paradox by introducing an internal state vector `h_t` (hidden memory)
that updates recurrently over time:

```math
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
```

At timestep 3:
- The hidden state `h_2` holds the historical context that the network just processed `'e'`.
- The new state `h_3 = \tanh(x_3 W_{xh} + h_2 W_{hh} + b_h)` combines current `'l'` with past `'e'`.

At timestep 4:
- The prior hidden state is now `h_3` (which holds the context that the network already processed one `'l'`).
- The new state `h_4 = \tanh(x_4 W_{xh} + h_3 W_{hh} + b_h)` combines current `'l'` with past `'l'`.

Because `h_2 != h_3`:

```math
h_3 \ne h_4 \implies o_3 \ne o_4 \implies \hat{y}_3 \ne \hat{y}_4
```

The RNN successfully distinguishes identical inputs by conditioning its decision on the **entire historical path**!

---

## Q2 — Network Architecture & Sequence Topologies

In recurrent neural networks, inputs and outputs can be configured across time in **5 distinct sequence topologies**:

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

### Topology Comparison

| Topology | Input Dimensions | Output Dimensions | Loss Computation | Canonical Application | Used in Project 07? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **One-to-One** | Single `(B, D)` | Single `(B, K)` | Standard loss: `L = loss(y, ŷ)` | Traditional tabular / image classification | No (ANN/CNN) |
| **One-to-Many** | Single `(B, D)` | Sequence `(B, T_y, K)` | Sum over output steps: `L = sum_t L_t` | Image Captioning (Image -> words) | No |
| **Many-to-One** | Sequence `(B, T_x, D)` | Single `(B, K)` | Computed only at step `T`: `L = L_T` | Sentiment Analysis (Sentence -> Sentiment) | Dedicated to Project 08 |
| **Many-to-Many (Synch)** | Sequence `(B, T, D)` | Sequence `(B, T, K)` | Sum over all timesteps: `L = sum_t L_t` | **Character-Level Language Modeling**, POS Tagging | **Yes (Chosen for Project 07)** |
| **Many-to-Many (Seq2Seq)** | Sequence `(B, T_x, D)` | Sequence `(B, T_y, K)` | Sum over decoder steps: `L = sum_t L_t` | Machine Translation, Text Summarization | Advanced |

---

### The Unrolled Recurrent Architecture Across 4 Timesteps

Unrolling the recurrent loop across the 4 timesteps of `"hell"` reveals the complete computational graph:

```text
x₁ ('h')          x₂ ('e')          x₃ ('l')          x₄ ('l')
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐
│ W_xh │          │ W_xh │          │ W_xh │          │ W_xh │
└──────┘          └──────┘          └──────┘          └──────┘
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
  (+) ◄── W_hh ───── (+) ◄── W_hh ───── (+) ◄── W_hh ───── (+)
   ▲   (from h₀=0)   ▲   (from h₁)     ▲   (from h₂)     ▲   (from h₃)
   │                 │                 │                 │
┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐
│ tanh │          │ tanh │          │ tanh │          │ tanh │
└──────┘          └──────┘          └──────┘          └──────┘
   │                 │                 │                 │
   ├──► h₁ ──────────┼──► h₂ ──────────┼──► h₃ ──────────┼──► h₄
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐
│ W_hy │          │ W_hy │          │ W_hy │          │ W_hy │
└──────┘          └──────┘          └──────┘          └──────┘
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
  (+) + b_y         (+) + b_y         (+) + b_y         (+) + b_y
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
Logits o₁         Logits o₂         Logits o₃         Logits o₄
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
Softmax           Softmax           Softmax           Softmax
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
Probs ŷ₁          Probs ŷ₂          Probs ŷ₃          Probs ŷ₄
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
Loss L₁ (target 'e')Loss L₂ (target 'l')Loss L₃ (target 'l')Loss L₄ (target 'o')
```

> [!IMPORTANT]
> **Parameter Sharing Invariant**:
> Notice that `W_xh`, `W_hh`, `W_hy`, `b_h`, and `b_y` appear at **every single timestep**.
> They are not separate layers; they are the exact same matrices reused across time.
> Any update to `W_hh` during backpropagation must sum gradients across all timesteps where `W_hh` was active.

---

## Q3 — Parameters, Tensor Shapes, and Parameter Budget

For our minimal character-level sequence model:
- Batch size: `B = 1`
- Sequence length: `T = 4` (`['h', 'e', 'l', 'l']`)
- Input feature dimension: `D = 4` (one-hot character vocabulary size)
- Hidden state dimension: `H = 3` (recurrent memory capacity)
- Output classes: `K = 4` (vocabulary size)

### Parameter and Tensor Shape Table

| Parameter / Tensor | Notation | Shape | Parameter Count | Mathematical Role |
| :--- | :--- | :--- | :--- | :--- |
| Input Sequence | `X` | `(1, 4, 4)` | 0 | Batch of one-hot character vectors |
| Input at step `t` | `x_t` | `(1, 4)` | 0 | Slice corresponding to current character |
| Input-to-Hidden Weights | `W_xh` | `(4, 3)` | 4 × 3 = 12 | Linear projection from input features to hidden state |
| Recurrent Weights | `W_hh` | `(3, 3)` | 3 × 3 = 9 | Temporal transition matrix carrying memory across steps |
| Hidden Bias | `b_h` | `(1, 3)` | 3 | Additive bias offset for hidden pre-activation |
| Pre-activation at step `t` | `z_t` | `(1, 3)` | 0 | Linear combination `x_t @ W_xh + h_{t-1} @ W_hh + b_h` |
| Hidden State at step `t` | `h_t` | `(1, 3)` | 0 | Activated recurrent memory `tanh(z_t)` |
| Hidden-to-Output Weights | `W_hy` | `(3, 4)` | 3 × 4 = 12 | Linear projection from hidden state to class logits |
| Output Bias | `b_y` | `(1, 4)` | 4 | Additive class bias offsets |
| Unnormalized Logits | `o_t` | `(1, 4)` | 0 | Raw unnormalized scores `h_t @ W_hy + b_y` |
| Output Probabilities | `ŷ_t` | `(1, 4)` | 0 | Normalized probability distribution `softmax(o_t)` |
| Target at step `t` | `Y_t` | `(1, 4)` | 0 | Ground-truth one-hot label vector |
| Step Loss | `L_t` | `(1,)` | 0 | Categorical cross-entropy scalar at step `t` |

#### Total Learnable Parameter Count:

```math
N_{\mathrm{params}} = (D \cdot H) + (H \cdot H) + H + (H \cdot K) + K
```

Substituting `D = 4`, `H = 3`, `K = 4`:

```math
N_{\mathrm{params}} = (4 \cdot 3) + (3 \cdot 3) + 3 + (3 \cdot 4) + 4 = 12 + 9 + 3 + 12 + 4 = 40 \text{ parameters}
```

A compact parameter budget of **40 scalars** allows us to hand-calculate every matrix dot product,
intermediate activation, and BPTT gradient update down to 6 decimal places.

---

## Q4 — The Activation Function: tanh (Hyperbolic Tangent)

In standard RNNs, the recurrent state activation is universally chosen to be the **hyperbolic tangent (`tanh`)**:

```math
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}
```

```text
               tanh(z)
                  ▲
               1.0│            . ─── 1.0 (asymptotic limit)
                  │         .
                  │       /
                  │      /
──────────────────┼─────/───────────► z
                 /│
                / │
               .  │
 -1.0 ────────    │-1.0
                  ▼
```

### Properties of tanh:
1. **Bounded Output Range**:
   ```math
   \tanh(z) \in (-1.0, \, +1.0)
   ```
2. **Zero-Centered**:
   ```math
   \tanh(0) = 0, \quad \tanh(-z) = -\tanh(z)
   ```
3. **Smooth Non-Linearity**: Infinitely differentiable everywhere.

### Why tanh for Recurrent Hidden States (Instead of ReLU or Sigmoid)?
1. **Bounding Repeated Multiplications Through Time**:
   In an RNN, the hidden state is repeatedly transformed across `T` timesteps:
   `h_t ~ f(... f(f(h_0)))`.
   - If **ReLU** is used: `ReLU(z) = max(0, z)` has an unbounded positive range `[0, inf)`. Repeated matrix multiplications by `W_hh` can cause the activations to explode to infinity within a few timesteps.
   - If **Sigmoid** is used: `sigmoid(z) in (0, 1)` is strictly positive (non-zero-centered), which introduces systematic positive bias drift in the hidden state representation.
   - **tanh** strictly constrains all hidden state values within `(-1, +1)` while preserving both positive and negative representations centered around zero.

### Derivative of tanh:

```math
\frac{\mathrm{d}\tanh(z)}{\mathrm{d}z} = 1 - \tanh^2(z)
```

#### Derivation:
Using the quotient rule on `tanh(z) = sinh(z) / cosh(z)`:

```math
\frac{\mathrm{d}}{\mathrm{d}z} \left( \frac{\sinh(z)}{\cosh(z)} \right) = \frac{\cosh(z)\cosh(z) - \sinh(z)\sinh(z)}{\cosh^2(z)} = \frac{\cosh^2(z) - \sinh^2(z)}{\cosh^2(z)}
```

Recalling the fundamental hyperbolic identity `cosh^2(z) - sinh^2(z) = 1`:

```math
\frac{\mathrm{d}\tanh(z)}{\mathrm{d}z} = \frac{1}{\cosh^2(z)} = 1 - \frac{\sinh^2(z)}{\cosh^2(z)} = 1 - \tanh^2(z)
```

> [!TIP]
> In code, once the forward hidden state `h_t = tanh(z_t)` is computed and cached, its derivative requires **no expensive exponentials**:
> ```python
> dtanh = 1.0 - h_t ** 2
> ```

---

## Q5 — Concrete Numerical Setup (Matching verify_rnn.py)

To achieve absolute mathematical transparency, we establish concrete numerical values for all parameters.
These exact values match the automated verification script [`verify_rnn.py`](file:///Users/pasan/.gemini/antigravity-ide/brain/d3b699e8-4410-4b8c-b2cb-728045b5ca32/scratch/verify_rnn.py).

### 1. Initial State:
The initial hidden state at `t = 0` is the zero vector:

```math
h_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{1 \times 3}
```

### 2. Input-to-Hidden Weights `W_xh` (4 × 3):

```text
W_xh = np.array([
    [ 0.5, -0.2,  0.1],   # connects 'h' [1,0,0,0] to hidden
    [-0.1,  0.4, -0.3],   # connects 'e' [0,1,0,0] to hidden
    [ 0.2, -0.1,  0.5],   # connects 'l' [0,0,1,0] to hidden
    [ 0.3,  0.2, -0.4],   # connects 'o' [0,0,0,1] to hidden
], dtype=np.float32)
```

### 3. Hidden-to-Hidden Recurrent Weights `W_hh` (3 × 3):

```text
W_hh = np.array([
    [ 0.1,  0.3, -0.1],
    [-0.2,  0.1,  0.4],
    [ 0.3, -0.2,  0.2],
], dtype=np.float32)
```

### 4. Hidden Bias `b_h` (1 × 3):

```text
b_h = np.array([[0.1, -0.1, 0.0]], dtype=np.float32)
```

### 5. Hidden-to-Output Weights `W_hy` (3 × 4):

```text
W_hy = np.array([
    [ 0.4, -0.3,  0.5, -0.1],
    [-0.2,  0.5, -0.1,  0.3],
    [ 0.1, -0.4,  0.2,  0.5],
], dtype=np.float32)
```

### 6. Output Bias `b_y` (1 × 4):

```text
b_y = np.array([[0.0, 0.1, -0.1, 0.2]], dtype=np.float32)
```

---

## Q6 — Forward Pass: Step-by-Step Numerical Walkthrough Across All 4 Timesteps

We now trace every mathematical operation step-by-step from `t = 1` to `t = 4`.

At every timestep `t`:
1. **Pre-activation**:
   ```math
   z_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h
   ```
2. **Hidden state**:
   ```math
   h_t = \tanh(z_t)
   ```
3. **Logits**:
   ```math
   o_t = h_t W_{hy} + b_y
   ```
4. **Softmax probabilities**:
   ```math
   \hat{y}_t = \mathrm{softmax}(o_t)
   ```
5. **Categorical cross-entropy loss**:
   ```math
   L_t = -\ln(\hat{y}_{t, \mathrm{target}} + \epsilon)
   ```

---

### Timestep 1 (t = 1): Input 'h', Target 'e'

#### 1. Input and Prior State:
```text
x_1 = [1.0, 0.0, 0.0, 0.0]  ('h')
h_0 = [0.0, 0.0, 0.0]
```

#### 2. Pre-activation `z_1`:
Because `x_1` is one-hot at index 0, `x_1 @ W_xh` simply selects row 0 of `W_xh`:
```text
x_1 @ W_xh = [0.5, -0.2, 0.1]
h_0 @ W_hh = [0.0, 0.0, 0.0]
b_h        = [0.1, -0.1, 0.0]
```

Summing the terms:
```math
z_1 = [0.5 + 0.0 + 0.1, \quad -0.2 + 0.0 - 0.1, \quad 0.1 + 0.0 + 0.0] = [0.6, \, -0.3, \, 0.1]
```

#### 3. Hidden State `h_1 = tanh(z_1)`:
```math
h_{1, 0} = \tanh(0.6) = 0.537050
```
```math
h_{1, 1} = \tanh(-0.3) = -0.291313
```
```math
h_{1, 2} = \tanh(0.1) = 0.099668
```
```text
h_1 = [0.537050, -0.291313, 0.099668]
```

#### 4. Output Logits `o_1 = h_1 @ W_hy + b_y`:
Computing dot product of `h_1` with each column of `W_hy`:
```text
o_{1, 0} = (0.537050 * 0.4) + (-0.291313 * -0.2) + (0.099668 * 0.1) + 0.0 = 0.283049
o_{1, 1} = (0.537050 * -0.3) + (-0.291313 * 0.5) + (0.099668 * -0.4) + 0.1 = -0.246638
o_{1, 2} = (0.537050 * 0.5) + (-0.291313 * -0.1) + (0.099668 * 0.2) - 0.1 = 0.217590
o_{1, 3} = (0.537050 * -0.1) + (-0.291313 * 0.3) + (0.099668 * 0.5) + 0.2 = 0.108735
```
```text
o_1 = [0.283049, -0.246638, 0.217590, 0.108735]
```

#### 5. Softmax Probabilities `p_1`:
Max logit `max(o_1) = 0.283049`.
Subtracting max for numerical stability:
```text
shifted = [0.0, -0.529687, -0.065459, -0.174314]
exp     = [1.0, 0.588790, 0.936636, 0.839972]
sum_exp = 3.365398
```
Dividing by sum:
```text
p_1 = [0.297136, 0.174951, 0.278309, 0.249604]
```

#### 6. Step Loss `L_1`:
Target is `'e'` (index 1), so `Y_1 = [0, 1, 0, 0]`:
```math
L_1 = -\ln(p_{1, 1}) = -\ln(0.174951) = 1.743252
```

---

### Timestep 2 (t = 2): Input 'e', Target 'l'

#### 1. Input and Prior State:
```text
x_2 = [0.0, 1.0, 0.0, 0.0]  ('e')
h_1 = [0.537050, -0.291313, 0.099668]
```

#### 2. Pre-activation `z_2`:
`x_2 @ W_xh` selects row 1 of `W_xh`:
```text
x_2 @ W_xh = [-0.1, 0.4, -0.3]
```
Now compute the recurrent transition `h_1 @ W_hh`:
```text
(h_1 @ W_hh)[0] = (0.537050 * 0.1) + (-0.291313 * -0.2) + (0.099668 * 0.3) = 0.141868
(h_1 @ W_hh)[1] = (0.537050 * 0.3) + (-0.291313 * 0.1) + (0.099668 * -0.2) = 0.112050
(h_1 @ W_hh)[2] = (0.537050 * -0.1) + (-0.291313 * 0.4) + (0.099668 * 0.2) = -0.150296
```
Adding `x_2 @ W_xh`, `h_1 @ W_hh`, and `b_h = [0.1, -0.1, 0.0]`:
```math
z_{2, 0} = -0.1 + 0.141868 + 0.1 = 0.141868
```
```math
z_{2, 1} = 0.4 + 0.112050 - 0.1 = 0.412050
```
```math
z_{2, 2} = -0.3 - 0.150296 + 0.0 = -0.450296
```
```text
z_2 = [0.141868, 0.412050, -0.450296]
```

#### 3. Hidden State `h_2 = tanh(z_2)`:
```text
h_2 = [tanh(0.141868), tanh(0.412050), tanh(-0.450296)]
    = [0.140924, 0.390212, -0.422143]
```

#### 4. Output Logits `o_2 = h_2 @ W_hy + b_y`:
```text
o_2 = [-0.063887, 0.421686, -0.152988, 0.091900]
```

#### 5. Softmax Probabilities `p_2`:
Max logit is `0.421686`:
```text
p_2 = [0.212385, 0.345148, 0.194280, 0.248188]
```

#### 6. Step Loss `L_2`:
Target is `'l'` (index 2), so `Y_2 = [0, 0, 1, 0]`:
```math
L_2 = -\ln(p_{2, 2}) = -\ln(0.194280) = 1.638457
```

---

### Timestep 3 (t = 3): Input 'l', Target 'l'

#### 1. Input and Prior State:
```text
x_3 = [0.0, 0.0, 1.0, 0.0]  ('l')
h_2 = [0.140924, 0.390212, -0.422143]
```

#### 2. Pre-activation `z_3`:
`x_3 @ W_xh` selects row 2 of `W_xh`:
```text
x_3 @ W_xh = [0.2, -0.1, 0.5]
```
Recurrent contribution `h_2 @ W_hh`:
```text
h_2 @ W_hh = [-0.190593, 0.165727, 0.057564]
```
Summing with `b_h = [0.1, -0.1, 0.0]`:
```text
z_3 = [0.2 - 0.190593 + 0.1,  -0.1 + 0.165727 - 0.1,  0.5 + 0.057564 + 0.0]
    = [0.109407, -0.034273, 0.557564]
```

#### 3. Hidden State `h_3 = tanh(z_3)`:
```text
h_3 = [tanh(0.109407), tanh(-0.034273), tanh(0.557564)]
    = [0.108973, -0.034260, 0.506168]
```

#### 4. Output Logits `o_3 = h_3 @ W_hy + b_y`:
```text
o_3 = [0.101058, -0.152289, 0.059146, 0.431909]
```

#### 5. Softmax Probabilities `p_3`:
```text
p_3 = [0.242289, 0.188064, 0.232344, 0.337303]
```

#### 6. Step Loss `L_3`:
Target is `'l'` (index 2), so `Y_3 = [0, 0, 1, 0]`:
```math
L_3 = -\ln(p_{3, 2}) = -\ln(0.232344) = 1.459537
```

---

### Timestep 4 (t = 4): Input 'l', Target 'o'

#### 1. Input and Prior State:
```text
x_4 = [0.0, 0.0, 1.0, 0.0]  ('l' -- exact same vector as x_3!)
h_3 = [0.108973, -0.034260, 0.506168]
```

#### 2. Pre-activation `z_4`:
`x_4 @ W_xh` selects row 2 of `W_xh`:
```text
x_4 @ W_xh = [0.2, -0.1, 0.5]
```
Recurrent contribution `h_3 @ W_hh`:
```text
h_3 @ W_hh = [0.169600, -0.071968, 0.076632]
```
Summing with `b_h`:
```text
z_4 = [0.2 + 0.169600 + 0.1,  -0.1 - 0.071968 - 0.1,  0.5 + 0.076632 + 0.0]
    = [0.469600, -0.271968, 0.576632]
```

#### 3. Hidden State `h_4 = tanh(z_4)`:
```text
h_4 = [tanh(0.469600), tanh(-0.271968), tanh(0.576632)]
    = [0.437876, -0.265455, 0.520213]
```

> [!NOTE]
> **Proof of Disambiguation**:
> Compare the two consecutive hidden states on input `'l'`:
> ```text
> h_3 = [ 0.108973, -0.034260,  0.506168 ]   (after seeing 'e')
> h_4 = [ 0.437876, -0.265455,  0.520213 ]   (after seeing 'l')
> ```
> Despite identical input vectors `x_3 == x_4 == [0, 0, 1, 0]`, the hidden states differ significantly.

#### 4. Output Logits `o_4 = h_4 @ W_hy + b_y`:
```text
o_4 = [0.280263, -0.372176, 0.249526, 0.336683]
```

#### 5. Softmax Probabilities `p_4`:
```text
p_4 = [0.281805, 0.146757, 0.273275, 0.298162]
```

#### 6. Step Loss `L_4`:
Target is `'o'` (index 3), so `Y_4 = [0, 0, 0, 1]`:
```math
L_4 = -\ln(p_{4, 3}) = -\ln(0.298162) = 1.210118
```

---

## Q7 — Total Sequence Loss

In a synchronous Many-to-Many RNN, the total loss across the entire word sequence is the scalar sum of all `T = 4` individual timestep losses:

```math
L_{\mathrm{seq}} = \sum_{t=1}^{T} L_t = L_1 + L_2 + L_3 + L_4
```

Substituting our calculated step losses:

```math
L_{\mathrm{seq}} = 1.743252 + 1.638457 + 1.459537 + 1.210118 = 6.051364
```

Average loss per character:

```math
\bar{L} = \frac{L_{\mathrm{seq}}}{T} = \frac{6.051364}{4} = 1.512841
```

---

## Q8 — Computational Graph & Temporal Chain Rule

In feedforward backpropagation, error gradients flow straight back through layers from output to input.
In a Recurrent Neural Network, error gradients must flow **both across layers and backward through time**. This is called **Backpropagation Through Time (BPTT)**.

### The Two Gradient Flow Paths to Hidden State `h_t`:

At any intermediate timestep `t`, the hidden state vector `h_t` influences the loss function along **two separate pathways**:

```text
               Path 1 (Immediate Output):
               h_t ──► W_hy ──► o_t ──► p_t ──► Loss L_t
               
               Path 2 (Future Time Evolution):
               h_t ──► W_hh ──► z_{t+1} ──► h_{t+1} ──► ... ──► Loss L_{>t}
```

By the multivariable chain rule, the total derivative of the sequence loss with respect to `h_t` is the **sum of both paths**:

```math
\frac{\partial L}{\partial h_t} = \underbrace{\frac{\partial L_t}{\partial h_t}}_{\text{Path 1: Current Output}} + \underbrace{\frac{\partial L_{>t}}{\partial h_t}}_{\text{Path 2: Future Timesteps}}
```

---

## Q9 — Backpropagation Through Time (BPTT): First-Principles Matrix Calculus

We derive every parameter gradient step-by-step using matrix calculus.

### 1. Output Layer Gradient (`∂L_t / ∂o_t`)
For categorical cross-entropy with Softmax, the derivative with respect to logits simplifies to:

```math
\delta_{o_t} = \frac{\partial L_t}{\partial o_t} = \hat{y}_t - Y_t \in \mathbb{R}^{1 \times K}
```

### 2. Output Projection Parameters (`W_hy`, `b_y`)
Because `o_t = h_t @ W_hy + b_y`:

```math
\frac{\partial L_t}{\partial W_{hy}} = h_t^T \delta_{o_t}, \quad \frac{\partial L_t}{\partial b_y} = \delta_{o_t}
```

Since `W_hy` and `b_y` are shared across all timesteps, we sum across `t = 1 ... T`:

```math
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} h_t^T \delta_{o_t} \in \mathbb{R}^{H \times K}
```

```math
\frac{\partial L}{\partial b_y} = \sum_{t=1}^{T} \delta_{o_t} \in \mathbb{R}^{1 \times K}
```

---

### 3. Total Gradient at Hidden State (`∂L / ∂h_t`)
- From Path 1 (output):
  ```math
  \delta_{o_t} W_{hy}^T
  ```
- From Path 2 (next timestep `t + 1`):
  ```math
  \delta_{z_{t+1}} W_{hh}^T
  ```

Combining both paths:

```math
\delta_{h_t} = \frac{\partial L}{\partial h_t} = \delta_{o_t} W_{hy}^T + \delta_{z_{t+1}} W_{hh}^T \in \mathbb{R}^{1 \times H}
```

> [!NOTE]
> At the final timestep `t = T = 4`, there is no future timestep, so the incoming future gradient is zero (`\delta_{z_{T+1}} = \mathbf{0}`), which yields `\delta_{h_T} = \delta_{o_T} W_{hy}^T`.

---

### 4. Backpropagating Through tanh (`∂L / ∂z_t`)
Using the derivative of `tanh` from Q4:

```math
\delta_{z_t} = \frac{\partial L}{\partial z_t} = \delta_{h_t} \odot (1 - h_t^2) \in \mathbb{R}^{1 \times H}
```

Where `\odot` represents the element-wise Hadamard product (element-wise multiplication).

---

### 5. Gradients with Respect to Shared Recurrent Parameters (`W_xh`, `W_hh`, `b_h`)
Recall the forward pre-activation equation:

```math
z_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h
```

Taking partial derivatives:
1. **Input-to-hidden weights `W_xh`**:
   ```math
   \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} x_t^T \delta_{z_t} \in \mathbb{R}^{D \times H}
   ```
2. **Hidden-to-hidden recurrent weights `W_hh`**:
   ```math
   \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} h_{t-1}^T \delta_{z_t} \in \mathbb{R}^{H \times H}
   ```
3. **Hidden bias `b_h`**:
   ```math
   \frac{\partial L}{\partial b_h} = \sum_{t=1}^{T} \delta_{z_t} \in \mathbb{R}^{1 \times H}
   ```

---

## Q10 — Step-by-Step Numerical BPTT Walkthrough

We now run the backward equations in reverse chronological order from `t = 4` down to `t = 1`.

### Backward Step t = 4 (Final Timestep):

1. **Logit error**:
   ```text
   d_o4 = p_4 - Y_4 = [0.281805, 0.146757, 0.273275, 0.298162] - [0, 0, 0, 1]
        = [0.281805, 0.146757, 0.273275, -0.701838]
   ```
2. **Gradient to h_4**:
   Since `t = 4` is the final step, no gradient arrives from `t = 5`:
   ```math
   d\_h4 = d\_o4 @ W_{hy}^T = [0.275517, -0.220861, -0.326786]
   ```
3. **Gradient through tanh**:
   ```math
   1 - h_4^2 = 1 - [0.437876^2, (-0.265455)^2, 0.520213^2] = [0.808265, 0.929534, 0.729378]
   ```
   ```math
   d\_z4 = d\_h4 \odot (1 - h_4^2) = [0.222690, -0.205298, -0.238351]
   ```
4. **Gradient passed back to h_3**:
   ```math
   d\_h\_next = d\_z4 @ W_{hh}^T = [-0.015485, -0.160408, 0.060197]
   ```

---

### Backward Step t = 3:

1. **Logit error**:
   ```text
   d_o3 = p_3 - Y_3 = [0.242289, 0.188064, 0.232344, 0.337303] - [0, 0, 1, 0]
        = [0.242289, 0.188064, -0.767656, 0.337303]
   ```
2. **Gradient to h_3 (Sum of both paths)**:
   ```math
   d\_h3 = (d\_o3 @ W_{hy}^T) + d\_h\_next
         = [-0.377062, 0.223531, -0.035877] + [-0.015485, -0.160408, 0.060197]
         = [-0.392547, 0.063123, 0.024320]
   ```
3. **Gradient through tanh**:
   ```math
   d\_z3 = d\_h3 \odot (1 - h_3^2) = [-0.387886, 0.063049, 0.018089]
   ```
4. **Gradient passed back to h_2**:
   ```math
   d\_h\_next = d\_z3 @ W_{hh}^T = [-0.021683, 0.091118, -0.125358]
   ```

---

### Backward Step t = 2:

1. **Logit error**:
   ```text
   d_o2 = p_2 - Y_2 = [0.212385, 0.345148, 0.194280, 0.248188] - [0, 0, 1, 0]
        = [0.212385, 0.345148, -0.805720, 0.248188]
   ```
2. **Gradient to h_2**:
   ```math
   d\_h2 = (d\_o2 @ W_{hy}^T) + d\_h\_next
         = [-0.446269, 0.285125, -0.153870] + [-0.021683, 0.091118, -0.125358]
         = [-0.467952, 0.376243, -0.279228]
   ```
3. **Gradient through tanh**:
   ```math
   d\_z2 = d\_h2 \odot (1 - h_2^2) = [-0.458659, 0.318954, -0.229469]
   ```
4. **Gradient passed back to h_1**:
   ```math
   d\_h\_next = d\_z2 @ W_{hh}^T = [0.072767, 0.031840, -0.247282]
   ```

---

### Backward Step t = 1:

1. **Logit error**:
   ```text
   d_o1 = p_1 - Y_1 = [0.297136, 0.174951, 0.278309, 0.249604] - [0, 1, 0, 0]
        = [0.297136, -0.825049, 0.278309, 0.249604]
   ```
2. **Gradient to h_1**:
   ```math
   d\_h1 = (d\_o1 @ W_{hy}^T) + d\_h\_next
         = [0.480564, -0.424902, 0.540197] + [0.072767, 0.031840, -0.247282]
         = [0.553331, -0.393062, 0.292915]
   ```
3. **Gradient through tanh**:
   ```math
   d\_z1 = d\_h1 \odot (1 - h_1^2) = [0.393738, -0.359705, 0.290005]
   ```

---

### Final Accumulated Parameter Gradients:

Summing across all 4 timesteps:

#### 1. Output Projection Gradients (`dW_hy`, `db_y`):
```text
dW_hy = sum(h_t.T @ d_ot)
      = [[ 0.339306, -0.309698,  0.071928, -0.101536],
         [-0.086792,  0.329628, -0.441719,  0.198883],
         [ 0.209196, -0.056395,  0.121466, -0.274267]]

db_y  = sum(d_ot)
      = [[ 1.033615, -0.145080, -1.021792,  0.133257]]
```

#### 2. Recurrent Weight Gradients (`dW_xh`, `dW_hh`, `db_h`):
```text
dW_xh = sum(x_t.T @ d_zt)
      = [[ 0.393738, -0.359705,  0.290005],   # gradient from 'h' at t=1
         [-0.458659,  0.318954, -0.229469],   # gradient from 'e' at t=2
         [-0.165196, -0.142250, -0.220262],   # accumulated from 'l' at t=3 and t=4!
         [ 0.000000,  0.000000,  0.000000]]   # 'o' is never an input, zero gradient

dW_hh = sum(h_{t-1}.T @ d_zt)
      = [[-0.276718,  0.157807, -0.146661],
         [-0.025374, -0.061280,  0.082072],
         [ 0.230748, -0.098741, -0.151152]]

db_h  = sum(d_zt)
      = [[-0.230117, -0.183001, -0.159725]]
```

---

## Q11 — Gradient Pathology: Vanishing and Exploding Gradients

Why do vanilla RNNs struggle to learn dependencies over long sequences (`T > 10`)?

### The Mathematical Proof

Consider the gradient of the loss at timestep `T` with respect to the hidden state at an early timestep `t`:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \frac{\partial h_j}{\partial h_{j-1}}
```

Expanding the single-step Jacobian matrix:

```math
\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}(1 - h_j^2) W_{hh}^T
```

Substituting this into the product chain:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \left( \operatorname{diag}(1 - h_j^2) W_{hh}^T \right)
```

Notice that this is a **repeated matrix product of length `(T - t)`**:

```math
\prod_{j=t+1}^{T} W_{hh}^T \approx \left( W_{hh}^T \right)^{T - t}
```

Let the eigendecomposition of `W_hh^T` be `Q \Lambda Q^{-1}`, where `\lambda_{max}` is the largest eigenvalue:

```math
\left( W_{hh}^T \right)^{T - t} = Q \Lambda^{T - t} Q^{-1}
```

### Two Pathological Regimes:

1. **Vanishing Gradient** (when `\lambda_{max} < 1` and `|tanh'| <= 1`):
   ```math
   \lim_{T - t \to \infty} \Lambda^{T - t} = 0 \implies \frac{\partial L_T}{\partial h_t} \to \mathbf{0}
   ```
   The error signal exponentially decays to zero as it travels backward through time. Early inputs cannot receive gradient updates from late errors, making it impossible to learn long-range temporal dependencies.

2. **Exploding Gradient** (when `\lambda_{max} > 1`):
   ```math
   \lim_{T - t \to \infty} \Lambda^{T - t} = \infty \implies \left\| \frac{\partial L_T}{\partial h_t} \right\| \to \infty
   ```
   The error gradient grows exponentially, causing weight updates `\Delta W = -\eta \nabla L` to overflow to `NaN` and completely destabilize training.

---

### Solutions in Vanilla RNNs:

#### 1. Gradient Clipping:
To prevent gradient explosions, rescale the gradient vector whenever its global `L_2` norm exceeds a predefined threshold:

```math
g \gets g \cdot \min\left(1.0, \, \frac{\text{threshold}}{\|g\|_2 + \epsilon}\right)
```

#### 2. Orthogonal Initialization:
Initialize `W_hh` such that all eigenvalues have absolute value `|\lambda_i| = 1.0`, preserving gradient norm over initial steps.

#### 3. Architectural Solution (The Path Forward):
Gated memory architectures introduce an additive identity gradient highway:
- **LSTM (Long Short-Term Memory)** in Project 09 (`09-lstm`).
- **GRU (Gated Recurrent Unit)** in Project 10 (`10-gru`).

---

## Q12 — Complete Mathematical Mapping to rnn.py Implementation

This cross-reference connects every mathematical equation derived above directly to its implementation in [`src/rnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/07-rnn-from-scratch/src/rnn.py):

| Code Function / Block | Mathematical Equation | Variables & Dimensions |
| :--- | :--- | :--- |
| `rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h)` | `z_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h`, `h_t = tanh(z_t)` | `x_t: (1, 4)`, `h_prev: (1, 3)` -> `h_t: (1, 3)` |
| `rnn_forward(X, h_0, W_xh, W_hh, b_h, W_hy, b_y)` | Unrolled recurrence over `T` steps + output projection | `X: (1, 4, 4)` -> `h: (5, 1, 3)`, `logits: (4, 1, 4)` |
| `softmax(logits)` | `\hat{y}_{t, k} = \exp(o_k - \max(o)) / \sum \exp(o_j - \max(o))` | `logits: (1, 4)` -> `probs: (1, 4)` |
| `sequence_cross_entropy(P, Y)` | `L_seq = -\sum_t \sum_k Y_{t, k} * \ln(\hat{y}_{t, k} + \epsilon)` | `P: (4, 4)`, `Y: (4, 4)` -> scalar `L_seq = 6.051364` |
| `rnn_backward(Y, cache, params)` | Full BPTT algorithm accumulating `dW_xh, dW_hh, db_h, dW_hy, db_y` | Upstream loss -> parameter gradients |
| `clip_gradients(grads, max_norm)` | `g <- g * (max_norm / \max(max_norm, ||g||))` | Prevents gradient explosion |
| `sample(h_prev, seed_char, length)` | Autoregressive sampling: sample next char from `\hat{y}_t`, feed as `x_{t+1}` | Generates `"hello"` character by character |

---

## Summary of the Complete Mathematical Pipeline

```text
====================================================================================================
STEP                          FORWARD OPERATION                               BACKWARD GRADIENT (BPTT)
====================================================================================================
1. Recurrent Pre-Activation   z_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h         dW_xh = sum(x_t.T @ d_zt)
                                                                              dW_hh = sum(h_{t-1}.T @ d_zt)
                                                                              db_h  = sum(d_zt)

2. Non-linear Memory State    h_t = tanh(z_t)                                 d_zt  = d_ht * (1 - h_t^2)
                                                                              d_ht  = d_ot @ W_hy.T + d_z_{t+1} @ W_hh.T

3. Output Projection Layer    o_t = h_t @ W_hy + b_y                          dW_hy = sum(h_t.T @ d_ot)
                                                                              db_y  = sum(d_ot)

4. Probability Normalization  y_hat_t = softmax(o_t)                          d_ot  = y_hat_t - Y_t
   & Sequence Loss            L_seq = -sum_t sum_k (Y_t * ln(y_hat_t))
====================================================================================================
```

This completes the foundational mathematics for Project 07. The next step is validating
each of these formulas in executable NumPy code in [`src/rnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/07-rnn-from-scratch/src/rnn.py).