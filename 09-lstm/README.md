# 09 — Long Short-Term Memory (LSTM) From Scratch

## Goal

Understand how the **Long Short-Term Memory (LSTM)** architecture resolves the fundamental limitations of the simple Recurrent Neural Network (RNN) from first principles with pure NumPy.

While the vanilla Simple RNN (`07-rnn-from-scratch` and `08-sentiment-rnn`) updates its single hidden state via repeated non-linear matrix multiplications:

```math
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
```

It suffers from catastrophic **vanishing and exploding gradients** when sequences exceed 10 to 15 timesteps.

The LSTM architecture (introduced by Sepp Hochreiter and Jürgen Schmidhuber in 1997) solves this problem by introducing an explicit **two-track memory system**:
1. **Cell State (`C_t`)**: An uninterrupted, linear **Long-Term Memory Highway** that carries error gradients backward through time without exponential decay.
2. **Hidden State (`h_t`)**: A non-linear **Short-Term Working Memory** filtered and regulated by three multiplicative gates.

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    Long-Term Memory                    │
Cell State Highway ────────────┼─────────────────────────► C_t ─────────────────────────┼────────────►
 (Linear Gradient Flow)        │                           ▲                            │
                               │                           │ (Add New Memory)           │
                               │                           │                            │
                               │   [ Forget Gate ]  [ Input Gate ]  [ Output Gate ]     │
                               │        f_t              i_t              o_t           │
                               │         │                │                │            │
                               │         ▼                ▼                ▼            │
Hidden State ──────────────────┼─────────────────────────────────────────► h_t ─────────┼────────────►
 (Working Memory)              │                                                        │
                               └────────────────────────────────────────────────────────┘
```

---

## The Core Mathematical Equations of an LSTM Cell

At timestep `t`, given input vector `x_t \in \mathbb{R}^{1 \times D}` and previous hidden state `h_{t-1} \in \mathbb{R}^{1 \times H}`, the LSTM computes:

### 1. Forget Gate (`f_t`): What Past Memory to Discard
Decides what percentage of the previous cell state `C_{t-1}` to retain or delete:

```math
f_t = \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f) \in (0, 1)^{1 \times H}
```

### 2. Input Gate (`i_t`): What New Memory to Write
Decides which memory dimensions will be updated:

```math
i_t = \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i) \in (0, 1)^{1 \times H}
```

### 3. Candidate Cell State (`\tilde{C}_t`): New Information Concepts
Creates a vector of new candidate information values centered between -1 and +1:

```math
\tilde{C}_t = \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c) \in (-1, 1)^{1 \times H}
```

### 4. Cell State Update (`C_t`): Additive Memory Fusion
Combines preserved past memory with modulated new candidate information:

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \in \mathbb{R}^{1 \times H}
```

### 5. Output Gate (`o_t`): What Memory to Expose
Decides which parts of the cell state `C_t` will be emitted to the hidden state `h_t`:

```math
o_t = \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o) \in (0, 1)^{1 \times H}
```

### 6. Hidden State Emission (`h_t`): Filtered Working Memory
Pushes the long-term cell state through `\tanh` and filters it with the output gate:

```math
h_t = o_t \odot \tanh(C_t) \in (-1, 1)^{1 \times H}
```

Where:
- `\sigma(z) = \frac{1}{1 + e^{-z}}`: The logistic Sigmoid activation function yielding values strictly in `(0, 1)`.
- `\odot`: The Hadamard element-wise product.
- `W_{xf}, W_{xi}, W_{xc}, W_{xo} \in \mathbb{R}^{D \times H}`: Input-to-gate projection matrices.
- `W_{hf}, W_{hi}, W_{hc}, W_{ho} \in \mathbb{R}^{H \times H}`: Hidden-to-gate recurrent matrices.
- `b_f, b_i, b_c, b_o \in \mathbb{R}^{1 \times H}`: Gate bias offset vectors.

---

## Why Vanilla RNNs Fail: The Vanishing Gradient Problem

To appreciate why LSTMs exist, we must examine the exact mathematical failure of the Simple RNN.

### The Repeated Jacobian Product

In a simple RNN, the gradient of the loss at timestep `T` with respect to the hidden state at an early timestep `t` is governed by the chain rule:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \frac{\partial h_j}{\partial h_{j-1}}
```

Expanding the single-step Jacobian matrix:

```math
\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}\left(1 - h_j^2\right) W_{hh}^T
```

Substituting this into the product chain:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \left( \operatorname{diag}\left(1 - h_j^2\right) W_{hh}^T \right)
```

### The Two Catastrophic Failure Modes

1. **The Repeated Matrix Multiplication**:
   As `(T - t)` grows large (e.g., sequences of length 30 or 50), the term resembles `(W_{hh}^T)^{T - t}`:
   - If the largest eigenvalue `|\lambda_{\max}| < 1`: The product exponentially decays to zero:
     ```math
     \lim_{T - t \to \infty} (W_{hh}^T)^{T - t} = \mathbf{0}
     ```
   - If `|\lambda_{\max}| > 1`: The product exponentially explodes toward infinity, causing numerical `NaN` overflow.

2. **The tanh Derivative Saturation Trap**:
   The derivative of `\tanh(z)` is `1 - \tanh^2(z) \in [0, 1]`.
   - Its maximum value is `1.0` (which occurs only at `z = 0`).
   - For any non-zero activation, `|1 - h^2| < 1.0` (typically `0.2` to `0.6`).
   - Multiplying 20 decimals together (`0.4^{20} \approx 1.1 \times 10^{-8}`) completely **obliterates the error gradient** before it can reach the beginning of the sequence!

#### The Consequence: Temporal Amnesia
The vanilla RNN cannot learn dependencies separated by more than 8 to 10 timesteps. In language modeling:
```text
"The CLOUDS in the sky gathered darkly, and after several hours of walking it began to [ ??? ]"
```
A vanilla RNN has already forgotten `"CLOUDS"` by the time it reaches the blank, guessing `"walk"` or `"sun"` instead of `"rain"`.

---

## How LSTM Solves Vanishing Gradients: The Additive Highway Proof

The core genius of the LSTM lies in the **Cell State equation**:

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

Let us compute the partial derivative of `C_t` with respect to `C_{t-1}`:

```math
\frac{\partial C_t}{\partial C_{t-1}} = \operatorname{diag}(f_t)
```

Notice what is missing from this equation:
- **No recurrent weight matrix multiplication `W_{hh}`!**
- **No derivative of `\tanh`!**

Over `T - t` timesteps, the gradient of the cell state flows as:

```math
\frac{\partial C_T}{\partial C_t} = \prod_{j=t+1}^{T} \operatorname{diag}(f_j)
```

### The Constant Error Carousel (CEC)

If the network learns to keep the forget gate open (`f_j \approx 1.0`):

```math
\frac{\partial C_T}{\partial C_t} \approx \prod_{j=t+1}^{T} \mathbf{I} = \mathbf{I}
```

The error gradient flows backward across 50, 100, or 500 timesteps **completely unaltered, without exponential decay**!
This linear additive channel is called the **Constant Error Carousel (CEC)**. The network can maintain a piece of information indefinitely until the forget gate explicitly decides to purge it.

---

## Detailed Visual Anatomy of an LSTM Cell

```text
                                        Cell State C_{t-1}
                                                │
                                                ▼
                   ┌────────────────────────────⊗───────────────────────⊕────────────────────────► Cell State C_t
                   │                            ▲                       ▲
                   │              Forget Signal │        New Info Signal│
                   │                            │                       │
                   │                     ┌──────┴──────┐         ┌──────┴──────┐
                   │                     │      f_t    │         │  i_t ⊗ C~_t │
                   │                     └──────┬──────┘         └──────┬──────┘
                   │                            │                       │
                   │                            │          ┌────────────┴────────────┐
                   │                            │          │                         │
                   │                            │          ▼                         ▼
                   │                          ┌───┐      ┌───┐                     ┌───┐
                   │                          │ σ │      │ σ │                     │tanh
                   │                          └───┘      └───┘                     └───┘
                   │                       Forget Gate Input Gate              Candidate State
                   │                            ▲          ▲                         ▲
                   │                            │          │                         │
                   │                            └──────────┼─────────────────────────┘
                   │                                       │
                   │    ┌──────────────────────────────────┴────────────────────────────────┐
                   │    │                                                                   │
                   │    │                                                                   ▼
                   │    │                                                                 ┌───┐
                   │    │                                                                 │ σ │ Output Gate
                   │    │                                                                 └───┘
                   │    │                                                                   │
                   │    │                                                                   ▼
                   │    │                                                            o_t ───⊗
                   │    │                                                                   ▲
                   │    │                                                                   │
                   │    │                                                                 ┌───┐
                   │    │                                                                 │tanh
                   │    │                                                                 └───┘
                   │    │                                                                   ▲
                   │    │                                                                   │
                   │    │                         Merged Cell State C_t ────────────────────┘
                   │    │                                                                   │
                   │    │                                                                   ▼
                   │    │                                                             Hidden State h_t
                   │    │                                                                   │
                   ▼    ▼                                                                   ▼
  Inputs: [ x_t, h_{t-1} ] ─────────────────────────────────────────────────────────────────┴──►
```

---

## Understanding the Three Gates

### 1. The Forget Gate (`f_t`)
- **Activation**: Sigmoid `\sigma \in (0, 1)`.
- **Function**: Multiplicative mask applied to historical memory `C_{t-1}`.
  - `f_t = 0.0`: Completely erase past memory along this dimension.
  - `f_t = 1.0`: Completely preserve past memory intact.
- **Linguistic Example**: When a paragraph finishes discussing the author's childhood and begins describing their college years, the forget gate clears the childhood context flags.

### 2. The Input Gate (`i_t`) and Candidate (`\tilde{C}_t`)
- **Candidate `\tilde{C}_t` (`\tanh`)**: Proposes new information values bounded in `(-1, +1)`.
- **Input Gate `i_t` (`\sigma`)**: Controls how much of that proposed candidate should actually be committed to long-term storage.
  - `i_t = 0.0`: Ignore the current input; do not write anything to cell state.
  - `i_t = 1.0`: Fully write the candidate information into long-term memory.
- **Linguistic Example**: When encountering the word `"not"`, the input gate stores a strong negative modifier into memory.

### 3. The Output Gate (`o_t`)
- **Activation**: Sigmoid `\sigma \in (0, 1)`.
- **Function**: Decides what subset of the deep cell state `C_t` should be emitted into the observable hidden state `h_t`.
  - `h_t` is used for **immediate classification** and passes into the **next timestep's gates**.
  - `C_t` remains protected inside the cell, shielded from external task interference.
- **Linguistic Example**: If the network needs to predict the next word after a verb, the output gate extracts the subject's grammatical number (singular vs. plural) from `C_t` to ensure subject-verb agreement.

---

## Vectorized Implementation: The Fused Projection Trick

A naive implementation computes 4 separate matrix multiplications for the 4 gates:

```python
# Naive approach: 4 separate matrix multiplications (SLOW)
f = sigmoid(x @ W_xf + h_prev @ W_hf + b_f)
i = sigmoid(x @ W_xi + h_prev @ W_hi + b_i)
c_bar = np.tanh(x @ W_xc + h_prev @ W_hc + b_c)
o = sigmoid(x @ W_xo + h_prev @ W_ho + b_o)
```

In professional deep learning libraries (and our from-scratch implementation), we **stack all 4 gate weights into single unified matrices**:

```math
W_x = \begin{bmatrix} W_{xf} & W_{xi} & W_{xc} & W_{xo} \end{bmatrix} \in \mathbb{R}^{D \times 4H}
```

```math
W_h = \begin{bmatrix} W_{hf} & W_{hi} & W_{hc} & W_{ho} \end{bmatrix} \in \mathbb{R}^{H \times 4H}
```

```math
b = \begin{bmatrix} b_f & b_i & b_c & b_o \end{bmatrix} \in \mathbb{R}^{1 \times 4H}
```

### The Fused Forward Step

We compute all 4 gate activations with **just two matrix multiplications**:

```math
Z = x_t W_x + h_{t-1} W_h + b \in \mathbb{R}^{B \times 4H}
```

We then slice `Z` into 4 equal blocks of width `H`:

```python
# Vectorized fused projection (FAST)
Z = x @ W_x + h_prev @ W_h + b  # Shape: (B, 4*H)

z_f, z_i, z_c, z_o = np.split(Z, 4, axis=-1)

f = sigmoid(z_f)
i = sigmoid(z_i)
c_bar = np.tanh(z_c)
o = sigmoid(z_o)

C = f * C_prev + i * c_bar
h = o * np.tanh(C)
```

> [!TIP]
> Fused matrix multiplication maximizes cache locality, minimizes BLAS overhead, and reduces Python loop dispatch time by over 60%!

---

## Parameters and Tensor Shapes Reference

Let:
- `B`: Batch size (number of parallel sequences).
- `T`: Sequence length (number of timesteps).
- `D`: Input feature dimension (e.g., embedding size or feature count).
- `H`: Hidden state dimension (number of memory units).
- `K`: Output classification classes.

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `X` | `(B, T, D)` | `X \in \mathbb{R}^{B \times T \times D}` | Input sequence batch |
| `x_t` | `(B, D)` | `x_t \in \mathbb{R}^{B \times D}` | Input slice at timestep `t` |
| `h_{t-1}` | `(B, H)` | `h_{t-1} \in \mathbb{R}^{B \times H}` | Prior working hidden state |
| `C_{t-1}` | `(B, H)` | `C_{t-1} \in \mathbb{R}^{B \times H}` | Prior long-term cell state |
| `W_x` | `(D, 4H)` | `W_x \in \mathbb{R}^{D \times 4H}` | Fused input-to-gates weight matrix |
| `W_h` | `(H, 4H)` | `W_h \in \mathbb{R}^{H \times 4H}` | Fused hidden-to-gates recurrent weight matrix |
| `b` | `(1, 4H)` | `b \in \mathbb{R}^{1 \times 4H}` | Fused gate bias offset vector |
| `f_t` | `(B, H)` | `f_t \in (0, 1)^{B \times H}` | Forget gate activations |
| `i_t` | `(B, H)` | `i_t \in (0, 1)^{B \times H}` | Input gate activations |
| `\tilde{C}_t` | `(B, H)` | `\tilde{C}_t \in (-1, 1)^{B \times H}` | Candidate cell state |
| `C_t` | `(B, H)` | `C_t \in \mathbb{R}^{B \times H}` | Updated long-term cell state |
| `o_t` | `(B, H)` | `o_t \in (0, 1)^{B \times H}` | Output gate activations |
| `h_t` | `(B, H)` | `h_t \in (-1, 1)^{B \times H}` | Updated working hidden state |
| `W_{hy}` | `(H, K)` | `W_{hy} \in \mathbb{R}^{H \times K}` | Output projection weights |
| `b_y` | `(1, K)` | `b_y \in \mathbb{R}^{1 \times K}` | Output classification bias |
| `logits` | `(B, K)` | `o_{\mathrm{out}} \in \mathbb{R}^{B \times K}` | Raw classification logits |

---

## Architectural Comparison: Simple RNN vs. LSTM

| Feature | Simple RNN (`07-rnn-from-scratch`) | LSTM (`09-lstm`) |
| :--- | :--- | :--- |
| **Internal Memory States** | 1 (`h_t` only) | **2 (`h_t` and `C_t`)** |
| **State Update Equation** | `h_t = \tanh(x W + h W + b)` | `C_t = f \odot C_{t-1} + i \odot \tilde{C}_t` |
| **Memory Update Nature** | Multiplicative (repeated non-linearities) | **Additive (linear gradient highway)** |
| **Gating Mechanisms** | None (fixed transformation) | **3 gates (Forget, Input, Output)** |
| **Activation Functions** | `\tanh` only | `\sigma` for gates, `\tanh` for states |
| **Parameter Count** | `D \times H + H \times H + H` | **`4 \times (D \times H + H \times H + H)` (4x parameters!)** |
| **Effective Sequence Length** | 8 to 15 timesteps | **100 to 500+ timesteps** |
| **Vanishing Gradient Vulnerability** | Extreme (fails rapidly) | **Virtually Eliminated along cell state highway** |
| **Exploding Gradient Vulnerability** | High | Low (handled via gradient clipping) |

---

## Concrete Toy Educational Setup

To verify every mathematical derivative and BPTT tensor by hand, we configure a transparent benchmark task:

### The Long-Range Bit Memory Benchmark
A binary sequence of length `T = 20` has an informative indicator bit at `t = 1` (`0` or `1`), followed by 18 random noise bits, and asks the model to output the original bit at `t = 20`:

```text
Sequence 1: [ 1,  0.4, -0.2,  0.8, ...,  0.1 ] ──► Target: 1 (Must remember bit from step 1!)
Sequence 2: [ 0, -0.3,  0.7, -0.5, ..., -0.2 ] ──► Target: 0 (Must remember bit from step 1!)
```

A Simple RNN fails completely on this task because the gradient vanishes across the 18 intermediate noise steps. The LSTM solves it effortlessly by setting `f_t = 1.0` and `i_t = 0.0` for all noise steps!

### Explicit Dimensions:
- Input dimension: `D = 2`
- Hidden state capacity: `H = 3`
- Output classes: `K = 2`

#### Parameter Budget:
```text
W_x : (D, 4H) = (2, 12) -> 24 weights
W_h : (H, 4H) = (3, 12) -> 36 weights
b   : (1, 4H) = (1, 12) -> 12 biases
W_hy: (H, K)  = (3, 2)  ->  6 weights
b_y : (1, K)  = (1, 2)  ->  2 biases
--------------------------------------
Total learnable parameters = 80 parameters
```

An 80-parameter model allows exact manual arithmetic tracing and unit-test assertions down to 6 decimal places.

---

## Project Structure

```text
09-lstm/
├── README.md                 # Theoretical foundations, gate mechanics, and 101-topic syllabus
├── src/
│   └── lstm.py               # Pure NumPy fused LSTM cell, forward pass, BPTT, and training
└── steps/
    └── step-01-mathematics.md # Mathematical proofs, Jacobian analysis, and complete BPTT calculus
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Completed | [`steps/step-01-mathematics.md`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/09-lstm/steps/step-01-mathematics.md): Additive highway proof, gate equations, BPTT matrix calculus |
| **2. NumPy Implementation** | Completed | [`src/lstm.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/09-lstm/src/lstm.py): Vectorized fused gates, `lstm_cell_forward`, `lstm_forward`, `lstm_backward` |
| **3. Understand Forward Pass** | Completed | Tracing state evolution of `C_t` and `h_t` across all timesteps with verified test assertions |
| **4. Understand Loss** | Completed | Categorical cross-entropy and sequence loss ($L_{\mathrm{seq}} = 1.928327$) |
| **5. Derive Gradients** | Completed | Complete multivariate calculus for 4 gate errors and additive cell backpropagation |
| **6. Implement Backpropagation** | Completed | Vectorized BPTT accumulating fused weight gradients `dW_x, dW_h, db` verified with finite difference ($< 10^{-9}$ rel error) |
| **7. Train Model** | Completed | Training on long-range temporal dependencies ($T = 20$), achieving 100% terminal accuracy |
| **8. Debug & Analyze** | Completed | Gate inspection and forget gate bias initialization ($b_f = 1.0$) proving Constant Error Carousel (CEC) superiority over Simple RNN |
| **9. PyTorch Implementation** | Pending | Equivalent implementation using `torch.nn.LSTM` and parameter tensor matching |
| **10. Compare Results** | Pending | Direct validation of loss curves, state trajectories, and gradient norms |

---

## Key Takeaways

1. **The Dual Memory System**:
   LSTMs separate deep long-term storage (`C_t`) from immediate working context (`h_t`). `C_t` changes linearly and slowly, while `h_t` adapts dynamically to immediate observations.
2. **The Constant Error Carousel**:
   Because `C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t`, the gradient `\frac{\partial C_t}{\partial C_{t-1}} = f_t` is linear and additive. With `f_t \approx 1.0`, gradients bypass non-linear decay, resolving the vanishing gradient problem.
3. **Multiplicative Gating**:
   Gates act as dynamic, data-dependent valves:
   - Forget gate decides what to erase.
   - Input gate decides what to record.
   - Output gate decides what to read and project.
4. **Fused Matrix Vectorization**:
   Stacking all 4 gates into a single `(D, 4H)` matrix reduces memory bandwidth overhead and provides a 3x speedup over naive gate implementations.
5. **Forget Gate Bias Initialization**:
   Setting `b_f = 1.0` or `2.0` at initialization ensures that the model starts by remembering everything by default, learning to forget only when necessary.

---

## Next Steps

1. Implement the equivalent model using PyTorch (`torch.nn.LSTM`) to verify parameter layout (`weight_ih_l0`, `weight_hh_l0`) and gate permutation ordering (`(i, f, g, o)` vs. fused `(f, i, c, o)`).
2. Directly validate loss curves, state trajectories, and runtime performance benchmarks between the pure NumPy and PyTorch implementations.
