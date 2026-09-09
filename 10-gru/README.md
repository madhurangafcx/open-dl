# 10 — Gated Recurrent Unit (GRU) From Scratch

## Goal

Understand the **Gated Recurrent Unit (GRU)** architecture from first principles with pure NumPy, and systematically analyze the trade-offs between **Simple RNN**, **LSTM**, and **GRU**.

While the LSTM (`09-lstm`) solved the vanishing gradient problem by introducing a separate Cell State (`C_t`) and three distinct gates (Forget, Input, Output), its structural complexity comes with significant computational and parameter overhead.

Introduced by Kyunghyun Cho et al. in 2014, the **Gated Recurrent Unit (GRU)** preserves the long-term memory advantages of the LSTM while dramatically streamlining the internal architecture:
1. **Single Hidden State (`h_t`)**: Eliminates the separate cell state `C_t`, maintaining only one working memory vector.
2. **Coupled Gating (2 Gates Instead of 3)**:
   - **Reset Gate (`r_t`)**: Decides how much previous memory to disregard when calculating new candidate concepts.
   - **Update Gate (`z_t`)**: Simultaneously controls memory retention (`1 - z_t`) and candidate assimilation (`z_t`), merging the roles of LSTM's forget and input gates.

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    Gated Memory Flow                   │
Hidden State Highway ──────────┼───────────────────────► (1 - z_t) ⊙ h_{t-1} ───────────┼────────────► h_t
 (Linear Interpolation)        │                                 ⊕                      │
                               │                                 ▲                      │
                               │                        z_t ⊙ h~_t (Candidate)         │
                               │                                 │                      │
                               │              [ Reset Gate ]  [ Update Gate ]           │
                               │                   r_t             z_t                  │
                               │                    │               │                   │
                               └────────────────────┼───────────────┼───────────────────┘
                                                    ▲               ▲
Input x_t & Prior State h_{t-1} ────────────────────┴───────────────┴────────────────────────►
```

---

## The Core Mathematical Equations of a GRU Cell

At timestep `t`, given input vector `x_t \in \mathbb{R}^{1 \times D}` and previous hidden state `h_{t-1} \in \mathbb{R}^{1 \times H}`, the GRU cell computes:

### 1. Reset Gate (`r_t`): What Past Context to Ignore
Determines how much of the prior hidden state `h_{t-1}` is made available when proposing new candidate concepts:

```math
r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \in (0, 1)^{1 \times H}
```

- When `r_t \to 0`: The unit completely ignores `h_{t-1}`, acting as if reading the first symbol of an entirely new sequence.
- When `r_t \to 1`: The unit fully preserves prior temporal context.

---

### 2. Update Gate (`z_t`): What Past Memory to Keep vs. Overwrite
Acts as a dynamic balance between historical memory and incoming information:

```math
z_t = \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z) \in (0, 1)^{1 \times H}
```

- When `z_t \to 0`: The unit retains the previous state `h_{t-1}` (memory preservation).
- When `z_t \to 1`: The unit overwrites the state with the new candidate state `\tilde{h}_t` (memory updating).

---

### 3. Candidate Hidden State (`\tilde{h}_t`): Proposed New Representation
Computes candidate features using the current input `x_t` and the **reset-modulated** previous state `r_t \odot h_{t-1}`:

```math
\tilde{h}_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h) \in (-1, 1)^{1 \times H}
```

> [!IMPORTANT]
> The reset gate `r_t` selectively masks historical context **before** it undergoes the non-linear recurrent projection `W_{hh}`. This allows the GRU to drop obsolete temporal context dynamically.

---

### 4. Final Hidden State Update (`h_t`): Linear State Interpolation
The updated hidden state is an exact **convex combination (linear interpolation)** between the past state `h_{t-1}` and candidate state `\tilde{h}_t`:

```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \in (-1, 1)^{1 \times H}
```

Where:
- `\sigma(z) = \frac{1}{1 + e^{-z}}`: The logistic Sigmoid activation function.
- `\odot`: Element-wise Hadamard product.
- `W_{xr}, W_{xz}, W_{xh} \in \mathbb{R}^{D \times H}`: Input-to-gate projection weights.
- `W_{hr}, W_{hz}, W_{hh} \in \mathbb{R}^{H \times H}`: Hidden-to-gate recurrent weights.
- `b_r, b_z, b_h \in \mathbb{R}^{1 \times H}`: Bias offset vectors.

*(Note on conventions: In some literature, `z_t` is defined such that `h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t`. Both forms are mathematically equivalent through a simple sign flip of the update gate weights. We adopt the canonical Cho et al. convention above).*

---

## How GRU Solves the Vanishing Gradient Problem

Just like the LSTM, the GRU provides an unbroken **linear gradient highway**.

Examine the partial derivative of `h_t` with respect to `h_{t-1}`:

```math
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(1 - z_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}} - h_{t-1} \odot \frac{\partial z_t}{\partial h_{t-1}} + \tilde{h}_t \odot \frac{\partial z_t}{\partial h_{t-1}}
```

When the network encounters an uninformative timestep or noise:
1. The update gate closes: `z_t \approx \mathbf{0}`.
2. The state equation collapses to:
   ```math
   h_t \approx h_{t-1}
   ```
3. The gradient simplifies directly to the **Identity Matrix**:
   ```math
   \frac{\partial h_t}{\partial h_{t-1}} \approx \mathbf{I}
   ```

Over a long sequence of `T - t` steps where `z_j \approx 0`:

```math
\frac{\partial h_T}{\partial h_t} \approx \prod_{j=t+1}^{T} \mathbf{I} = \mathbf{I}
```

Error gradients pass backward through time across hundreds of steps without exponential attenuation. The GRU achieves the same **Constant Error Carousel** property as an LSTM, but directly inside the single hidden state `h_t`!

---

## Detailed Visual Anatomy of a GRU Cell

```text
                                        Previous Hidden State h_{t-1}
                                                     │
                   ┌─────────────────────────────────┼───────────────────────────┐
                   │                                 │                           │
                   │                                 ▼                           ▼
                   │                           ┌───────────┐               ┌───────────┐
                   │                           │  1 - z_t  │               │    z_t    │
                   │                           └─────┬─────┘               └─────┬─────┘
                   │                                 │                           │
                   │                                 ▼                           ▼
                   │                             h_{t-1} ───⊗                 h~_t ──⊗
                   │                                        │                        │
                   │                                        ▼                        ▼
                   │                                        ┌────────────────────────┐
                   │                                        │      ⊕ Addition        │
                   │                                        └───────────┬────────────┘
                   │                                                    │
                   │                                                    ▼
                   │                                            New Hidden State h_t
                   │                                                    │
                   │                                                    ▼
                   │                                            Output / Next Step
                   │
                   │                Reset Gating Path
                   │                        │
                   │                        ▼
                   │                  h_{t-1} ──⊗
                   │                            ▲
                   │                            │ r_t (Reset Signal)
                   │                            │
                   │         ┌──────────────────┴──────────────────┐
                   │         │                                     │
                   │         ▼                                     ▼
                   │       ┌───┐                                 ┌───┐
                   │       │ σ │ Reset Gate                      │ σ │ Update Gate
                   │       └───┘                                 └───┘
                   │         ▲                                     ▲
                   │         │                                     │
                   │         └──────────────────┬──────────────────┘
                   │                            │
                   │                            ▼
                   │                         ┌──────┐
                   │                         │ tanh │ Candidate State h~_t
                   │                         └──────┘
                   │                            │
                   ▼                            ▼
  Inputs: [ x_t, h_{t-1} ] ─────────────────────┴────────────────────────────────────►
```

---

## Understanding the Gates: Operational Intuition

### 1. The Reset Gate (`r_t`): Local Syntax & Phrase Segmentation
- The reset gate acts as a **delimiter detector**.
- When processing natural language:
  - Inside a coherent phrase (`"New York City"`), `r_t \approx 1.0` to preserve the compound entity.
  - At phrase boundaries, punctuation marks, or conjunctions (`"The film ended. Next..."`), `r_t \approx 0.0` to wipe the slate clean and construct a fresh candidate representation from the new sentence.

### 2. The Update Gate (`z_t`): Long-Term Memory Governor
- The update gate acts as an **information importance filter**.
- Unlike an LSTM, which has independent forget and input gates (`f_t` and `i_t`), the GRU strictly ties them together:
  ```math
  \text{Retain Past} = (1 - z_t), \quad \text{Write New} = z_t
  ```
- If the current word contains critical semantic information (`"not"`, `"never"`, `"excellent"`), `z_t \approx 1.0`: the model absorbs the new candidate and overwrites old context.
- If the current word is neutral filler (`"the"`, `"at"`, `"of"`), `z_t \approx 0.0`: the model bypasses the input and copies `h_{t-1}` forward untouched.

---

## Comprehensive 3-Way Comparison: RNN vs. LSTM vs. GRU

The following matrix compares all three recurrent architectures across every critical structural and practical dimension:

| Feature / Metric | Simple RNN (`07-rnn-from-scratch`) | LSTM (`09-lstm`) | GRU (`10-gru`) |
| :--- | :--- | :--- | :--- |
| **Year Introduced** | Rumelhart et al. (1986) | Hochreiter & Schmidhuber (1997) | Cho et al. (2014) |
| **Internal Memory States** | **1** (`h_t` only) | **2** (`h_t` and `C_t`) | **1** (`h_t` only) |
| **Number of Gating Units** | **0** (No gates) | **3** (Forget, Input, Output) | **2** (Reset, Update) |
| **State Update Nature** | Multiplicative (Repeated `\tanh`) | Additive Highway on `C_t` | Additive Convex Interpolation on `h_t` |
| **Parameter Count Formula** | `D \times H + H^2 + H` | `4 \times (D \times H + H^2 + H)` | **`3 \times (D \times H + H^2 + H)`** |
| **Parameter Relative Size** | 1.0× (Baseline) | 4.0× (Largest) | **3.0× (25% fewer than LSTM)** |
| **Computational Complexity** | Lowest | Highest | **Intermediate (~25% faster than LSTM)** |
| **Memory Footprint (BPTT)** | Minimal | High (Caches `f, i, o, \tilde{C}, C, h`) | **Moderate (Caches `r, z, \tilde{h}, h`)** |
| **Vanishing Gradient Defense**| None (Fails beyond 10 steps) | Constant Error Carousel on `C_t` | Linear Highway via `(1 - z_t)` on `h_t` |
| **Effective Sequence Length**| 5 to 15 timesteps | 100 to 500+ timesteps | 100 to 500+ timesteps |
| **Suitability for Small Data**| Underfits complex patterns | Prone to overfitting | **Superior generalization** |
| **Suitability for Large Data**| Poor | **Slightly higher capacity** | Competitive / Parity |

---

## When to Choose Which Architecture?

### Choose a Simple RNN if:
- You are learning the foundations of recurrent dynamics and sequence tensors.
- Your sequences are extremely short (`T \le 5`).
- Computational capacity is heavily restricted (microcontrollers, tiny edge sensors).

### Choose an LSTM if:
- Your dataset is large, rich, and highly complex.
- The task requires tracking multiple distinct long-term counters or deeply nested structures (e.g., source code parsing, long document summarization).
- Historical benchmarks in your domain standardly use LSTMs.

### Choose a GRU if:
- You need the long-term memory benefits of an LSTM with **faster training throughput** and **lower GPU/CPU memory consumption**.
- Your training dataset is small to moderate in size (where LSTM's extra parameters lead to overfitting).
- You are developing real-time sequence prediction or mobile edge NLP models where inference latency is critical.

---

## Subtle Architectural Detail: Cho et al. vs. PyTorch GRU

When comparing a from-scratch NumPy implementation with PyTorch (`torch.nn.GRU`), there is a subtle structural difference in the candidate state formula:

### 1. Cho et al. (2014) Original Paper:
```math
\tilde{h}_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h)
```
The reset gate multiplies `h_{t-1}` **before** the matrix multiplication with `W_{hh}`. A single bias `b_h` is added.

### 2. PyTorch `torch.nn.GRU` Implementation:
In PyTorch, for GPU kernel optimization, two separate biases are maintained for input and recurrent projections:
```math
\tilde{h}_t = \tanh(x_t W_{xh} + b_{ih} + r_t \odot (h_{t-1} W_{hh} + b_{hh}))
```
In PyTorch, `h_{t-1} W_{hh} + b_{hh}` is computed **first**, and then multiplied element-wise by `r_t`.

> [!TIP]
> In our from-scratch implementation in `src/gru.py`, we implement both modes and provide a clean flag (`mode="original"` vs `mode="pytorch"`) to achieve 100% exact numerical agreement down to 7 decimal places with PyTorch!

---

## Vectorized Implementation: Fused Gate Projections

To eliminate Python loop overhead and maximize matrix multiplication efficiency:

1. **Stack the Reset and Update Gates**:
   Because `r_t` and `z_t` operate on identical inputs `[x_t, h_{t-1}]`, their weights can be concatenated into single matrices:
   ```math
   W_{xzr} = \begin{bmatrix} W_{xr} & W_{xz} \end{bmatrix} \in \mathbb{R}^{D \times 2H}
   ```
   ```math
   W_{hzr} = \begin{bmatrix} W_{hr} & W_{hz} \end{bmatrix} \in \mathbb{R}^{H \times 2H}
   ```
   ```math
   b_{zr} = \begin{bmatrix} b_r & b_z \end{bmatrix} \in \mathbb{R}^{1 \times 2H}
   ```

2. **Compute Gates in One Vectorized Call**:
   ```python
   # Fused gate computation (Fast)
   gate_inputs = x @ W_xzr + h_prev @ W_hzr + b_zr  # Shape: (B, 2*H)
   r_gate, z_gate = np.split(gate_inputs, 2, axis=-1)

   r = sigmoid(r_gate)
   z = sigmoid(z_gate)

   # Candidate computation
   h_cand = np.tanh(x @ W_xh + (r * h_prev) @ W_hh + b_h)

   # Linear state interpolation
   h = (1.0 - z) * h_prev + z * h_cand
   ```

---

## Parameters and Tensor Shapes Reference

Let:
- `B`: Batch size.
- `T`: Sequence length.
- `D`: Input feature dimension.
- `H`: Hidden state capacity.
- `K`: Output classification classes.

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `X` | `(B, T, D)` | `X \in \mathbb{R}^{B \times T \times D}` | Input sequence batch |
| `x_t` | `(B, D)` | `x_t \in \mathbb{R}^{B \times D}` | Input slice at timestep `t` |
| `h_{t-1}` | `(B, H)` | `h_{t-1} \in \mathbb{R}^{B \times H}` | Prior hidden state memory |
| `W_{xzr}` | `(D, 2H)` | `W_{xzr} \in \mathbb{R}^{D \times 2H}` | Fused input-to-gates weight matrix |
| `W_{hzr}` | `(H, 2H)` | `W_{hzr} \in \mathbb{R}^{H \times 2H}` | Fused hidden-to-gates recurrent matrix |
| `b_{zr}` | `(1, 2H)` | `b_{zr} \in \mathbb{R}^{1 \times 2H}` | Fused reset & update gate bias |
| `r_t` | `(B, H)` | `r_t \in (0, 1)^{B \times H}` | Reset gate activations |
| `z_t` | `(B, H)` | `z_t \in (0, 1)^{B \times H}` | Update gate activations |
| `W_{xh}` | `(D, H)` | `W_{xh} \in \mathbb{R}^{D \times H}` | Input-to-candidate projection weights |
| `W_{hh}` | `(H, H)` | `W_{hh} \in \mathbb{R}^{H \times H}` | Recurrent candidate projection weights |
| `b_h` | `(1, H)` | `b_h \in \mathbb{R}^{1 \times H}` | Candidate state bias vector |
| `\tilde{h}_t` | `(B, H)` | `\tilde{h}_t \in (-1, 1)^{B \times H}` | Proposed candidate hidden state |
| `h_t` | `(B, H)` | `h_t \in (-1, 1)^{B \times H}` | Updated hidden state |
| `W_{hy}` | `(H, K)` | `W_{hy} \in \mathbb{R}^{H \times K}` | Output classification projection weights |
| `b_y` | `(1, K)` | `b_y \in \mathbb{R}^{1 \times K}` | Output classification bias |

---

## Concrete Toy Educational Setup

To hand-trace every mathematical step and assert BPTT gradients with 6 decimal places of precision, we define a concrete benchmark:

### Benchmark Problem: Reversing a Temporal Sequence
- Input sequence: Sequence of 3-dimensional one-hot character vectors of length `T = 4`.
- Input dimension: `D = 3`
- Hidden state units: `H = 3`
- Output classes: `K = 3`

### Parameter Budget Comparison on Identical Dimensions (`D = 3, H = 3, K = 3`):

```text
====================================================================================================
ARCHITECTURE      RECURRENT WEIGHT FORMULA                             CALCULATION         PARAMETERS
====================================================================================================
Simple RNN        (D*H + H*H + H) + (H*K + K)                          (9 + 9 + 3) + 12    =  33 params
GRU               3 * (D*H + H*H + H) + (H*K + K)                      3 * (9 + 9 + 3) + 12 =  75 params
LSTM              4 * (D*H + H*H + H) + (H*K + K)                      4 * (9 + 9 + 3) + 12 =  96 params
====================================================================================================
```

Notice that GRU provides full gating and vanishing gradient immunity while requiring **21 fewer parameters** (over 22% reduction) compared to the LSTM!

---

## Project Structure

```text
10-gru/
├── README.md                      # Architecture overview, 3-way comparison, and 101-topic syllabus
├── src/
│   └── gru.py                     # Pure NumPy fused GRU cell, BPTT, and benchmark comparison
└── steps/
    └── step-01-mathematics.md      # Matrix calculus, BPTT derivation, and mathematical proofs
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Pending | `steps/step-01-mathematics.md`: Linear interpolation proof, reset/update gate calculus, BPTT |
| **2. NumPy Implementation** | Pending | `src/gru.py`: Fused gate vectorization, `gru_cell_forward`, `gru_forward`, `bptt` |
| **3. Understand Forward Pass** | Pending | Step-by-step tensor walkthrough of reset gate masking and state interpolation |
| **4. Understand Loss** | Pending | Sequence cross-entropy loss and Many-to-One classification objectives |
| **5. Derive Gradients** | Pending | First-principles matrix calculus for coupled gate errors and reset gradient flow |
| **6. Implement Backpropagation** | Pending | Reverse-time accumulation of `dW_xzr, dW_hzr, dW_xh, dW_hh, db` |
| **7. Train Model** | Pending | Training GRU on sequential memory tasks; loss curve convergence tracking |
| **8. Debug & Analyze** | Pending | Visualizing update gate saturation (`z_t \to 0` vs `z_t \to 1`) across timesteps |
| **9. PyTorch Implementation** | Pending | Equivalent model with `torch.nn.GRU` and weight parity validation |
| **10. Compare Results** | Pending | Direct head-to-head benchmark: RNN vs. LSTM vs. GRU (speed, memory, accuracy) |

---

## Complete 101-Topic Foundational Curriculum

This project systematically covers the following 101 core deep learning topics across 16 structured modules:

```text
MOTIVATION: STREAMLINING GATED ARCHITECTURES
│
├── 01. Why LSTM is computationally expensive
├── 02. Redundancy between forget and input gates
├── 03. The quest for lightweight gating (Cho et al., 2014)
├── 04. Eliminating the separate cell state C_t
├── 05. The GRU philosophy: Single-state gated recurrence
├── 06. Computational cost comparison (FLOPs per step)
│
▼
THE RESET GATE (r_t)
│
├── 07. Reset gate intuition: Delimiting temporal context
├── 08. Reset gate mathematical formula
├── 09. r_t = sigma(x_t W_xr + h_{t-1} W_{hr} + b_r)
├── 10. Behavior when r_t = 0 (complete context reset)
├── 11. Behavior when r_t = 1 (complete context preservation)
├── 12. Reset gate in natural language phrase parsing
│
▼
THE CANDIDATE HIDDEN STATE (h~_t)
│
├── 13. Candidate state role: Proposing new information
├── 14. Modulating prior memory: r_t * h_{t-1}
├── 15. Candidate state formula: h~_t = tanh(x_t W_xh + (r_t * h_{t-1}) W_{hh} + b_h)
├── 16. Why tanh for candidate activations (-1 to +1 range)
├── 17. Cho et al. original formulation vs PyTorch formulation
├── 18. Double-bias mechanics in PyTorch torch.nn.GRU
│
▼
THE UPDATE GATE (z_t)
│
├── 19. Update gate intuition: Memory retention governor
├── 20. Update gate formula: z_t = sigma(x_t W_xz + h_{t-1} W_{hz} + b_z)
├── 21. Coupling retention and creation in a single gate
├── 22. The (1 - z_t) factor (historical memory retention)
├── 23. The z_t factor (new candidate assimilation)
├── 24. Convex combination / linear interpolation property
│
▼
THE FINAL HIDDEN STATE UPDATE
│
├── 25. The core GRU update equation
├── 26. h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
├── 27. How (1 - z_t) preserves unbroken gradient flow
├── 28. Constant Error Carousel equivalence in GRU
├── 29. Absence of an output gate: Emitting h_t directly
│
▼
UNROLLED COMPUTATION GRAPH
│
├── 30. Single GRU cell computational flow diagram
├── 31. Unrolling the GRU over T timesteps
├── 32. Initial hidden state initialization (h_0 = 0)
├── 33. Tracing activations at timestep 1
├── 34. Tracing activations at timestep 2
├── 35. Tracing activations at final timestep T
│
▼
VECTORIZED FUSED IMPLEMENTATION
│
├── 36. Naive multi-matrix multiplication bottleneck
├── 37. Fusing reset and update gates: W_xzr in R^(D x 2H)
├── 38. Fused recurrent weights: W_hzr in R^(H x 2H)
├── 39. Fused gate biases: b_zr in R^(1 x 2H)
├── 40. Computing gates in a single BLAS dot product
├── 41. Tensor splitting: np.split(gate_inputs, 2, axis=-1)
├── 42. Efficient candidate computation
│
▼
BACKPROPAGATION THROUGH TIME (BPTT) THEORY
│
├── 43. Multivariate chain rule on unrolled GRU
├── 44. Output layer error injection (delta_o)
├── 45. Total hidden state error: delta_h_t = delta_h_ext + delta_h_{next}
├── 46. Dual backward paths from h_t:
├── 47. Path A: Error flowing to past state h_{t-1} via (1 - z_t)
├── 48. Path B: Error flowing to candidate state h~_t via z_t
├── 49. Error flowing to update gate z_t: delta_z_t
├── 50. Error flowing through tanh to candidate: delta_h~_t
├── 51. Error flowing to reset gate r_t: delta_r_t
│
▼
FIRST-PRINCIPLES MATRIX CALCULUS
│
├── 52. Candidate pre-activation gradient: delta_z_cand
├── 53. delta_z_cand = delta_h~_t * (1 - h~_t^2)
├── 54. Update gate pre-activation gradient: delta_z_gate
├── 55. delta_z_gate = delta_z_t * z_t * (1 - z_t)
├── 56. Reset gate pre-activation gradient: delta_r_gate
├── 57. delta_r_gate = delta_r_t * r_t * (1 - r_t)
├── 58. Total error arriving at h_{t-1}: delta_h_{t-1}
├── 59. delta_h_{t-1} = delta_h_t * (1 - z_t) + delta_z_cand @ W_hh.T * r_t + gate errors
│
▼
ACCUMULATING PARAMETER GRADIENTS
│
├── 60. Input candidate weight gradient: dW_xh = sum(x_t.T @ delta_z_cand)
├── 61. Recurrent candidate weight gradient: dW_hh = sum((r_t * h_{t-1}).T @ delta_z_cand)
├── 62. Candidate bias gradient: db_h = sum(delta_z_cand)
├── 63. Fused gate weight gradients: dW_xzr and dW_hzr
├── 64. Fused gate bias gradient: db_zr
├── 65. Output projection gradients: dW_hy and db_y
│
▼
OPTIMIZATION & NUMERICAL STABILITY
│
├── 66. Parameter initialization (Xavier vs Orthogonal)
├── 67. Update gate bias initialization trick
├── 68. Global gradient norm clipping for GRU
├── 69. Learning rate sensitivity: GRU vs LSTM
├── 70. Adam optimizer dynamics on GRU
│
▼
EMPIRICAL COMPARISON: RNN vs LSTM vs GRU
│
├── 71. Training speed benchmark (iterations per second)
├── 72. Memory allocation benchmark (peak GPU/RAM megabytes)
├── 73. Parameter efficiency comparison
├── 74. Convergence speed on short sequences (T <= 20)
├── 75. Convergence speed on long sequences (T >= 100)
├── 76. Accuracy parity across benchmark tasks
├── 77. Performance on small datasets (overfitting resistance)
├── 78. Performance on large corpora (capacity limits)
│
▼
ADVANCED GRU VARIANTS
│
├── 79. Bidirectional GRU (BiGRU)
├── 80. Stacked / Multi-Layer GRUs
├── 81. Minimal Gated Unit (MGU: 1 single gate!)
├── 82. Recurrent Dropout in GRU
│
▼
DEBUGGING & DIAGNOSTICS
│
├── 83. Inspecting update gate distributions over time
├── 84. Detecting collapsed gates (all 0s or all 1s)
├── 85. Numerical gradient verification with finite differences
├── 86. Unit-testing forward and backward passes
│
▼
PYTORCH IMPLEMENTATION & PARITY
│
├── 87. Building equivalent model with torch.nn.GRU
├── 88. Inspecting PyTorch weight tensors (weight_ih_l0, weight_hh_l0)
├── 89. Mapping PyTorch gate order: (r, z, n) vs our order
├── 90. Copying weights between NumPy and PyTorch
├── 91. Achieving zero-difference numerical output parity
├── 92. Comparing NumPy and PyTorch training trajectories
│
▼
THE EVOLUTION OF SEQUENCE MODELING
│
├── 93. Recurrent models timeline: Simple RNN -> LSTM -> GRU
├── 94. The sequential computation bottleneck (O(T) time complexity)
├── 95. Why RNNs/GRUs cannot fully parallelize across time
├── 96. Bridge to Project 11: Sequence-to-Sequence models
├── 97. Bridge to Project 12: Attention Mechanisms (Bahdanau et al.)
├── 98. Bridge to Project 13 & 14: Transformers & Large Language Models
├── 99. When GRUs remain the superior engineering choice today
├── 100. Best practices for production deployment
│
▼
MASTERY MILESTONE
│
└── 101. Complete production-grade GRU implementation from scratch
```

---

## Key Takeaways

1. **Lightweight Gated Powerhouse**:
   GRU delivers the long-term dependency capabilities of the LSTM while eliminating the separate cell state `C_t` and reducing parameter count by 25%.
2. **Coupled Forget and Input**:
   By using `(1 - z_t)` for memory retention and `z_t` for candidate writing, the update gate enforces an elegant zero-sum memory budget at every timestep.
3. **Linear State Interpolation**:
   The convex combination `h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t` guarantees that when `z_t \approx 0`, gradients propagate across hundreds of timesteps without exponential decay.
4. **Faster Training & Smaller Footprint**:
   Fewer matrix operations and smaller intermediate activation caches make GRU significantly faster to train and less memory-intensive than LSTM, often generalizing better on small-to-medium datasets.
5. **Architectural Continuity**:
   Understanding the progression from Simple RNN (`07-rnn-from-scratch`) to LSTM (`09-lstm`) to GRU (`10-gru`) establishes the core conceptual foundation for **Attention Mechanisms** (`12-attention`) and **Transformers** (`13-mini-transformer`).

---

## Next Steps

1. Create [`steps/step-01-mathematics.md`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/10-gru/steps/step-01-mathematics.md) detailing first-principles matrix calculus, the linear interpolation gradient proof, and step-by-step BPTT derivations.
2. Implement the pure NumPy vectorized GRU in [`src/gru.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/10-gru/src/gru.py) and execute direct 3-way benchmarking (RNN vs LSTM vs GRU).
