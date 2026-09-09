# Step 1 — Mathematics for Gated Recurrent Unit (GRU)

Goal: understand the complete mathematics of a **Gated Recurrent Unit (GRU)** network from first principles.
This document provides the rigorous theoretical foundations, the mathematical proof of the **Linear State Interpolation Gradient Highway**, gate calculus, Cho et al. vs PyTorch architectural nuances, fused matrix vectorization, step-by-step forward traces, and Backpropagation Through Time (BPTT) matrix calculus that will be implemented with pure NumPy from scratch.

---

## Q1 — Why GRU? Streamlining LSTM Without Sacrificing Long-Term Memory

In deep sequence learning, the simple Vanilla RNN (`07-rnn-from-scratch`) failed due to the **Vanishing Gradient Problem**: repeated Jacobian matrix multiplications `(W_{hh}^T)^{T-t}` cause error signals to exponentially vanish to zero over 8 to 10 timesteps.

The Long Short-Term Memory (LSTM, `09-lstm`) solved this problem by introducing a dedicated, linear **Cell State (`C_t`)** regulated by three separate gating mechanisms: Forget Gate (`f_t`), Input Gate (`i_t`), and Output Gate (`o_t`).

While the LSTM revolutionized NLP and speech recognition, it introduced significant computational and memory penalties:
1. **Memory Redundancy**: An LSTM maintains two full hidden vectors per timestep: the working hidden state `h_t \in \mathbb{R}^{1 \times H}` and the internal cell state `C_t \in \mathbb{R}^{1 \times H}`.
2. **Computational Overhead**: An LSTM requires four full matrix projections per timestep (`4 \times (D \times H + H^2 + H)` parameters), demanding eight matrix multiplications per step in un-fused form.
3. **Gating Redundancy**: In practice, the forget gate `f_t` and input gate `i_t` are often inversely correlated: when a model forgets old information, it almost always writes new information in its place. Maintaining two independent gates to control these coupled actions is often superfluous.

### The GRU Innovation (Cho et al., 2014)

Introduced by Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio in 2014, the **Gated Recurrent Unit (GRU)** preserves the long-term gradient preservation properties of the LSTM while dramatically streamlining the internal mechanics:

1. **Elimination of the Separate Cell State**: The GRU maintains **only one recurrent memory vector (`h_t`)**, which serves simultaneously as the persistent memory store and the external output representation.
2. **Coupled Gating (2 Gates Instead of 3)**:
   - **Reset Gate (`r_t`)**: Decides how much previous memory `h_{t-1}` to disregard when calculating new candidate concepts.
   - **Update Gate (`z_t`)**: Simultaneously controls memory retention (`1 - z_t`) and candidate assimilation (`z_t`), merging the roles of LSTM's forget and input gates into a single convex combination.
3. **25% Parameter Reduction**: The GRU reduces the recurrent parameter budget from `4 \times (DH + H^2 + H)` down to `3 \times (DH + H^2 + H)`, drastically decreasing GPU memory footprint, speeding up BPTT unrolling, and mitigating overfitting on small to medium-sized datasets.

---

## Q2 — The GRU Memory Mechanism: Linear State Interpolation as a Gradient Highway

The core innovation that enables a GRU to avoid the vanishing gradient problem without a separate cell state is its **additive convex combination update rule**:

```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
```

Where:
- `h_{t-1} \in (-1, 1)^{1 \times H}` is the prior hidden state.
- `\tilde{h}_t \in (-1, 1)^{1 \times H}` is the candidate hidden state.
- `z_t \in (0, 1)^{1 \times H}` is the update gate activation.
- `\odot` is the element-wise Hadamard product.

---

### The Mathematical Proof of Unbroken Gradient Flow

Let us compute the total Jacobian matrix `\frac{\partial h_t}{\partial h_{t-1}}` using the multivariate product rule:

```math
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(1 - z_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}} + (\tilde{h}_t - h_{t-1}) \odot \frac{\partial z_t}{\partial h_{t-1}}
```

Now consider an uninformative timestep, a long-distance semantic dependency, or a period where historical context must simply be remembered (for example, keeping a grammatical subject in memory across 30 intermediate modifying clauses):

1. The network learns to close the update gate for those dimensions:
   ```math
   z_t \to \mathbf{0}
   ```
2. The state equation collapses to identity passthrough:
   ```math
   h_t \approx (1 - \mathbf{0}) \odot h_{t-1} + \mathbf{0} \odot \tilde{h}_t = h_{t-1}
   ```
3. Because `z_t \to \mathbf{0}`, the non-linear candidate derivative term `z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}` vanishes to zero.
4. Because `z_t = \sigma(z_{\mathrm{pre}, t}) \to 0`, the gate derivative `\frac{\partial z_t}{\partial h_{t-1}} = z_t (1 - z_t) W_{hz}^T \to \mathbf{0}` also vanishes to zero!
5. Therefore, the Jacobian matrix simplifies directly to the **Identity Matrix**:
   ```math
   \frac{\partial h_t}{\partial h_{t-1}} \approx \operatorname{diag}(1 - \mathbf{0}) = \mathbf{I}
   ```

Over `T - t` consecutive timesteps where `z_j \approx \mathbf{0}`:

```math
\frac{\partial h_T}{\partial h_t} = \prod_{j=t+1}^{T} \frac{\partial h_j}{\partial h_{j-1}} \approx \prod_{j=t+1}^{T} \mathbf{I} = \mathbf{I}
```

#### Physical Meaning:
The error gradient `\frac{\partial L}{\partial h_T}` flows backward across tens or hundreds of timesteps **without any matrix multiplication by `W_{hh}` and without passing through saturating `\tanh'` derivatives**!
The term `(1 - z_t) \odot h_{t-1}` acts as an uninterrupted **linear gradient highway**, granting the GRU the same mathematical superpower as the LSTM's Constant Error Carousel (CEC), but directly within the primary hidden state vector.

---

## Q3 — The Internal Gate Formulations and Functional Roles

A standard GRU cell contains three operational sub-networks:

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

### 1. The Reset Gate (`r_t`): Delimiter & Boundary Detector
Decides how much of the historical context in `h_{t-1}` should be visible when proposing new candidate concepts:

```math
r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \in (0, 1)^{1 \times H}
```

- When `r_{t, k} \to 1.0`: The `k`-th dimension of historical memory is fully retained and made available to the candidate state.
- When `r_{t, k} \to 0.0`: The `k`-th dimension of historical memory is completely wiped out, forcing the unit to act as if reading the first symbol of a fresh sequence (e.g., at sentence boundaries or paragraph shifts).

### 2. The Update Gate (`z_t`): Information Importance Governor
Determines the precise balance between maintaining historical knowledge and overwriting it with new candidate features:

```math
z_t = \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z) \in (0, 1)^{1 \times H}
```

- When `z_{t, k} \to 0.0`: The unit ignores incoming information and carries the prior memory `h_{t-1, k}` forward untouched (`1 - z_t \to 1.0`).
- When `z_{t, k} \to 1.0`: The unit completely overwrites prior memory with the newly computed candidate representation `\tilde{h}_{t, k}`.

### 3. The Candidate Hidden State (`\tilde{h}_t`): Proposed New Concept Representation
Computes candidate features from current input `x_t` and the **reset-modulated** previous state `r_t \odot h_{t-1}`:

```math
\tilde{h}_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h) \in (-1, 1)^{1 \times H}
```

Notice the placement of `r_t`:
- In standard RNN: candidate depends directly on `h_{t-1} W_{hh}`.
- In GRU: `h_{t-1}` is first filtered element-wise by `r_t` **before** the recurrent transformation `W_{hh}`. Obsolete contextual features are masked out before entering the non-linear projection.

### 4. Hidden State Synthesis (`h_t`): The Convex Combination
The new hidden state is computed as an exact linear interpolation:

```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \in (-1, 1)^{1 \times H}
```

---

## Q4 — Architectural Nuance: Cho et al. (2014) vs. PyTorch GRU

When writing a from-scratch GRU in pure NumPy, one encounters a critical architectural nuance between the original paper by Cho et al. (2014) and PyTorch's `torch.nn.GRU`.

### 1. Cho et al. (2014) Canonical Paper Formulation
In the original paper, the candidate hidden state is computed with a **single bias**:

```math
\tilde{h}_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h)
```

Here:
1. The reset gate modulates the previous state: `h_{\mathrm{reset}} = r_t \odot h_{t-1}`.
2. The modulated state undergoes matrix multiplication: `h_{\mathrm{reset}} W_{hh}`.
3. The single bias `b_h` is added.

### 2. PyTorch `torch.nn.GRU` Formulation
In PyTorch, to optimize CUDA kernel execution and fuse input projections across timesteps, separate biases are maintained for input and hidden projections:

```math
\tilde{h}_t = \tanh(x_t W_{xh} + b_{ih} + r_t \odot (h_{t-1} W_{hh} + b_{hh}))
```

Notice the crucial difference:
1. The recurrent projection `h_{t-1} W_{hh} + b_{hh}` is evaluated **first**.
2. The reset gate `r_t` multiplies the projected vector **after** the recurrent linear layer.
3. Two separate bias vectors `b_{ih}` and `b_{hh}` are maintained for each gate (a total of 6 bias vectors).

### Comparison Summary:

| Property | Cho et al. (2014) Canonical | PyTorch `torch.nn.GRU` |
| :--- | :--- | :--- |
| **Reset Gate Placement** | Multiplies `h_{t-1}` **before** `W_{hh}`: `(r_t \odot h_{t-1}) W_{hh}` | Multiplies recurrent projection **after** `W_{hh}`: `r_t \odot (h_{t-1} W_{hh} + b_{hh})` |
| **Bias Vectors** | Single bias per gate: `b_r, b_z, b_h` (3 vectors) | Double bias per gate: `b_{ir}, b_{hr}, b_{iz}, b_{hz}, b_{ih}, b_{hh}` (6 vectors) |
| **Total Biases** | `3 \times H` | `6 \times H` |
| **Theoretical Meaning** | Filters semantic context before projection | Scales affine representation after projection |

> [!TIP]
> In our mathematical derivations and concrete numerical walkthrough below, we implement the **Cho et al. (2014) canonical formulation**, which is theoretically pure and elegant. In `src/gru.py`, we implement both modes and provide a clean switch (`mode="canonical"` vs `mode="pytorch"`).

---

## Q5 — Fused Projection Vectorization: High-Performance Architecture

Evaluating an un-fused GRU requires 6 separate matrix multiplications per timestep:
- `x_t W_{xr}`, `h_{t-1} W_{hr}`
- `x_t W_{xz}`, `h_{t-1} W_{hz}`
- `x_t W_{xh}`, `(r_t \odot h_{t-1}) W_{hh}`

In high-performance deep learning libraries and our NumPy implementation, we employ **Fused Gate Vectorization**:

### 1. Fusing Reset and Update Gates (`W_{rz}`):
Because both `r_t` and `z_t` share the exact same inputs `[x_t, h_{t-1}]`, we concatenate their projection matrices into unified `2H` blocks:

```math
W_{x, rz} = \begin{bmatrix} W_{xr} & W_{xz} \end{bmatrix} \in \mathbb{R}^{D \times 2H}
```

```math
W_{h, rz} = \begin{bmatrix} W_{hr} & W_{hz} \end{bmatrix} \in \mathbb{R}^{H \times 2H}
```

```math
b_{rz} = \begin{bmatrix} b_r & b_z \end{bmatrix} \in \mathbb{R}^{1 \times 2H}
```

Then, both gates are evaluated with **just two matrix multiplications**:

```math
Z_{rz} = x_t W_{x, rz} + h_{t-1} W_{h, rz} + b_{rz} \in \mathbb{R}^{1 \times 2H}
```

We split `Z_{rz}` down the feature axis:

```math
z_{r, \mathrm{pre}}, z_{z, \mathrm{pre}} = \operatorname{split}(Z_{rz}, 2, \mathrm{axis}=-1)
```

```math
r_t = \sigma(z_{r, \mathrm{pre}}), \quad z_t = \sigma(z_{z, \mathrm{pre}})
```

### 2. Sequence-Level Input GEMM Optimization
Notice that the input projection for all three components:
- `x_t W_{xr}`, `x_t W_{xz}`, `x_t W_{xh}`

**does not depend on any previous hidden state `h_{t-1}`**!
Therefore, given an entire input sequence `X \in \mathbb{R}^{B \times T \times D}`, we can precompute the full input projection across all timesteps simultaneously with a **single massive batched matrix multiplication** before entering the sequential recurrent loop:

```math
W_x = \begin{bmatrix} W_{xr} & W_{xz} & W_{xh} \end{bmatrix} \in \mathbb{R}^{D \times 3H}
```

```math
X_{\mathrm{proj}} = X W_x \in \mathbb{R}^{B \times T \times 3H}
```

Inside the sequential timestep loop, only the recurrent dot products `h_{t-1} W_{h, rz}` and `(r_t \odot h_{t-1}) W_{hh}` need to be computed! This optimization accelerates training and inference throughput by over 40%.

---

## Q6 — Parameters, Tensor Shapes, and Parameter Budget

Let us rigorously derive the complete parameter count for a GRU cell and contrast it directly with Simple RNN and LSTM:

Let:
- `D`: Input feature dimension.
- `H`: Hidden state dimension.
- `K`: Number of output prediction classes.

### Parameter Breakdown of GRU Cell:

1. **Reset Gate (`r_t`)**:
   - Input weights `W_{xr}`: `D \times H`
   - Recurrent weights `W_{hr}`: `H \times H = H^2`
   - Bias `b_r`: `1 \times H`
   - Subtotal = `DH + H^2 + H`

2. **Update Gate (`z_t`)**:
   - Input weights `W_{xz}`: `D \times H`
   - Recurrent weights `W_{hz}`: `H \times H = H^2`
   - Bias `b_z`: `1 \times H`
   - Subtotal = `DH + H^2 + H`

3. **Candidate Hidden State (`\tilde{h}_t`)**:
   - Input weights `W_{xh}`: `D \times H`
   - Recurrent weights `W_{hh}`: `H \times H = H^2`
   - Bias `b_h`: `1 \times H`
   - Subtotal = `DH + H^2 + H`

4. **Output Classification Layer (`y_t`)**:
   - Projection weights `W_{hy}`: `H \times K`
   - Bias `b_y`: `1 \times K`
   - Subtotal = `HK + K`

### Total GRU Parameter Formula:

```math
P_{\mathrm{GRU}} = 3 \times (D \times H + H^2 + H) + (H \times K + K)
```

---

### Comparative Architecture Audit:

| Metric / Parameter | Simple RNN (`07-rnn`) | LSTM (`09-lstm`) | GRU (`10-gru`) |
| :--- | :--- | :--- | :--- |
| **Number of Sub-Networks** | 1 (`h`) | 4 (`f, i, \tilde{C}, o`) | **3 (`r, z, \tilde{h}`)** |
| **Recurrent Weights Formula** | `DH + H^2 + H` | `4(DH + H^2 + H)` | **`3(DH + H^2 + H)`** |
| **Output Weights Formula** | `HK + K` | `HK + K` | **`HK + K`** |
| **Total for `D=2, H=3, K=2`** | `2(3)+9+3 + 6+2 = \mathbf{26}` | `4(6+9+3) + 6+2 = \mathbf{80}` | **`3(6+9+3) + 6+2 = \mathbf{62}`** |
| **Total for `D=128, H=256, K=10`** | `98,570` | `394,250` | **`295,690` (25% fewer)** |
| **Hidden States per Step** | 1 (`h_t`) | 2 (`h_t, C_t`) | **1 (`h_t`)** |
| **BPTT Caching Memory** | Lowest | Highest | **Intermediate** |

---

## Q7 — Concrete Numerical Setup (Matching verify_gru.py)

To guarantee absolute mathematical verification down to 6 decimal places, we define a concrete, reproducible test setup:

### Dimensions:
- Batch Size `B = 1`
- Sequence Length `T = 3`
- Input Dimension `D = 2`
- Hidden Dimension `H = 3`
- Output Classes `K = 2`

---

### Input Sequence `X \in \mathbb{R}^{1 \times 3 \times 2}`:

```math
X = \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix} = \begin{bmatrix}
1.0 & -0.5 \\
0.5 &  1.0 \\
-1.0 & 0.5
\end{bmatrix}
```

---

### One-Hot Target Sequence `Y \in \mathbb{R}^{1 \times 3 \times 2}`:

```math
Y = \begin{bmatrix}
y_1 \\
y_2 \\
y_3
\end{bmatrix} = \begin{bmatrix}
0.0 & 1.0 \quad (\text{Class 1}) \\
1.0 & 0.0 \quad (\text{Class 0}) \\
0.0 & 1.0 \quad (\text{Class 1})
\end{bmatrix}
```

---

### Weight Matrices and Biases:

#### 1. Reset Gate Parameters:
```math
W_{xr} = \begin{bmatrix} 0.2 & -0.1 & 0.3 \\ -0.3 & 0.4 & 0.1 \end{bmatrix}, \quad
W_{hr} = \begin{bmatrix} 0.1 & -0.2 & 0.1 \\ 0.3 & 0.1 & -0.1 \\ -0.2 & 0.2 & 0.4 \end{bmatrix}, \quad
b_r = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}
```

#### 2. Update Gate Parameters:
```math
W_{xz} = \begin{bmatrix} -0.2 & 0.3 & -0.1 \\ 0.4 & -0.2 & 0.3 \end{bmatrix}, \quad
W_{hz} = \begin{bmatrix} -0.1 & 0.3 & 0.2 \\ 0.2 & -0.1 & 0.1 \\ 0.1 & 0.2 & -0.3 \end{bmatrix}, \quad
b_z = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}
```

#### 3. Candidate Hidden State Parameters:
```math
W_{xh} = \begin{bmatrix} 0.3 & 0.2 & -0.3 \\ -0.1 & 0.4 & 0.2 \end{bmatrix}, \quad
W_{hh} = \begin{bmatrix} 0.2 & 0.1 & -0.2 \\ -0.3 & 0.2 & 0.1 \\ 0.1 & -0.1 & 0.3 \end{bmatrix}, \quad
b_h = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}
```

#### 4. Output Classification Layer:
```math
W_{hy} = \begin{bmatrix} 0.4 & -0.3 \\ -0.2 & 0.5 \\ 0.3 & 0.1 \end{bmatrix}, \quad
b_y = \begin{bmatrix} 0.0 & 0.0 \end{bmatrix}
```

Initial recurrent hidden state:

```math
h_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}
```

---

## Q8 — Forward Pass: Step-by-Step Numerical Walkthrough

We now trace the forward computation across all 3 timesteps with exact arithmetic.

---

### Timestep 1 (`t = 1`):

#### 1. Inputs:
```text
x_1 = [1.0, -0.5]
h_0 = [0.0,  0.0,  0.0]
```

#### 2. Reset Gate Pre-activation and Activation:
```math
z_{r, 1} = x_1 W_{xr} + h_0 W_{hr} + b_r
         = [1.0(0.2) + (-0.5)(-0.3), 1.0(-0.1) + (-0.5)(0.4), 1.0(0.3) + (-0.5)(0.1)]
         = [0.2 + 0.15, -0.1 - 0.2, 0.3 - 0.05]
         = [0.35, -0.30, 0.25]
```
```math
r_1 = \sigma(z_{r, 1}) = [\sigma(0.35), \sigma(-0.30), \sigma(0.25)]
    = [0.586618, 0.425557, 0.562177]
```

#### 3. Update Gate Pre-activation and Activation:
```math
z_{z, 1} = x_1 W_{xz} + h_0 W_{hz} + b_z
         = [1.0(-0.2) + (-0.5)(0.4), 1.0(0.3) + (-0.5)(-0.2), 1.0(-0.1) + (-0.5)(0.3)]
         = [-0.2 - 0.2, 0.3 + 0.1, -0.1 - 0.15]
         = [-0.40, 0.40, -0.25]
```
```math
z_1 = \sigma(z_{z, 1}) = [\sigma(-0.40), \sigma(0.40), \sigma(-0.25)]
    = [0.401312, 0.598688, 0.437823]
```

#### 4. Candidate Hidden State:
```math
h_{\mathrm{reset}, 0} = r_1 \odot h_0 = [0.0, 0.0, 0.0]
```
```math
z_{h, 1} = x_1 W_{xh} + h_{\mathrm{reset}, 0} W_{hh} + b_h
         = [1.0(0.3) + (-0.5)(-0.1), 1.0(0.2) + (-0.5)(0.4), 1.0(-0.3) + (-0.5)(0.2)]
         = [0.3 + 0.05, 0.2 - 0.2, -0.3 - 0.1]
         = [0.35, 0.00, -0.40]
```
```math
\tilde{h}_1 = \tanh(z_{h, 1}) = [\tanh(0.35), \tanh(0.00), \tanh(-0.40)]
            = [0.336376, 0.000000, -0.379949]
```

#### 5. New Hidden State Update:
```math
h_1 = (1 - z_1) \odot h_0 + z_1 \odot \tilde{h}_1
    = [0.0, 0.0, 0.0] + [0.401312(0.336376), 0.598688(0.0), 0.437823(-0.379949)]
    = [0.134992, 0.000000, -0.166351]
```

#### 6. Output Projection, Softmax, and Loss:
```math
o_1 = h_1 W_{hy} + b_y = [0.134992(0.4) + (-0.166351)(0.3), 0.134992(-0.3) + (-0.166351)(0.1)]
    = [0.053997 - 0.049905, -0.040498 - 0.016635]
    = [0.004091, -0.057133]
```
```math
\hat{y}_1 = \operatorname{softmax}([0.004091, -0.057133]) = [0.515301, 0.484699]
```
```math
L_1 = -\ln(\hat{y}_{1, \mathrm{class} 1}) = -\ln(0.484699) = 0.724228
```

---

### Timestep 2 (`t = 2`):

#### 1. Inputs:
```text
x_2 = [0.5, 1.0]
h_1 = [0.134992, 0.000000, -0.166351]
```

#### 2. Reset Gate:
```math
z_{r, 2} = x_2 W_{xr} + h_1 W_{hr} + b_r
```
- `x_2 W_{xr} = [0.5(0.2) + 1.0(-0.3), 0.5(-0.1) + 1.0(0.4), 0.5(0.3) + 1.0(0.1)] = [-0.20, 0.35, 0.25]`
- `h_1 W_{hr} = [0.134992(0.1) - 0.166351(-0.2), 0.134992(-0.2) - 0.166351(0.2), 0.134992(0.1) - 0.166351(0.4)] = [0.046769, -0.060268, -0.053041]`
- `z_{r, 2} = [-0.20 + 0.046769, 0.35 - 0.060268, 0.25 - 0.053041] = [-0.153231, 0.289732, 0.196959]`
```math
r_2 = \sigma([-0.153231, 0.289732, 0.196959]) = [0.461767, 0.571930, 0.549081]
```

#### 3. Update Gate:
```math
z_{z, 2} = x_2 W_{xz} + h_1 W_{hz} + b_z
```
- `x_2 W_{xz} = [0.5(-0.2) + 1.0(0.4), 0.5(0.3) + 1.0(-0.2), 0.5(-0.1) + 1.0(0.3)] = [0.30, -0.05, 0.25]`
- `h_1 W_{hz} = [0.134992(-0.1) - 0.166351(0.1), 0.134992(0.3) - 0.166351(0.2), 0.134992(0.2) - 0.166351(-0.3)] = [-0.030134, 0.007227, 0.076904]`
- `z_{z, 2} = [0.30 - 0.030134, -0.05 + 0.007227, 0.25 + 0.076904] = [0.269866, -0.042773, 0.326904]`
```math
z_2 = \sigma([0.269866, -0.042773, 0.326904]) = [0.567060, 0.489308, 0.581006]
```

#### 4. Candidate Hidden State:
```math
h_{\mathrm{reset}, 1} = r_2 \odot h_1 = [0.461767(0.134992), 0.571930(0.0), 0.549081(-0.166351)]
                     = [0.062335, 0.000000, -0.091340]
```
```math
z_{h, 2} = x_2 W_{xh} + h_{\mathrm{reset}, 1} W_{hh} + b_h
```
- `x_2 W_{xh} = [0.5(0.3) + 1.0(-0.1), 0.5(0.2) + 1.0(0.4), 0.5(-0.3) + 1.0(0.2)] = [0.05, 0.50, 0.05]`
- `h_{\mathrm{reset}, 1} W_{hh} = [0.062335(0.2) - 0.091340(0.1), 0.062335(0.1) - 0.091340(-0.1), 0.062335(-0.2) - 0.091340(0.3)] = [0.003333, 0.015367, -0.039869]`
- `z_{h, 2} = [0.05 + 0.003333, 0.50 + 0.015367, 0.05 - 0.039869] = [0.053333, 0.515367, 0.010131]`
```math
\tilde{h}_2 = \tanh([0.053333, 0.515367, 0.010131]) = [0.053282, 0.474117, 0.010131]
```

#### 5. New Hidden State:
```math
h_2 = (1 - z_2) \odot h_1 + z_2 \odot \tilde{h}_2
    = [0.432940(0.134992) + 0.567060(0.053282), 0.510692(0.0) + 0.489308(0.474117), 0.418994(-0.166351) + 0.581006(0.010131)]
    = [0.058444 + 0.030214, 0.0 + 0.231989, -0.069700 + 0.005886]
    = [0.088658, 0.231989, -0.063814]
```

#### 6. Output Projection and Loss:
```math
o_2 = h_2 W_{hy} + b_y = [-0.030079, 0.083016]
```
```math
\hat{y}_2 = \operatorname{softmax}([-0.030079, 0.083016]) = [0.471756, 0.528244]
```
```math
L_2 = -\ln(\hat{y}_{2, \mathrm{class} 0}) = -\ln(0.471756) = 0.751293
```

---

### Timestep 3 (`t = 3`):

#### 1. Inputs:
```text
x_3 = [-1.0, 0.5]
h_2 = [0.088658, 0.231989, -0.063814]
```

#### 2. Reset Gate:
```math
z_{r, 3} = x_3 W_{xr} + h_2 W_{hr} + b_r = [-0.35, 0.30, -0.25] + [0.091225, -0.007295, -0.039859]
         = [-0.258775, 0.292705, -0.289859]
```
```math
r_3 = \sigma(z_{r, 3}) = [0.435665, 0.572658, 0.428038]
```

#### 3. Update Gate:
```math
z_{z, 3} = x_3 W_{xz} + h_2 W_{hz} + b_z = [0.40, -0.40, 0.25] + [0.031151, -0.009364, 0.060075]
         = [0.431151, -0.409364, 0.310075]
```
```math
z_3 = \sigma(z_{z, 3}) = [0.606148, 0.399065, 0.576903]
```

#### 4. Candidate Hidden State:
```math
h_{\mathrm{reset}, 2} = r_3 \odot h_2 = [0.038625, 0.132851, -0.027315]
```
```math
z_{h, 3} = x_3 W_{xh} + h_{\mathrm{reset}, 2} W_{hh} + b_h = [-0.35, 0.00, 0.40] + [-0.034862, 0.033164, -0.002634]
         = [-0.384862, 0.033164, 0.397366]
```
```math
\tilde{h}_3 = \tanh(z_{h, 3}) = [-0.366922, 0.033152, 0.377693]
```

#### 5. New Hidden State:
```math
h_3 = (1 - z_3) \odot h_2 + z_3 \odot \tilde{h}_3
    = [0.393852(0.088658) + 0.606148(-0.366922), 0.600935(0.231989) + 0.399065(0.033152), 0.423097(-0.063814) + 0.576903(0.377693)]
    = [-0.187491, 0.152640, 0.190893]
```

#### 6. Output Projection and Loss:
```math
o_3 = h_3 W_{hy} + b_y = [-0.048257, 0.151657]
```
```math
\hat{y}_3 = \operatorname{softmax}([-0.048257, 0.151657]) = [0.450187, 0.549813]
```
```math
L_3 = -\ln(\hat{y}_{3, \mathrm{class} 1}) = -\ln(0.549813) = 0.598178
```

---

## Q9 — Total Sequence Loss

For sequential classification, the total loss is the unweighted sum of categorical cross-entropy losses across all 3 timesteps:

```math
L_{\mathrm{seq}} = \sum_{t=1}^{3} L_t = L_1 + L_2 + L_3
```

Substituting our exact values:

```math
L_{\mathrm{seq}} = 0.724228 + 0.751293 + 0.598178 = 2.073698
```

---

## Q10 — Backpropagation Through Time (BPTT) for GRU: First-Principles Matrix Calculus

In GRU BPTT, the error signal propagates backward through both the **Linear State Highway** and the **Reset-Modulated Recurrent Non-linearities**.

---

### 1. Output Classification Gradients
At each timestep `t`, the upstream gradient from softmax cross-entropy is:

```math
\delta_{\mathrm{logit}, t} = \hat{y}_t - y_t \in \mathbb{R}^{1 \times K}
```

This contributes to the output projection parameters:

```math
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T h_t^T \delta_{\mathrm{logit}, t} \in \mathbb{R}^{H \times K}
```

```math
\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \delta_{\mathrm{logit}, t} \in \mathbb{R}^{1 \times K}
```

---

### 2. Total Gradient With Respect to Hidden State `h_t`
The total error arriving at hidden state `h_t` originates from two sources:
1. The immediate classification loss at step `t`: `\delta_{\mathrm{logit}, t} W_{hy}^T`
2. The recurrent error propagating backward from step `t+1`: `\delta_{h_t, \mathrm{recurrent}}`

```math
\delta_{h_t} = \frac{\partial L}{\partial h_t} = \delta_{\mathrm{logit}, t} W_{hy}^T + \delta_{h_t, \mathrm{recurrent}} \in \mathbb{R}^{1 \times H}
```

---

### 3. Gradient With Respect to Candidate State `\tilde{h}_t`
From the state equation `h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t`:

```math
\frac{\partial L}{\partial \tilde{h}_t} = \delta_{h_t} \odot z_t \in \mathbb{R}^{1 \times H}
```

Propagating through the candidate activation `\tanh(z_{h, t})`:

```math
\delta_{z_{h, t}} = \frac{\partial L}{\partial z_{h, t}} = \frac{\partial L}{\partial \tilde{h}_t} \odot (1 - \tilde{h}_t^2) = \delta_{h_t} \odot z_t \odot (1 - \tilde{h}_t^2) \in \mathbb{R}^{1 \times H}
```

Accumulating candidate parameters:
```math
\frac{\partial L}{\partial W_{xh}} += x_t^T \delta_{z_{h, t}}, \quad
\frac{\partial L}{\partial W_{hh}} += (r_t \odot h_{t-1})^T \delta_{z_{h, t}}, \quad
\frac{\partial L}{\partial b_h} += \delta_{z_{h, t}}
```

---

### 4. Gradient With Respect to Reset Gate `r_t`
Notice that `r_t` enters the candidate pre-activation via `h_{\mathrm{reset}, t-1} = r_t \odot h_{t-1}`:

```math
\delta_{h_{\mathrm{reset}, t-1}} = \frac{\partial L}{\partial h_{\mathrm{reset}, t-1}} = \delta_{z_{h, t}} W_{hh}^T \in \mathbb{R}^{1 \times H}
```

By the chain rule with respect to `r_t`:

```math
\frac{\partial L}{\partial r_t} = \delta_{h_{\mathrm{reset}, t-1}} \odot h_{t-1} = (\delta_{z_{h, t}} W_{hh}^T) \odot h_{t-1} \in \mathbb{R}^{1 \times H}
```

Propagating through the Sigmoid activation `r_t = \sigma(z_{r, t})`:

```math
\delta_{z_{r, t}} = \frac{\partial L}{\partial z_{r, t}} = \frac{\partial L}{\partial r_t} \odot r_t \odot (1 - r_t) \in \mathbb{R}^{1 \times H}
```

Accumulating reset parameters:
```math
\frac{\partial L}{\partial W_{xr}} += x_t^T \delta_{z_{r, t}}, \quad
\frac{\partial L}{\partial W_{hr}} += h_{t-1}^T \delta_{z_{r, t}}, \quad
\frac{\partial L}{\partial b_r} += \delta_{z_{r, t}}
```

---

### 5. Gradient With Respect to Update Gate `z_t`
Notice that `z_t` appears directly in the state interpolation `h_t = h_{t-1} + z_t \odot (\tilde{h}_t - h_{t-1})`:

```math
\frac{\partial L}{\partial z_t} = \delta_{h_t} \odot (\tilde{h}_t - h_{t-1}) \in \mathbb{R}^{1 \times H}
```

Propagating through the Sigmoid activation `z_t = \sigma(z_{z, t})`:

```math
\delta_{z_{z, t}} = \frac{\partial L}{\partial z_{z, t}} = \frac{\partial L}{\partial z_t} \odot z_t \odot (1 - z_t) \in \mathbb{R}^{1 \times H}
```

Accumulating update parameters:
```math
\frac{\partial L}{\partial W_{xz}} += x_t^T \delta_{z_{z, t}}, \quad
\frac{\partial L}{\partial W_{hz}} += h_{t-1}^T \delta_{z_{z, t}}, \quad
\frac{\partial L}{\partial b_z} += \delta_{z_{z, t}}
```

---

### 6. The 4-Way Gradient Highway Propagating to Prior Hidden State `h_{t-1}`
Where did `h_{t-1}` appear in the forward computation of timestep `t`?
It influenced the computation through **four distinct mathematical pathways**:

```math
\delta_{h_{t-1}, \mathrm{recurrent}} = \frac{\partial L}{\partial h_{t-1}} = \sum_{k=1}^4 \text{Path}_k
```

1. **Path 1 — Direct Linear State Highway**:
   ```math
   \text{Path}_1 = \delta_{h_t} \odot (1 - z_t)
   ```
2. **Path 2 — Candidate State Highway (via Reset Modulation)**:
   ```math
   \text{Path}_2 = \delta_{h_{\mathrm{reset}, t-1}} \odot r_t = (\delta_{z_{h, t}} W_{hh}^T) \odot r_t
   ```
3. **Path 3 — Update Gate Recurrent Projection**:
   ```math
   \text{Path}_3 = \delta_{z_{z, t}} W_{hz}^T
   ```
4. **Path 4 — Reset Gate Recurrent Projection**:
   ```math
   \text{Path}_4 = \delta_{z_{r, t}} W_{hr}^T
   ```

Summing all 4 paths yields the complete recurrent error vector:

```math
\delta_{h_{t-1}, \mathrm{recurrent}} = \delta_{h_t} \odot (1 - z_t) + (\delta_{z_{h, t}} W_{hh}^T) \odot r_t + \delta_{z_{z, t}} W_{hz}^T + \delta_{z_{r, t}} W_{hr}^T
```

> [!IMPORTANT]
> When `z_t \approx \mathbf{0}`, Path 1 dominates: `\delta_{h_{t-1}} \approx \delta_{h_t}`. Error flows completely unhindered backward through time.

---

## Q11 — Step-by-Step Numerical BPTT Walkthrough

We now trace the backward calculations from `t = 3` down to `t = 1` matching [`scratch/verify_gru.py`](file:///Users/pasan/.gemini/antigravity-ide/brain/d3b699e8-4410-4b8c-b2cb-728045b5ca32/scratch/verify_gru.py) to 6 decimal places.

---

### Backward Timestep 3 (`t = 3`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 3} = \hat{y}_3 - y_3 = [0.450187, 0.549813] - [0.0, 1.0] = [0.450187, -0.450187]
```

#### 2. Hidden State Error:
Since `t = 3` is the final timestep, `\delta_{h_3, \mathrm{recurrent}} = \mathbf{0}`:
```math
\delta_{h_3} = \delta_{\mathrm{logit}, 3} W_{hy}^T = [0.450187(0.4) - 0.450187(-0.3), 0.450187(-0.2) - 0.450187(0.5), 0.450187(0.3) - 0.450187(0.1)]
             = [0.315131, -0.315131, 0.090037]
```

#### 3. Candidate Pre-activation Error:
```math
\delta_{z_{h, 3}} = \delta_{h_3} \odot z_3 \odot (1 - \tilde{h}_3^2)
```
- `\delta_{h_3} \odot z_3 = [0.315131(0.606148), -0.315131(0.399065), 0.090037(0.576903)] = [0.191016, -0.125758, 0.051943]`
- `1 - \tilde{h}_3^2 = [1 - (-0.366922)^2, 1 - (0.033152)^2, 1 - (0.377693)^2] = [0.865368, 0.998901, 0.857348]`
- `\delta_{z_{h, 3}} = [0.165299, -0.125619, 0.044533]`

#### 4. Reset Modulation Error and Reset Gate Error:
```math
\delta_{h_{\mathrm{reset}, 2}} = \delta_{z_{h, 3}} W_{hh}^T
                              = [0.165299(0.2) - 0.125619(0.1) + 0.044533(-0.2), \dots]
                              = [0.011591, -0.070260, 0.042452]
```
```math
\delta_{r_3} = \delta_{h_{\mathrm{reset}, 2}} \odot h_2 = [0.011591(0.088658), -0.070260(0.231989), 0.042452(-0.063814)]
             = [0.001028, -0.016300, -0.002709]
```
```math
\delta_{z_{r, 3}} = \delta_{r_3} \odot r_3 \odot (1 - r_3) = [0.000253, -0.003989, -0.000663]
```

#### 5. Update Gate Error:
```math
\delta_{z_3} = \delta_{h_3} \odot (\tilde{h}_3 - h_2) = [0.315131(-0.455580), -0.315131(-0.198837), 0.090037(0.441507)]
             = [-0.143568, 0.062659, 0.039752]
```
```math
\delta_{z_{z, 3}} = \delta_{z_3} \odot z_3 \odot (1 - z_3) = [-0.034274, 0.015027, 0.009703]
```

#### 6. Propagating Backward to `h_2`:
```math
\text{Path}_1 = \delta_{h_3} \odot (1 - z_3) = [0.124115, -0.189373, 0.038095]
```
```math
\text{Path}_2 = \delta_{h_{\mathrm{reset}, 2}} \odot r_3 = [0.005050, -0.040235, 0.018171]
```
```math
\text{Path}_3 = \delta_{z_{z, 3}} W_{hz}^T = [0.009876, -0.007387, -0.003333]
```
```math
\text{Path}_4 = \delta_{z_{r, 3}} W_{hr}^T = [0.000757, -0.000257, -0.001114]
```
```math
\delta_{h_2, \mathrm{recurrent}} = \sum_{k=1}^4 \text{Path}_k = [0.139798, -0.237253, 0.051819]
```

---

### Backward Timestep 2 (`t = 2`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 2} = \hat{y}_2 - y_2 = [0.471756, 0.528244] - [1.0, 0.0] = [-0.528244, 0.528244]
```

#### 2. Hidden State Error (Merged with Highway):
```math
\delta_{h_2} = \delta_{\mathrm{logit}, 2} W_{hy}^T + \delta_{h_2, \mathrm{recurrent}}
             = [-0.369771, 0.369771, -0.105649] + [0.139798, -0.237253, 0.051819]
             = [-0.229973, 0.132518, -0.053830]
```

#### 3. Intermediate Gate Errors:
- Candidate pre-activation error: `\delta_{z_{h, 2}} = [-0.130038, 0.050266, -0.031272]`
- Reset modulation error: `\delta_{h_{\mathrm{reset}, 1}} = [-0.014727, 0.045938, -0.027412]`
- Reset gate pre-activation error: `\delta_{z_{r, 2}} = [-0.000494, 0.000000, 0.001129]`
- Update gate pre-activation error: `\delta_{z_{z, 2}} = [0.004613, 0.015700, -0.002313]`

#### 4. Propagating Backward to `h_1`:
```math
\text{Path}_1 = \delta_{h_2} \odot (1 - z_2) = [-0.099565, 0.067676, -0.022554]
```
```math
\text{Path}_2 = \delta_{h_{\mathrm{reset}, 1}} \odot r_2 = [-0.006800, 0.026273, -0.015051]
```
```math
\text{Path}_3 = \delta_{z_{z, 2}} W_{hz}^T = [0.003786, -0.000879, 0.004295]
```
```math
\text{Path}_4 = \delta_{z_{r, 2}} W_{hr}^T = [0.000063, -0.000261, 0.000550]
```
```math
\delta_{h_1, \mathrm{recurrent}} = \sum_{k=1}^4 \text{Path}_k = [-0.102515, 0.092809, -0.032760]
```

---

### Backward Timestep 1 (`t = 1`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 1} = \hat{y}_1 - y_1 = [0.515301, 0.484699] - [0.0, 1.0] = [0.515301, -0.515301]
```

#### 2. Hidden State Error:
```math
\delta_{h_1} = \delta_{\mathrm{logit}, 1} W_{hy}^T + \delta_{h_1, \mathrm{recurrent}}
             = [0.360711, -0.360711, 0.103060] + [-0.102515, 0.092809, -0.032760]
             = [0.258196, -0.267902, 0.070300]
```

#### 3. Intermediate Gate Errors:
- Candidate pre-activation error: `\delta_{z_{h, 1}} = [0.091893, -0.160389, 0.026336]`
- Reset modulation error: `\delta_{h_{\mathrm{reset}, 0}} = [-0.002927, -0.057012, 0.033129]`
- Reset gate pre-activation error: `\delta_{z_{r, 1}} = [0.0, 0.0, 0.0]` (since `h_0 = \mathbf{0}`)
- Update gate pre-activation error: `\delta_{z_{z, 1}} = [0.020867, 0.0, -0.006574]`

---

### Final Accumulated Parameter Gradients Across Sequence

#### 1. Output Classification Layer:
```math
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^3 h_t^T \delta_{\mathrm{logit}, t} = \begin{bmatrix}
-0.061678 & 0.061678 \\
-0.053830 & 0.053830 \\
 0.033926 & -0.033926
\end{bmatrix}, \quad
\frac{\partial L}{\partial b_y} = \sum_{t=1}^3 \delta_{\mathrm{logit}, t} = \begin{bmatrix} 0.437245 & -0.437245 \end{bmatrix}
```

#### 2. Reset Gate Parameters:
```math
\frac{\partial L}{\partial W_{xr}} = \begin{bmatrix}
-0.000500 &  0.003989 & 0.001228 \\
-0.000368 & -0.001994 & 0.000797
\end{bmatrix}, \quad
\frac{\partial L}{\partial W_{hr}} = \begin{bmatrix}
-0.000044 & -0.000354 &  0.000094 \\
 0.000059 & -0.000925 & -0.000154 \\
 0.000066 &  0.000255 & -0.000145
\end{bmatrix}, \quad
\frac{\partial L}{\partial b_r} = \begin{bmatrix} -0.000241 & -0.003989 & 0.000466 \end{bmatrix}
```

#### 3. Update Gate Parameters:
```math
\frac{\partial L}{\partial W_{xz}} = \begin{bmatrix}
 0.057448 & -0.007177 & -0.017434 \\
-0.022957 &  0.023213 &  0.005826
\end{bmatrix}, \quad
\frac{\partial L}{\partial W_{hz}} = \begin{bmatrix}
-0.002416 &  0.003452 &  0.000548 \\
-0.007951 &  0.003486 &  0.002251 \\
 0.001420 & -0.003571 & -0.000234
\end{bmatrix}, \quad
\frac{\partial L}{\partial b_z} = \begin{bmatrix} -0.008794 & 0.030727 & 0.000816 \end{bmatrix}
```

#### 4. Candidate Hidden State Parameters:
```math
\frac{\partial L}{\partial W_{xh}} = \begin{bmatrix}
-0.138426 & -0.009637 & -0.033834 \\
-0.093335 &  0.067651 & -0.022173
\end{bmatrix}, \quad
\frac{\partial L}{\partial W_{hh}} = \begin{bmatrix}
-0.001721 & -0.001719 & -0.000229 \\
 0.021960 & -0.016689 &  0.005916 \\
 0.007363 & -0.001160 &  0.001640
\end{bmatrix}, \quad
\frac{\partial L}{\partial b_h} = \begin{bmatrix} 0.127154 & -0.235742 & 0.039597 \end{bmatrix}
```

---

## Q12 — Complete Mathematical Mapping to gru.py Implementation

This cross-reference connects every mathematical equation derived above directly to its implementation in [`src/gru.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/10-gru/src/gru.py):

| Component / Function | Mathematical Equation | NumPy Variables & Dimensions |
| :--- | :--- | :--- |
| `gru_cell_forward(x_t, h_prev)` | Fused gate projection `Z_{rz} = x_t W_{x, rz} + h_{t-1} W_{h, rz} + b_{rz}` | `x_t: (B, D)`, `h_prev: (B, H)` -> `Z_rz: (B, 2H)` |
| `np.split(Z_rz, 2, axis=-1)` | Slices `z_{r, \mathrm{pre}}, z_{z, \mathrm{pre}}` | Splits `(B, 2H)` into two `(B, H)` arrays |
| Gate activations | `r = \sigma(z_r)`, `z = \sigma(z_z)` | `r: (B, H)`, `z: (B, H)` |
| Context modulation | `h_{\mathrm{reset}} = r \odot h_{t-1}` | `h_reset: (B, H)` |
| Candidate state | `\tilde{h}_t = \tanh(x_t W_{xh} + h_{\mathrm{reset}} W_{hh} + b_h)` | `h_cand: (B, H)` |
| State synthesis | `h_t = (1 - z) \odot h_{t-1} + z \odot \tilde{h}_t` | `h_t: (B, H)` |
| `gru_forward(X, h_0)` | Unrolls cell over `T` timesteps; caches all gates | `X: (B, T, D)` -> `H: (B, T, H)` |
| `dense_output(h_t, W_hy, b_y)` | `o_t = h_t W_{hy} + b_y` | `h_t: (B, H)` -> `logits: (B, K)` |
| `gru_backward(d_logits, cache)` | BPTT across `T`: computes 4-path highway error `\delta_{h_t}` | Upstream loss -> parameter gradients |
| Parameter updates | `dW_x += x_t^T \delta_z`, `dW_h += h_{\mathrm{context}}^T \delta_z`, `db += \delta_z` | `dW_x: (D, H)`, `dW_h: (H, H)`, `db: (1, H)` |

---

## Summary of the Complete Mathematical Pipeline

```text
========================================================================================================================
FORWARD GRU CELL PIPELINE (One Timestep)
========================================================================================================================
1. Fused Gate Pre-activations:         Z_rz = x_t @ W_x_rz + h_{t-1} @ W_h_rz + b_rz    Z_rz in R^(B x 2H)
2. Slicing Gates:                      zr_pre, zz_pre = split(Z_rz, 2)                  Each in R^(B x H)
3. Gate Activations:                   r = sigmoid(zr_pre)                              r in (0, 1) [Reset Gate]
                                       z = sigmoid(zz_pre)                              z in (0, 1) [Update Gate]
4. Reset-Modulated Context:            h_reset = r * h_{t-1}                            h_reset in R^(B x H)
5. Candidate State:                    zh_pre = x_t @ W_xh + h_reset @ W_hh + b_h       zh_pre in R^(B x H)
                                       h_cand = tanh(zh_pre)                            h_cand in (-1, 1)
6. Convex State Interpolation:         h_t = (1 - z) * h_{t-1} + z * h_cand             h_t in (-1, 1)
7. Output Projection & Softmax:        o_t = h_t @ W_hy + b_y                           o_t in R^(B x K)
                                       y_hat = softmax(o_t)                             y_hat in (0, 1)
========================================================================================================================
BACKWARD GRU BPTT PIPELINE (Reverse Time)
========================================================================================================================
1. Output Logit Error:                 d_logit = y_hat - Y_t                            d_logit in R^(B x K)
2. Output Weight Gradients:            dW_hy += h_t.T @ d_logit                         dW_hy in R^(H x K)
                                       db_y  += sum(d_logit)                            db_y in R^(1 x K)
3. Total Hidden Error:                 d_h = d_logit @ W_hy.T + d_h_recurrent           d_h in R^(B x H)
4. Candidate Pre-activation Error:     d_zh = d_h * z * (1 - h_cand^2)                  d_zh in R^(B x H)
                                       dW_xh += x_t.T @ d_zh                            dW_xh in R^(D x H)
                                       dW_hh += h_reset.T @ d_zh                        dW_hh in R^(H x H)
                                       db_h  += sum(d_zh)                               db_h in R^(1 x H)
5. Reset Modulation Error:             d_h_reset = d_zh @ W_hh.T                        d_h_reset in R^(B x H)
6. Reset Gate Pre-activation Error:    d_zr = (d_h_reset * h_{t-1}) * r * (1 - r)       d_zr in R^(B x H)
                                       dW_xr += x_t.T @ d_zr                            dW_xr in R^(D x H)
                                       dW_hr += h_{t-1}.T @ d_zr                        dW_hr in R^(H x H)
                                       db_r  += sum(d_zr)                               db_r in R^(1 x H)
7. Update Gate Pre-activation Error:   d_zz = (d_h * (h_cand - h_{t-1})) * z * (1 - z)  d_zz in R^(B x H)
                                       dW_xz += x_t.T @ d_zz                            dW_xz in R^(D x H)
                                       dW_hz += h_{t-1}.T @ d_zz                        dW_hz in R^(H x H)
                                       db_z  += sum(d_zz)                               db_z in R^(1 x H)
8. Propagate to Prior Step (t-1):
   Path 1 (Direct Highway):            Path_1 = d_h * (1 - z)
   Path 2 (Candidate via Reset):       Path_2 = d_h_reset * r
   Path 3 (Update Gate Recurrent):     Path_3 = d_zz @ W_hz.T
   Path 4 (Reset Gate Recurrent):      Path_4 = d_zr @ W_hr.T
   Total Recurrent Error:              d_h_recurrent = Path_1 + Path_2 + Path_3 + Path_4
========================================================================================================================
```

This completes the foundational mathematics for Project 10. The next step is validating
each of these formulas in executable NumPy code in [`src/gru.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/10-gru/src/gru.py).
