# Step 1 — Mathematics for Long Short-Term Memory (LSTM)

Goal: understand the complete mathematics of a **Long Short-Term Memory (LSTM)** network from first principles.
This document provides the rigorous theoretical foundations, the mathematical proof of the **Constant Error Carousel (CEC)**, gate calculus, fused matrix vectorization, step-by-step forward traces, and Backpropagation Through Time (BPTT) matrix calculus that will be implemented with pure NumPy from scratch.

---

## Q1 — Why LSTMs? The Vanishing Gradient Problem of Vanilla RNNs

In a standard Simple RNN (`07-rnn-from-scratch`), the recurrent hidden state is repeatedly updated across timesteps:

```math
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
```

While elegant, this formulation suffers from a severe mathematical vulnerability when learning dependencies over long sequences (`T > 10`).

### The Mathematical Proof of Vanishing Gradients

Consider the gradient of the loss at timestep `T` with respect to the hidden state at an early timestep `t`:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \frac{\partial h_j}{\partial h_{j-1}}
```

Expanding the single-step Jacobian matrix `\frac{\partial h_j}{\partial h_{j-1}}`:

```math
\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}\left(1 - h_j^2\right) W_{hh}^T
```

Substituting this into the product chain:

```math
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{j=t+1}^{T} \left( \operatorname{diag}\left(1 - h_j^2\right) W_{hh}^T \right)
```

Notice that this is a **repeated matrix product of length `(T - t)`**:

```math
\prod_{j=t+1}^{T} W_{hh}^T \approx \left( W_{hh}^T \right)^{T - t}
```

Let the eigendecomposition of `W_{hh}^T` be `Q \Lambda Q^{-1}`, where `\lambda_{\max}` is the largest eigenvalue:

```math
\left( W_{hh}^T \right)^{T - t} = Q \Lambda^{T - t} Q^{-1}
```

### The Two Catastrophic Regimes:

1. **Vanishing Gradient (`|\lambda_{\max}| < 1` and `|\tanh'| \le 1`)**:
   ```math
   \lim_{T - t \to \infty} \Lambda^{T - t} = \mathbf{0} \implies \frac{\partial L_T}{\partial h_t} \to \mathbf{0}
   ```
   Furthermore, the derivative of `\tanh(z)` is `1 - \tanh^2(z) \in [0, 1]`. For any non-zero activation, `|1 - h^2| < 1.0` (typically `0.3` to `0.7`). Multiplying 20 decimals together (`0.4^{20} \approx 1.1 \times 10^{-8}`) causes the error signal to exponentially decay to zero as it travels backward through time.
2. **Exploding Gradient (`|\lambda_{\max}| > 1`)**:
   ```math
   \lim_{T - t \to \infty} \Lambda^{T - t} = \infty \implies \left\| \frac{\partial L_T}{\partial h_t} \right\| \to \infty
   ```
   Weight updates overflow to `NaN`, completely destabilizing numerical training.

#### The Physical Consequence:
A simple RNN loses all memory of observations after 8 to 10 timesteps, making long-range sequence modeling mathematically impossible.

---

## Q2 — How LSTM Solves Vanishing Gradients: The Constant Error Carousel (CEC)

The LSTM architecture (Hochreiter & Schmidhuber, 1997) resolves this problem by separating memory into two distinct operational tracks:
1. **Hidden State (`h_t`)**: A non-linear **working memory** vector for immediate external interaction and classification.
2. **Cell State (`C_t`)**: An uninterrupted, linear **long-term memory highway**.

### The Fundamental Additive State Equation:

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

Where:
- `f_t \in (0, 1)^{1 \times H}` is the Forget Gate.
- `i_t \in (0, 1)^{1 \times H}` is the Input Gate.
- `\tilde{C}_t \in (-1, 1)^{1 \times H}` is the Candidate Cell State.
- `\odot` is the element-wise Hadamard product.

---

### The Mathematical Proof of Unbroken Gradient Flow

Let us compute the partial derivative of `C_t` with respect to `C_{t-1}`:

```math
\frac{\partial C_t}{\partial C_{t-1}} = \operatorname{diag}(f_t)
```

Look closely at this equation:
- **There is NO weight matrix multiplication `W_{hh}`!**
- **There is NO derivative of `\tanh`!**

Over `T - t` timesteps, the gradient propagating along the cell state highway is:

```math
\frac{\partial C_T}{\partial C_t} = \prod_{j=t+1}^{T} \frac{\partial C_j}{\partial C_{j-1}} = \prod_{j=t+1}^{T} \operatorname{diag}(f_j)
```

If the network learns to keep the forget gate open (`f_j \approx 1.0`):

```math
\frac{\partial C_T}{\partial C_t} \approx \prod_{j=t+1}^{T} \mathbf{I} = \mathbf{I} \quad (\text{Identity Matrix!})
```

The error gradient flows backward across 50, 100, or 500 timesteps **without any exponential attenuation**!
This linear additive channel is called the **Constant Error Carousel (CEC)**.

---

## Q3 — The 4 Internal Neural Networks (Gate Formulations & Roles)

An LSTM cell contains 4 distinct linear-projection neural networks regulating information flow:

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
                   │                            ▼          ▼                         ▼
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
                   │    ▼                                                                   │
  Inputs: [ x_t, h_{t-1} ] ─────────────────────────────────────────────────────────────────┴──►
```

### 1. The Forget Gate (`f_t`)
Decides what proportion of old memories in `C_{t-1}` should be erased or kept:

```math
f_t = \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f) \in (0, 1)^{1 \times H}
```

- `f_{t, k} \approx 0.0`: Completely erase memory dimension `k`.
- `f_{t, k} \approx 1.0`: Completely preserve memory dimension `k`.

### 2. The Input Gate (`i_t`)
Decides which memory dimensions will be written with new information:

```math
i_t = \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i) \in (0, 1)^{1 \times H}
```

### 3. The Candidate Cell State (`\tilde{C}_t`)
Generates new candidate information concepts bounded between -1 and +1:

```math
\tilde{C}_t = \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c) \in (-1, 1)^{1 \times H}
```

### 4. Cell State Update (`C_t`)
The core additive memory synthesis combining modulated past with modulated present:

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \in \mathbb{R}^{1 \times H}
```

### 5. The Output Gate (`o_t`)
Decides which parts of the long-term cell state `C_t` should be emitted into the external hidden state `h_t`:

```math
o_t = \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o) \in (0, 1)^{1 \times H}
```

### 6. Working Hidden State Emission (`h_t`)
Compresses `C_t` through `\tanh` and filters it with the output gate:

```math
h_t = o_t \odot \tanh(C_t) \in (-1, 1)^{1 \times H}
```

---

### Why Sigmoid for Gates and tanh for States?

- **Sigmoid `\sigma(z) \in (0, 1)`**: Acts as a **soft binary valve or switch**. Multiplying by `0` closes the valve (zero information pass); multiplying by `1` opens the valve completely.
- **tanh `\tanh(z) \in (-1, +1)`**: Acts as a **bounded, zero-centered representation**. It allows features to have positive and negative polarities without unbounded numerical explosion.

---

## Q4 — Fused Projection Vectorization: The High-Performance Architecture

In a naive implementation, evaluating an LSTM cell requires 4 separate matrix multiplications for the input `x_t` and 4 separate matrix multiplications for the recurrent state `h_{t-1}` (8 total dot products per timestep):

```python
# Naive execution: 8 separate matrix multiplications (SLOW)
f = sigmoid(x @ W_xf + h @ W_hf + b_f)
i = sigmoid(x @ W_xi + h @ W_hi + b_i)
c = np.tanh(x @ W_xc + h @ W_hc + b_c)
o = sigmoid(x @ W_xo + h @ W_ho + b_o)
```

In high-performance deep learning libraries (and our from-scratch implementation), we **fuse all 4 gate weights into unified matrices**:

```math
W_x = \begin{bmatrix} W_{xf} & W_{xi} & W_{xc} & W_{xo} \end{bmatrix} \in \mathbb{R}^{D \times 4H}
```

```math
W_h = \begin{bmatrix} W_{hf} & W_{hi} & W_{hc} & W_{ho} \end{bmatrix} \in \mathbb{R}^{H \times 4H}
```

```math
b = \begin{bmatrix} b_f & b_i & b_c & b_o \end{bmatrix} \in \mathbb{R}^{1 \times 4H}
```

### The Vectorized Fused Forward Pass:
Now, all 4 gates are evaluated with **just two matrix multiplications**:

```math
Z = x_t W_x + h_{t-1} W_h + b \in \mathbb{R}^{B \times 4H}
```

We then slice `Z` into 4 contiguous blocks of size `H`:

```python
# Fused vectorized projection (FAST)
Z = x @ W_x + h_prev @ W_h + b  # Shape: (B, 4*H)
z_f, z_i, z_c, z_o = np.split(Z, 4, axis=-1)

f = sigmoid(z_f)
i = sigmoid(z_i)
c_bar = np.tanh(z_c)
o = sigmoid(z_o)

C = f * C_prev + i * c_bar
h = o * np.tanh(C)
```

---

## Q5 — Parameters, Tensor Shapes, and Parameter Budget

To verify every calculation by hand down to 6 decimal places, we configure an exact educational setup:

### Problem Dimensions:
- Batch size: `B = 1`
- Sequence length: `T = 3` timesteps
- Input feature dimension: `D = 2`
- Hidden state capacity: `H = 3` memory units
- Fused gate dimension: `4H = 12`
- Output classification classes: `K = 2`

### Tensor Shapes Table:

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `x_t` | `(1, 2)` | `x_t \in \mathbb{R}^{1 \times D}` | Input vector at timestep `t` |
| `h_{t-1}` | `(1, 3)` | `h_{t-1} \in \mathbb{R}^{1 \times H}` | Prior working hidden state |
| `C_{t-1}` | `(1, 3)` | `C_{t-1} \in \mathbb{R}^{1 \times H}` | Prior long-term cell state |
| `W_x` | `(2, 12)` | `W_x \in \mathbb{R}^{D \times 4H}` | Fused input-to-gate weights |
| `W_h` | `(3, 12)` | `W_h \in \mathbb{R}^{H \times 4H}` | Fused hidden-to-gate recurrent weights |
| `b` | `(1, 12)` | `b \in \mathbb{R}^{1 \times 4H}` | Fused gate bias offset vector |
| `Z_t` | `(1, 12)` | `Z_t \in \mathbb{R}^{1 \times 4H}` | Fused linear pre-activations |
| `f_t` | `(1, 3)` | `f_t \in (0, 1)^{1 \times H}` | Forget gate activations |
| `i_t` | `(1, 3)` | `i_t \in (0, 1)^{1 \times H}` | Input gate activations |
| `\tilde{C}_t` | `(1, 3)` | `\tilde{C}_t \in (-1, 1)^{1 \times H}` | Candidate cell state |
| `C_t` | `(1, 3)` | `C_t \in \mathbb{R}^{1 \times H}` | Updated long-term cell state |
| `o_t` | `(1, 3)` | `o_t \in (0, 1)^{1 \times H}` | Output gate activations |
| `h_t` | `(1, 3)` | `h_t \in (-1, 1)^{1 \times H}` | Updated working hidden state |
| `W_{hy}` | `(3, 2)` | `W_{hy} \in \mathbb{R}^{H \times K}` | Output classification projection weights |
| `b_y` | `(1, 2)` | `b_y \in \mathbb{R}^{1 \times K}` | Output classification bias |
| `o_{\mathrm{out}, t}` | `(1, 2)` | `o_t \in \mathbb{R}^{1 \times K}` | Raw classification logits |
| `\hat{y}_t` | `(1, 2)` | `\hat{y}_t \in (0, 1)^{1 \times K}` | Softmax class probabilities |

### Parameter Budget:

```math
N_{\mathrm{params}} = \underbrace{(2 \cdot 12)}_{W_x} + \underbrace{(3 \cdot 12)}_{W_h} + \underbrace{12}_{b} + \underbrace{(3 \cdot 2)}_{W_{hy}} + \underbrace{2}_{b_y} = 24 + 36 + 12 + 6 + 2 = 80 \text{ parameters}
```

---

## Q6 — Concrete Numerical Setup (Matching verify_lstm.py)

These exact values match the automated verification script [`verify_lstm.py`](file:///Users/pasan/.gemini/antigravity-ide/brain/d3b699e8-4410-4b8c-b2cb-728045b5ca32/scratch/verify_lstm.py).

### 1. Input Sequence `X` (`T = 3`):
```math
x_1 = \begin{bmatrix} 0.5 & -0.3 \end{bmatrix}, \quad
x_2 = \begin{bmatrix} -0.2 & 0.4 \end{bmatrix}, \quad
x_3 = \begin{bmatrix} 0.3 & 0.1 \end{bmatrix}
```

### 2. One-Hot Targets `Y`:
```math
Y_1 = \begin{bmatrix} 1.0 & 0.0 \end{bmatrix}, \quad
Y_2 = \begin{bmatrix} 0.0 & 1.0 \end{bmatrix}, \quad
Y_3 = \begin{bmatrix} 1.0 & 0.0 \end{bmatrix}
```

### 3. Initial States:
```math
h_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}, \quad
C_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 \end{bmatrix}
```

### 4. Parameter Matrices:

```math
W_x = \begin{bmatrix}
 0.3 & -0.2 &  0.1 &   0.2 &  0.4 & -0.1 &   0.5 & -0.3 &  0.2 &   0.1 & -0.2 &  0.3 \\
-0.1 &  0.3 & -0.2 &  -0.3 &  0.1 &  0.2 &  -0.2 &  0.4 & -0.1 &   0.2 &  0.1 & -0.4
\end{bmatrix} \in \mathbb{R}^{2 \times 12}
```

```math
W_h = \begin{bmatrix}
 0.1 &  0.2 & -0.1 &  -0.2 &  0.1 &  0.3 &   0.2 & -0.1 &  0.1 &  -0.1 &  0.2 &  0.1 \\
-0.2 &  0.1 &  0.3 &   0.1 & -0.2 &  0.1 &  -0.1 &  0.3 & -0.2 &   0.3 & -0.1 &  0.2 \\
 0.3 & -0.1 &  0.2 &   0.2 &  0.1 & -0.1 &   0.1 & -0.2 &  0.3 &  -0.2 &  0.1 & -0.1
\end{bmatrix} \in \mathbb{R}^{3 \times 12}
```

```math
b = \begin{bmatrix}
 0.5 &  0.5 &  0.5 &   0.0 & -0.1 &  0.1 &   0.1 &  0.0 & -0.1 &   0.0 &  0.1 & -0.1
\end{bmatrix} \in \mathbb{R}^{1 \times 12}
```

```math
W_{hy} = \begin{bmatrix}
 0.4 & -0.3 \\
-0.2 &  0.5 \\
 0.3 & -0.1
\end{bmatrix} \in \mathbb{R}^{3 \times 2}, \quad
b_y = \begin{bmatrix} 0.1 & -0.1 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
```

---

## Q7 — Forward Pass: Step-by-Step Numerical Walkthrough

We now trace every mathematical operation across all 3 timesteps.

### Timestep 1 (t = 1)

#### 1. Fused Pre-activation `Z_1`:
Because `h_0 = \mathbf{0}`, `h_0 W_h = \mathbf{0}`:
```text
x_1 @ W_x = [0.18, -0.19, 0.11,  0.19,  0.17, -0.11,  0.31, -0.27, 0.13, -0.01, -0.13, 0.27]
b         = [0.50,  0.50, 0.50,  0.00, -0.10,  0.10,  0.10,  0.00,-0.10,  0.00,  0.10,-0.10]
```
```math
Z_1 = x_1 W_x + b = \begin{bmatrix} 0.68 & 0.31 & 0.61 & 0.19 & 0.07 & -0.01 & 0.41 & -0.27 & 0.03 & -0.01 & -0.03 & 0.17 \end{bmatrix}
```

#### 2. Slicing Gates and Applying Activations:
- **Forget gate `z_f = [0.68, 0.31, 0.61]`**:
  ```math
  f_1 = \sigma(z_f) = \begin{bmatrix} \sigma(0.68) & \sigma(0.31) & \sigma(0.61) \end{bmatrix} = \begin{bmatrix} 0.663739 & 0.576885 & 0.647941 \end{bmatrix}
  ```
- **Input gate `z_i = [0.19, 0.07, -0.01]`**:
  ```math
  i_1 = \sigma(z_i) = \begin{bmatrix} \sigma(0.19) & \sigma(0.07) & \sigma(-0.01) \end{bmatrix} = \begin{bmatrix} 0.547358 & 0.517493 & 0.497500 \end{bmatrix}
  ```
- **Candidate state `z_c = [0.41, -0.27, 0.03]`**:
  ```math
  \tilde{C}_1 = \tanh(z_c) = \begin{bmatrix} \tanh(0.41) & \tanh(-0.27) & \tanh(0.03) \end{bmatrix} = \begin{bmatrix} 0.388473 & -0.263625 & 0.029991 \end{bmatrix}
  ```
- **Output gate `z_o = [-0.01, -0.03, 0.17]`**:
  ```math
  o_1 = \sigma(z_o) = \begin{bmatrix} \sigma(-0.01) & \sigma(-0.03) & \sigma(0.17) \end{bmatrix} = \begin{bmatrix} 0.497500 & 0.492501 & 0.542398 \end{bmatrix}
  ```

#### 3. Cell State `C_1`:
Because `C_0 = \mathbf{0}`:
```math
C_1 = f_1 \odot C_0 + i_1 \odot \tilde{C}_1 = \mathbf{0} + [0.547358(0.388473), \; 0.517493(-0.263625), \; 0.497500(0.029991)]
    = \begin{bmatrix} 0.212633 & -0.136424 & 0.014921 \end{bmatrix}
```

#### 4. Working Hidden State `h_1`:
```math
\tanh(C_1) = \begin{bmatrix} \tanh(0.212633) & \tanh(-0.136424) & \tanh(0.014921) \end{bmatrix} = \begin{bmatrix} 0.209486 & -0.135584 & 0.014919 \end{bmatrix}
```
```math
h_1 = o_1 \odot \tanh(C_1) = [0.497500(0.209486), \; 0.492501(-0.135584), \; 0.542398(0.014919)]
    = \begin{bmatrix} 0.104219 & -0.066775 & 0.008092 \end{bmatrix}
```

#### 5. Output Projection, Softmax, & Loss:
```math
o_{\mathrm{out}, 1} = h_1 W_{hy} + b_y = \begin{bmatrix} 0.104219 & -0.066775 & 0.008092 \end{bmatrix} \begin{bmatrix} 0.4 & -0.3 \\ -0.2 & 0.5 \\ 0.3 & -0.1 \end{bmatrix} + \begin{bmatrix} 0.1 & -0.1 \end{bmatrix}
                   = \begin{bmatrix} 0.157470 & -0.165463 \end{bmatrix}
```
```math
\hat{y}_1 = \mathrm{softmax}(o_{\mathrm{out}, 1}) = \begin{bmatrix} 0.580039 & 0.419961 \end{bmatrix}
```
Target is class 0 (`Y_1 = [1.0, 0.0]`):
```math
L_1 = -\ln(0.580039) = 0.544660
```

---

### Timestep 2 (t = 2)

#### 1. Fused Pre-activation `Z_2`:
Input `x_2 = [-0.2, 0.4]`, prior state `h_1 = [0.104219, -0.066775, 0.008092]`:
```text
Z_2 = x_2 @ W_x + h_1 @ W_h + b
    = [0.426205, 0.673357, 0.371164, -0.185903, -0.115414, 0.223779, -0.051669, 0.187927, -0.153795, 0.027927, 0.208331, -0.323742]
```

#### 2. Slicing Gates and Activations:
- `f_2 = \sigma([0.426205, 0.673357, 0.371164]) = [0.604967, 0.662255, 0.591740]`
- `i_2 = \sigma([-0.185903, -0.115414, 0.223779]) = [0.453658, 0.471179, 0.555712]`
- `\tilde{C}_2 = \tanh([-0.051669, 0.187927, -0.153795]) = [-0.051623, 0.185746, -0.152594]`
- `o_2 = \sigma([0.027927, 0.208331, -0.323742]) = [0.506981, 0.551895, 0.419764]`

#### 3. Cell State `C_2`:
```math
C_2 = f_2 \odot C_1 + i_2 \odot \tilde{C}_2
    = [0.604967(0.212633) + 0.453658(-0.051623), \;
       0.662255(-0.136424) + 0.471179(0.185746), \;
       0.591740(0.014921) + 0.555712(-0.152594)]
    = \begin{bmatrix} 0.105217 & -0.002828 & -0.075969 \end{bmatrix}
```

#### 4. Working Hidden State `h_2`:
```math
\tanh(C_2) = \begin{bmatrix} 0.104830 & -0.002828 & -0.075824 \end{bmatrix}
```
```math
h_2 = o_2 \odot \tanh(C_2) = \begin{bmatrix} 0.053147 & -0.001561 & -0.031828 \end{bmatrix}
```

#### 5. Output Projection, Softmax, & Loss:
```math
o_{\mathrm{out}, 2} = h_2 W_{hy} + b_y = \begin{bmatrix} 0.112023 & -0.113542 \end{bmatrix}
```
```math
\hat{y}_2 = \mathrm{softmax}(o_{\mathrm{out}, 2}) = \begin{bmatrix} 0.556153 & 0.443847 \end{bmatrix}
```
Target is class 1 (`Y_2 = [0.0, 1.0]`):
```math
L_2 = -\ln(0.443847) = 0.812276
```

---

### Timestep 3 (t = 3)

#### 1. Fused Pre-activation `Z_3`:
Input `x_3 = [0.3, 0.1]`, prior state `h_2 = [0.053147, -0.001561, -0.031828]`:
```text
Z_3 = x_3 @ W_x + h_2 @ W_h + b
    = [0.576078, 0.483656, 0.497851, 0.012849, 0.032444, 0.108971, 0.237603, -0.049417, -0.053922, 0.050583, 0.057603, -0.041815]
```

#### 2. Slicing Gates and Activations:
- `f_3 = \sigma([0.576078, 0.483656, 0.497851]) = [0.640165, 0.618611, 0.621954]`
- `i_3 = \sigma([0.012849, 0.032444, 0.108971]) = [0.503212, 0.508110, 0.527216]`
- `\tilde{C}_3 = \tanh([0.237603, -0.049417, -0.053922]) = [0.233230, -0.049377, -0.053869]`
- `o_3 = \sigma([0.050583, 0.057603, -0.041815]) = [0.512643, 0.514397, 0.489548]`

#### 3. Cell State `C_3`:
```math
C_3 = f_3 \odot C_2 + i_3 \odot \tilde{C}_3
    = [0.640165(0.105217) + 0.503212(0.233230), \;
       0.618611(-0.002828) + 0.508110(-0.049377), \;
       0.621954(-0.075969) + 0.527216(-0.053869)]
    = \begin{bmatrix} 0.184720 & -0.026839 & -0.075650 \end{bmatrix}
```

#### 4. Working Hidden State `h_3`:
```math
\tanh(C_3) = \begin{bmatrix} 0.182648 & -0.026832 & -0.075506 \end{bmatrix}
```
```math
h_3 = o_3 \odot \tanh(C_3) = \begin{bmatrix} 0.093633 & -0.013802 & -0.036964 \end{bmatrix}
```

#### 5. Output Projection, Softmax, & Loss:
```math
o_{\mathrm{out}, 3} = h_3 W_{hy} + b_y = \begin{bmatrix} 0.129125 & -0.131295 \end{bmatrix}
```
```math
\hat{y}_3 = \mathrm{softmax}(o_{\mathrm{out}, 3}) = \begin{bmatrix} 0.564739 & 0.435261 \end{bmatrix}
```
Target is class 0 (`Y_3 = [1.0, 0.0]`):
```math
L_3 = -\ln(0.564739) = 0.571391
```

---

## Q8 — Total Sequence Loss

In a synchronous sequence model, the total sequence loss is the sum across all `T = 3` timesteps:

```math
L_{\mathrm{seq}} = L_1 + L_2 + L_3 = 0.544660 + 0.812276 + 0.571391 = 1.928327
```

Average loss per step:

```math
\bar{L} = \frac{1.928327}{3} = 0.642776
```

---

## Q9 — Backpropagation Through Time (BPTT) for LSTM: First-Principles Matrix Calculus

Because an LSTM has two state paths (`h_t` and `C_t`), backpropagation requires deriving the coupled multivariable chain rule.

### 1. Output Logit Error (`∂L_t / ∂o_{\mathrm{out}, t}`)
```math
\delta_{o_{\mathrm{out}, t}} = \hat{y}_t - Y_t \in \mathbb{R}^{1 \times K}
```
```math
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} h_t^T \delta_{o_{\mathrm{out}, t}}, \quad \frac{\partial L}{\partial b_y} = \sum_{t=1}^{T} \delta_{o_{\mathrm{out}, t}}
```

---

### 2. Total Gradient at Hidden State `h_t`
At any timestep `t`, `h_t` receives error from two sources:
1. Direct output classification: `\delta_{o_{\mathrm{out}, t}} W_{hy}^T`
2. Future recurrent step `t+1`: `\delta_{h_{t+1, \mathrm{in}}}`

```math
\delta_{h_t} = \frac{\partial L}{\partial h_t} = \delta_{o_{\mathrm{out}, t}} W_{hy}^T + \delta_{h_{t+1, \mathrm{in}}} \in \mathbb{R}^{1 \times H}
```
*(At the final step `t = T = 3`, `\delta_{h_{T+1, \mathrm{in}}} = \mathbf{0}`).*

---

### 3. Total Gradient at Cell State `C_t` (The Dual Pathway)
`C_t` influences the loss along **two distinct pathways**:
- **Path A (Immediate Output via `h_t`)**: `C_t \to \tanh(C_t) \to h_t \to L_t`
- **Path B (Future Cell State via `C_{t+1}`)**: `C_t \to f_{t+1} \odot C_t \to C_{t+1} \to L_{>t}`

By the chain rule:

```math
\delta_{C_t} = \frac{\partial L}{\partial C_t} = \underbrace{\delta_{h_t} \odot o_t \odot \left(1 - \tanh^2(C_t)\right)}_{\text{Path A: From current hidden state}} + \underbrace{\delta_{C_{t+1}} \odot f_{t+1}}_{\text{Path B: Unbroken highway from future cell state}}
```

> [!IMPORTANT]
> The second term `\delta_{C_{t+1}} \odot f_{t+1}` is the **Constant Error Carousel in reverse**!
> Notice there is no matrix multiplication or saturating derivative. Error propagates backward through time purely scaled by the forget gate `f_{t+1}`!

---

### 4. Gate Pre-activation Gradients (`\delta_{z_o}, \delta_{z_c}, \delta_{z_i}, \delta_{z_f}`)

Using the derivatives of Sigmoid (`\sigma' = \sigma(1 - \sigma)`) and tanh (`\tanh' = 1 - \tanh^2`):

1. **Output Gate `\delta_{z_o}`**:
   ```math
   \delta_{z_o} = \frac{\partial L}{\partial z_o} = \delta_{h_t} \odot \tanh(C_t) \odot o_t \odot (1 - o_t) \in \mathbb{R}^{1 \times H}
   ```
2. **Candidate State `\delta_{z_c}`**:
   ```math
   \delta_{z_c} = \frac{\partial L}{\partial z_c} = \delta_{C_t} \odot i_t \odot \left(1 - \tilde{C}_t^2\right) \in \mathbb{R}^{1 \times H}
   ```
3. **Input Gate `\delta_{z_i}`**:
   ```math
   \delta_{z_i} = \frac{\partial L}{\partial z_i} = \delta_{C_t} \odot \tilde{C}_t \odot i_t \odot (1 - i_t) \in \mathbb{R}^{1 \times H}
   ```
4. **Forget Gate `\delta_{z_f}`**:
   ```math
   \delta_{z_f} = \frac{\partial L}{\partial z_f} = \delta_{C_t} \odot C_{t-1} \odot f_t \odot (1 - f_t) \in \mathbb{R}^{1 \times H}
   ```

---

### 5. Fused Pre-activation Gradient `\delta_{Z_t}`
Concatenate the 4 gate gradients into a single `(1, 12)` row vector:

```math
\delta_{Z_t} = \begin{bmatrix} \delta_{z_f} & \delta_{z_i} & \delta_{z_c} & \delta_{z_o} \end{bmatrix} \in \mathbb{R}^{1 \times 4H}
```

---

### 6. Parameter Gradient Accumulation
```math
\frac{\partial L}{\partial W_x} = \sum_{t=1}^{T} x_t^T \delta_{Z_t} \in \mathbb{R}^{D \times 4H}
```
```math
\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} h_{t-1}^T \delta_{Z_t} \in \mathbb{R}^{H \times 4H}
```
```math
\frac{\partial L}{\partial b} = \sum_{t=1}^{T} \delta_{Z_t} \in \mathbb{R}^{1 \times 4H}
```

---

### 7. Gradients Flowing to Previous Timestep (`t-1`)
- **To previous hidden state**:
  ```math
  \delta_{h_{t-1, \mathrm{in}}} = \delta_{Z_t} W_h^T \in \mathbb{R}^{1 \times H}
  ```
- **To previous cell state**:
  ```math
  \delta_{C_{t-1, \mathrm{highway}}} = \delta_{C_t} \odot f_t \in \mathbb{R}^{1 \times H}
  ```

---

## Q10 — Step-by-Step Numerical BPTT Walkthrough

We now trace the backward equations from `t = 3` down to `t = 1`.

---

### Backward Step t = 3 (Final Timestep)

#### 1. Output Logit Error:
```text
d_logit_3 = prob_3 - Y_3 = [0.564739, 0.435261] - [1.0, 0.0] = [-0.435261, 0.435261]
```

#### 2. Hidden State Error `\delta_{h_3}`:
Because `t = 3` is the final step, `\delta_{h_{4, \mathrm{in}}} = \mathbf{0}`:
```math
\delta_{h_3} = d\_logit_3 W_{hy}^T = \begin{bmatrix} -0.435261 & 0.435261 \end{bmatrix} \begin{bmatrix} 0.4 & -0.2 & 0.3 \\ -0.3 & 0.5 & -0.1 \end{bmatrix}
             = [-0.304682, 0.304682, -0.174104]
```

#### 3. Output Gate Error `\delta_{z_o}`:
```text
d_zo = d_h_3 * tanh(C_3) * o_3 * (1 - o_3)
     = [-0.304682, 0.304682, -0.174104] * [0.182648, -0.026832, -0.075506] * [0.512643, 0.514397, 0.489548] * [0.487357, 0.485603, 0.510452]
     = [-0.013903, -0.002042, 0.003285]
```

#### 4. Cell State Error `\delta_{C_3}`:
Because `t = 3` is the final step, `\delta_{C_4} = \mathbf{0}`:
```math
\delta_{C_3} = \delta_{h_3} \odot o_3 \odot (1 - \tanh^2(C_3))
             = [-0.150983, 0.156615, -0.084746]
```

#### 5. Candidate, Input, and Forget Gate Errors:
- `\delta_{z_c} = \delta_{C_3} \odot i_3 \odot (1 - \tilde{C}_3^2) = [-0.071844, 0.079384, -0.044550]`
- `\delta_{z_i} = \delta_{C_3} \odot \tilde{C}_3 \odot i_3 \odot (1 - i_3) = [-0.008803, -0.001933, 0.001138]`
- `\delta_{z_f} = \delta_{C_3} \odot C_2 \odot f_3 \odot (1 - f_3) = [-0.003659, -0.000104, 0.001514]`

#### 6. Fused `\delta_{Z_3}`:
```text
d_Z_3 = [-0.003659, -0.000104, 0.001514, -0.008803, -0.001933, 0.001138, -0.071844, 0.079384, -0.044550, -0.013903, -0.002042, 0.003285]
```

#### 7. Propagating to Prior Step (`t = 2`):
- To hidden state: `\delta_{h_2, \mathrm{in}} = \delta_{Z_3} W_h^T = [-0.024081, 0.037395, -0.037030]`
- To cell state highway: `\delta_{C_2, \mathrm{highway}} = \delta_{C_3} \odot f_3 = [-0.096654, 0.096884, -0.052708]`

---

### Backward Step t = 2

#### 1. Output Logit Error:
```text
d_logit_2 = prob_2 - Y_2 = [0.556153, 0.443847] - [0.0, 1.0] = [0.556153, -0.556153]
```

#### 2. Total Hidden State Error `\delta_{h_2}`:
```math
\delta_{h_2} = d\_logit_2 W_{hy}^T + \delta_{h_2, \mathrm{in}}
             = [0.389307, -0.389307, 0.222461] + [-0.024081, 0.037395, -0.037030]
             = [0.365226, -0.351912, 0.185431]
```

#### 3. Output Gate Error `\delta_{z_o}`:
```text
d_zo = [0.009570, 0.000246, -0.003424]
```

#### 4. Total Cell State Error `\delta_{C_2}` (Dual Path!):
```math
\delta_{C_2} = \underbrace{\delta_{h_2} \odot o_2 \odot (1 - \tanh^2(C_2))}_{\text{From current hidden state: } [0.183128, -0.194217, 0.077389]} + \underbrace{\delta_{C_2, \mathrm{highway}}}_{\text{From future step: } [-0.096654, 0.096884, -0.052708]}
             = [0.086474, -0.097333, 0.024681]
```

#### 5. Candidate, Input, and Forget Gate Errors:
- `\delta_{z_c} = [0.039125, -0.044279, 0.013396]`
- `\delta_{z_i} = [-0.001106, -0.004505, -0.000930]`
- `\delta_{z_f} = [0.004394, 0.002970, 0.000089]`

#### 6. Fused `\delta_{Z_2}`:
```text
d_Z_2 = [0.004394, 0.002970, 0.000089, -0.001106, -0.004505, -0.000930, 0.039125, -0.044279, 0.013396, 0.009570, 0.000246, -0.003424]
```

#### 7. Propagating to Prior Step (`t = 1`):
- To hidden state: `\delta_{h_1, \mathrm{in}} = \delta_{Z_2} W_h^T = [0.012859, -0.017572, 0.015701]`
- To cell state highway: `\delta_{C_1, \mathrm{highway}} = \delta_{C_2} \odot f_2 = [0.052314, -0.064459, 0.014605]`

---

### Backward Step t = 1

#### 1. Output Logit Error:
```text
d_logit_1 = prob_1 - Y_1 = [0.580039, 0.419961] - [1.0, 0.0] = [-0.419961, 0.419961]
```

#### 2. Total Hidden State Error `\delta_{h_1}`:
```math
\delta_{h_1} = d\_logit_1 W_{hy}^T + \delta_{h_1, \mathrm{in}}
             = [-0.293973, 0.293973, -0.167984] + [0.012859, -0.017572, 0.015701]
             = [-0.281114, 0.276401, -0.152284]
```

#### 3. Output Gate Error `\delta_{z_o}`:
```text
d_zo = [-0.014722, -0.009367, -0.000564]
```

#### 4. Total Cell State Error `\delta_{C_1}`:
```math
\delta_{C_1} = \delta_{h_1} \odot o_1 \odot (1 - \tanh^2(C_1)) + \delta_{C_1, \mathrm{highway}}
             = [-0.133718, 0.133625, -0.082580] + [0.052314, -0.064459, 0.014605]
             = [-0.081403, 0.069166, -0.067975]
```

#### 5. Candidate, Input, and Forget Gate Errors:
- `\delta_{z_c} = [-0.037832, 0.033305, -0.033787]`
- `\delta_{z_i} = [-0.007835, -0.004553, -0.000510]`
- `\delta_{z_f} = \mathbf{0}` (because `C_0 = \mathbf{0}`)

#### 6. Fused `\delta_{Z_1}`:
```text
d_Z_1 = [0.0, 0.0, 0.0, -0.007835, -0.004553, -0.000510, -0.037832, 0.033305, -0.033787, -0.014722, -0.009367, -0.000564]
```

---

### Final Accumulated Gradients Across All Timesteps

#### 1. Output Classification Weights (`dW_hy`, `db_y`):
```math
\frac{\partial L}{\partial W_{hy}} = \begin{bmatrix} -0.054965 & 0.054965 \\ 0.033183 & -0.033183 \\ -0.005011 & 0.005011 \end{bmatrix}, \quad
\frac{\partial L}{\partial b_y} = \begin{bmatrix} -0.299069 & 0.299069 \end{bmatrix}
```

#### 2. Fused Input Weights `dW_x` (`(2, 12)`):
```math
\frac{\partial L}{\partial W_x} = \sum_{t=1}^3 x_t^T \delta_{Z_t} = \begin{bmatrix}
-0.001977 & -0.000625 &  0.000436 & -0.006337 & -0.001955 &  0.000273 & -0.048294 &  0.049324 & -0.032938 & -0.013446 & -0.005345 &  0.001388 \\
 0.001392 &  0.001178 &  0.000187 &  0.001028 & -0.000629 & -0.000105 &  0.019815 & -0.019765 &  0.011040 &  0.006854 &  0.002704 & -0.000872
\end{bmatrix}
```

#### 3. Fused Recurrent Weights `dW_h` (`(3, 12)`):
```math
\frac{\partial L}{\partial W_h} = \sum_{t=1}^3 h_{t-1}^T \delta_{Z_t} = \begin{bmatrix}
 0.000263 &  0.000304 &  0.000090 & -0.000583 & -0.000572 & -0.000036 &  0.000259 & -0.000396 & -0.000972 &  0.000258 & -0.000083 & -0.000182 \\
-0.000288 & -0.000198 & -0.000008 &  0.000088 &  0.000304 &  0.000060 & -0.002500 &  0.002833 & -0.000825 & -0.000617 & -0.000013 &  0.000224 \\
 0.000152 &  0.000027 & -0.000047 &  0.000271 &  0.000025 & -0.000044 &  0.002603 & -0.002885 &  0.001526 &  0.000520 &  0.000067 & -0.000132
\end{bmatrix}
```

#### 4. Fused Bias `db` (`(1, 12)`):
```math
\frac{\partial L}{\partial b} = \sum_{t=1}^3 \delta_{Z_t} = \begin{bmatrix}
 0.000735 &  0.002866 &  0.001603 & -0.017744 & -0.010990 & -0.000302 & -0.070551 &  0.068410 & -0.064941 & -0.019056 & -0.011163 & -0.000703
\end{bmatrix}
```

---

## Q11 — The Forget Gate Bias Initialization Trick (b_f = 1.0)

A critical practical insight introduced by Gers et al. (2000) and Jozefowicz et al. (2015) is the **forget gate bias initialization trick**.

### Why Zero Bias Initialization Fails:
If biases are initialized to zero (`b_f = 0.0`):
```math
f_t = \sigma(0.0) = 0.5
```
At the start of training, the model erases 50% of its memory at every single timestep! Over a sequence of 10 steps:
```math
0.5^{10} \approx 0.000976
```
The model inadvertently behaves like a vanishing-gradient RNN before it has had a chance to learn!

### The Solution: Positive Bias Initialization
Initialize the forget gate bias to `1.0` or `2.0`:
```math
f_t = \sigma(1.0) \approx 0.731, \quad \sigma(2.0) \approx 0.881
```
By default, the cell state highway remains wide open at the start of training, enabling error gradients to flow freely across the entire sequence. The network learns when to forget, rather than struggling to remember.

---

## Q12 — Complete Mathematical Mapping to lstm.py Implementation

This cross-reference connects every mathematical equation derived above directly to its implementation in [`src/lstm.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/09-lstm/src/lstm.py):

| Component / Function | Mathematical Equation | NumPy Variables & Dimensions |
| :--- | :--- | :--- |
| `lstm_cell_forward(x_t, h_prev, C_prev)` | Fused projection `Z = x_t W_x + h_{t-1} W_h + b` | `x_t: (B, D)`, `h_prev: (B, H)` -> `Z: (B, 4H)` |
| `np.split(Z, 4, axis=-1)` | Slices `z_f, z_i, z_c, z_o` | Splits `(B, 4H)` into four `(B, H)` arrays |
| Gate activations | `f = \sigma(z_f)`, `i = \sigma(z_i)`, `\tilde{C} = \tanh(z_c)`, `o = \sigma(z_o)` | All four outputs in shape `(B, H)` |
| State updates | `C_t = f \odot C_{t-1} + i \odot \tilde{C}_t`, `h_t = o \odot \tanh(C_t)` | `C_t: (B, H)`, `h_t: (B, H)` |
| `lstm_forward(X, h_0, C_0)` | Unrolls cell over `T` timesteps; caches all gates | `X: (B, T, D)` -> `H: (B, T, H), C: (B, T, H)` |
| `dense_output(h_t, W_hy, b_y)` | `o_{\mathrm{out}, t} = h_t W_{hy} + b_y` | `h_t: (B, H)` -> `logits: (B, K)` |
| `lstm_backward(d_logits, cache)` | Dual path BPTT: computes `\delta_{h_t}`, `\delta_{C_t}`, and `\delta_{Z_t}` | Upstream loss -> parameter gradients |
| Parameter updates | `dW_x += x_t^T \delta_{Z_t}`, `dW_h += h_{t-1}^T \delta_{Z_t}`, `db += \delta_{Z_t}` | `dW_x: (D, 4H)`, `dW_h: (H, 4H)`, `db: (1, 4H)` |

---

## Summary of the Complete Mathematical Pipeline

```text
========================================================================================================================
FORWARD LSTM CELL PIPELINE (One Timestep)
========================================================================================================================
1. Fused Gate Pre-activations:         Z = x_t @ W_x + h_{t-1} @ W_h + b                Z in R^(B x 4H)
2. Slicing Gates:                      z_f, z_i, z_c, z_o = split(Z, 4)                 Each in R^(B x H)
3. Gate Activations:                   f = sigmoid(z_f)                                 f in (0, 1)
                                       i = sigmoid(z_i)                                 i in (0, 1)
                                       c_bar = tanh(z_c)                                c_bar in (-1, 1)
                                       o = sigmoid(z_o)                                 o in (0, 1)
4. Additive Memory Fusion:             C_t = f * C_{t-1} + i * c_bar                    C_t in R^(B x H)
5. Working Hidden State Emission:      h_t = o * tanh(C_t)                              h_t in (-1, 1)
6. Output Projection & Softmax:        o_out = h_t @ W_hy + b_y                         o_out in R^(B x K)
                                       y_hat = softmax(o_out)                           y_hat in (0, 1)
========================================================================================================================
BACKWARD LSTM BPTT PIPELINE (Reverse Time)
========================================================================================================================
1. Output Logit Error:                 d_logit = y_hat - Y_t                            d_logit in R^(B x K)
2. Output Weight Gradients:            dW_hy += h_t.T @ d_logit                         dW_hy in R^(H x K)
                                       db_y  += sum(d_logit)                            db_y in R^(1 x K)
3. Total Hidden Error:                 d_h = d_logit @ W_hy.T + d_h_next                d_h in R^(B x H)
4. Output Gate Error:                  d_zo = d_h * tanh(C_t) * o * (1 - o)             d_zo in R^(B x H)
5. Total Cell State Error:             d_C = d_h * o * (1 - tanh^2(C_t)) + d_C_next     d_C in R^(B x H)
   (Dual Path Highway!)
6. Candidate State Error:              d_zc = d_C * i * (1 - c_bar^2)                   d_zc in R^(B x H)
7. Input Gate Error:                   d_zi = d_C * c_bar * i * (1 - i)                 d_zi in R^(B x H)
8. Forget Gate Error:                  d_zf = d_C * C_{t-1} * f * (1 - f)               d_zf in R^(B x H)
9. Fused Pre-activation Error:         d_Z = concatenate([d_zf, d_zi, d_zc, d_zo])      d_Z in R^(B x 4H)
10. Shared Weight Accumulation:        dW_x += x_t.T @ d_Z                              dW_x in R^(D x 4H)
                                       dW_h += h_{t-1}.T @ d_Z                          dW_h in R^(H x 4H)
                                       db   += sum(d_Z)                                 db in R^(1 x 4H)
11. Propagate to Prior Step (t-1):     d_h_next = d_Z @ W_h.T                           d_h_next in R^(B x H)
                                       d_C_next = d_C * f                               d_C_next in R^(B x H) (Highway!)
========================================================================================================================
```

This completes the foundational mathematics for Project 09. The next step is validating
each of these formulas in executable NumPy code in [`src/lstm.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/09-lstm/src/lstm.py).
