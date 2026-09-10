# 10 — GRU From Scratch

## Goal

Build and understand a **Gated Recurrent Unit (GRU)** network from first principles with pure NumPy. Introduced by Kyunghyun Cho et al. in 2014, the GRU streamlines the gating mechanics of the Long Short-Term Memory network (LSTM, built in `09-lstm`) while mitigating the catastrophic vanishing gradient problem of the simple Recurrent Neural Network (`07-rnn-from-scratch`).

The network implements the canonical recurrent sequence forward pipeline:

```text
r_t       = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)
z_t       = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)
h_reset   = r_t * h_{t-1}
h_cand    = tanh(x_t @ W_xh + h_reset @ W_hh + b_h)
h_t       = (1 - z_t) * h_{t-1} + z_t * h_cand
logits_t  = h_t @ W_hy + b_y
probs_t   = softmax(logits_t)
```

Where:

- `x_t` is the input feature vector at timestep `t` (shape `(B, D)`).
- `h_{t-1}` is the previous hidden state context vector (shape `(B, H)`).
- `r_t` is the reset gate deciding how much past memory to expose when proposing candidates.
- `z_t` is the update gate balancing past state retention `(1 - z_t)` against new candidate assimilation `z_t`.
- `h_reset` is the context memory masked by the reset gate prior to recurrent projection.
- `h_cand` ($\tilde{h}_t$) is the candidate state proposed for the current timestep.
- `h_t` is the updated hidden state synthesized via convex linear state interpolation.
- `logits_t` and `probs_t` are unnormalized scores and normalized class probabilities over `K` output classes.

```text
Input Sequence X (B, T, D)
       │
       ▼  Timestep Slice x_t (B, D)
 ┌────────────────────────────────────────────────────────┐
 │                    GRU Cell (Micro-Flow)               │
 │                                                        │
 │  r_t = σ(x_t W_xr + h_{t-1} W_hr + b_r)   [Reset]      │
 │  z_t = σ(x_t W_xz + h_{t-1} W_hz + b_z)   [Update]     │
 │  h~_t = tanh(x_t W_xh + (r_t ⊙ h_{t-1}) W_hh + b_h)   │
 │                                                        │
 │  h_t = (1 - z_t) ⊙ h_{t-1}  +  z_t ⊙ h~_t             │
 └─────────────────────────┬──────────────────────────────┘
                           │
                           ▼  Hidden State h_t (B, H)
                 Dense Readout Head
                           │
                           ▼  Logits o_t (B, K)
                     Softmax
                           │
                           ▼  Probabilities y_hat_t (B, K)
```

## Project structure

```text
10-gru/
├── README.md
├── src/
│   └── gru.py                    # Pure NumPy GRU with BPTT and synthetic benchmark
└── steps/
    └── step-01-mathematics.md    # Complete mathematical foundations (Q1–Q12)
```

## Learning workflow

Every project in this repository adheres to the standard 10-step sequence:

| Step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Mathematics | Complete | [`steps/step-01-mathematics.md`](steps/step-01-mathematics.md) (Q1–Q12: Cho vs. PyTorch formulations, linear gradient highway proof, fused gate vectorization, parameter budgets, 3-step forward trace, 4-way BPTT calculus, and code mapping) |
| 2. NumPy implementation | Complete | [`src/gru.py`](src/gru.py): 3-tier hierarchy (`gru_cell_forward`, `gru_forward`, `train_gru`), fused gate projections, batch-mean BPTT adjoints, and global norm clipping |
| 3. Understand forward pass | Complete | Verified exact hidden states and sequence loss matching manual derivation down to 6 decimal places ($L_1 = 0.724228, L_2 = 0.751293, L_3 = 0.598178, L_{\text{seq}} = 2.073698$) |
| 4. Understand loss | Complete | Multi-class categorical cross-entropy loss with numerical floor and consistent batch-mean scaling ($dL_t / do_t = (P_t - Y_t) / B$) |
| 5. Derive gradients | Complete | Derived 4-way BPTT recurrent error: linear highway path, reset candidate path, update gate path, and reset gate path |
| 6. Implement backpropagation | Complete | Implemented full BPTT in `gru_backward` with exact gradient accumulation for all 11 parameter tensors |
| 7. Train model | Complete | Trained GRU on synthetic long-range bit memory benchmark ($T = 20$), achieving 100% terminal accuracy |
| 8. Debug & analyze | Complete | Validated all parameter gradients against symmetric numerical finite differences ($< 10^{-8}$ rel error) and verified $B = 2$ batched gradient scaling |
| 9. PyTorch implementation | Pending | Equivalent pipeline using `torch.nn.GRU` and analysis of Cho et al. single-bias vs. PyTorch double-bias layout |
| 10. Compare results | Pending | Compare NumPy vs. PyTorch forward states, parameter layouts, and convergence dynamics |

## Parameters and tensor shapes

For the baseline verification example in `src/gru.py` ($B = 1, T = 3, D = 2, H = 3, K = 2$):

| Layer / Parameter | Tensor Key | Shape | Parameters | Mathematical Role |
| :--- | :--- | :--- | :--- | :--- |
| **Reset Input Weights** | `W_xr` | (2, 3) | 6 | Projects input features to reset gate |
| **Reset Recurrent Weights** | `W_hr` | (3, 3) | 9 | Projects previous state to reset gate |
| **Reset Bias** | `b_r` | (1, 3) | 3 | Additive bias for reset gate |
| **Update Input Weights** | `W_xz` | (2, 3) | 6 | Projects input features to update gate |
| **Update Recurrent Weights** | `W_hz` | (3, 3) | 9 | Projects previous state to update gate |
| **Update Bias** | `b_z` | (1, 3) | 3 | Additive bias for update gate |
| **Candidate Input Weights** | `W_xh` | (2, 3) | 6 | Projects input features to candidate state |
| **Candidate Recurrent Weights**| `W_hh` | (3, 3) | 9 | Projects reset-modulated context to candidate |
| **Candidate Bias** | `b_h` | (1, 3) | 3 | Additive bias for candidate state |
| **Output Readout Weights** | `W_hy` | (3, 2) | 6 | Projects hidden context to vocabulary/classes |
| **Output Readout Bias** | `b_y` | (1, 2) | 2 | Additive bias for output logits |
| **Total Parameters** | — | — | **62** | $3 \times (DH + H^2 + H) + HK + K$ |

### Comparative Architecture Audit:

| Architecture | Gated Sub-Networks | States per Step | Recurrent Parameters ($D=2, H=3$) | Total Params ($K=2$) |
| :--- | :--- | :--- | :--- | :--- |
| **Simple RNN (`07-rnn`)** | 1 (`h`) | 1 ($h_t$) | $DH + H^2 + H = 18$ | **26** |
| **LSTM (`09-lstm`)** | 4 ($f, i, \tilde{C}, o$) | 2 ($h_t, C_t$) | $4(DH + H^2 + H) = 72$ | **80** |
| **GRU (`10-gru`)** | 3 ($r, z, \tilde{h}$) | 1 ($h_t$) | $3(DH + H^2 + H) = 54$ | **62** *(22.5% fewer than LSTM)* |

## NumPy implementation details

`src/gru.py` follows the repository's 3-tier hierarchy and 8-part modular layout:

### 1. Fused Gate Vectorization (Micro-Flow)

Rather than evaluating reset and update gates via separate matrix multiplications, `gru_cell_forward` concatenates weights into $2H$ blocks:

```python
W_x_rz = np.hstack([parameters["W_xr"], parameters["W_xz"]])  # (D, 2H)
W_h_rz = np.hstack([parameters["W_hr"], parameters["W_hz"]])  # (H, 2H)
b_rz = np.hstack([parameters["b_r"], parameters["b_z"]])      # (1, 2H)

Z_rz = current_input @ W_x_rz + previous_hidden @ W_h_rz + b_rz  # (B, 2H)
z_r_pre, z_z_pre = Z_rz[:, :hidden_dim], Z_rz[:, hidden_dim:]

reset_gate = sigmoid(z_r_pre)
update_gate = sigmoid(z_z_pre)
```

### 2. Reset-Before Modulation & Convex Combination

```python
# Reset gate masks prior memory before recurrent affine projection
reset_modulated_hidden = reset_gate * previous_hidden

z_h_pre = (
    current_input @ parameters["W_xh"]
    + reset_modulated_hidden @ parameters["W_hh"]
    + parameters["b_h"]
)
candidate_hidden = np.tanh(z_h_pre)

# Convex linear state interpolation (Gradient Highway)
current_hidden = (1.0 - update_gate) * previous_hidden + update_gate * candidate_hidden
```

### 3. The 4-Way BPTT Gradient Highway

In `gru_backward`, error propagates to prior hidden state $h_{t-1}$ across all four active channels derived in `step-01-mathematics.md`:

```python
path_1 = total_hidden_error * (1.0 - z_t)           # Path 1: Direct Linear State Highway
path_2 = delta_h_reset * r_t                        # Path 2: Candidate State via Reset Mod
path_3 = delta_z_z @ parameters["W_hz"].T           # Path 3: Update Gate Recurrent
path_4 = delta_z_r @ parameters["W_hr"].T           # Path 4: Reset Gate Recurrent

future_hidden_error = path_1 + path_2 + path_3 + path_4
```

### 4. Batch-Mean Gradient Scaling

To guarantee that BPTT differentiates the exact batch-mean cross-entropy loss reported by `sequence_cross_entropy`, `output_score_gradients` normalizes by batch size $B$:

```python
def output_score_gradients(output_probabilities, target_sequence):
    batch_size = output_probabilities.shape[0]
    return (output_probabilities - target_sequence) / batch_size
```

## Numerical verification & results

Running `src/gru.py` verifies the exact numerical derivations from `steps/step-01-mathematics.md`:

### Checkpoint 1: Step-by-Step Forward Loss Parity

| Timestep | Input $x_t$ | Target Class | Logits $o_t$ | Softmax $\hat{y}_t$ | Cross-Entropy Loss $L_t$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$t = 1$** | `[1.0, -0.5]` | Class 1 | `[0.004091, -0.057133]` | `[0.515301, 0.484699]` | **0.724228** |
| **$t = 2$** | `[0.5, 1.0]` | Class 0 | `[-0.030079, 0.083016]` | `[0.471756, 0.528244]` | **0.751293** |
| **$t = 3$** | `[-1.0, 0.5]` | Class 1 | `[-0.048257, 0.151657]` | `[0.450187, 0.549813]` | **0.598178** |
| **Total** | — | — | — | — | **$L_{\text{seq}} = 2.073698$** |

*All forward losses match theoretical derivations down to 6 decimal places (assertion tolerance: $10^{-4}$).*

### Checkpoint 2: Finite-Difference Gradient Checking

Analytical BPTT gradients verified against central symmetric numerical differentiation ($\epsilon = 10^{-5}$):

| Parameter Tensor | Coordinate | Analytical Gradient | Numerical Gradient | Relative Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `W_xr` (Reset Input) | `[0, 1]` | `+0.0039889` | `+0.0039889` | $3.19 \times 10^{-10}$ | **PASS** |
| `W_hr` (Reset Recurrent) | `[1, 2]` | `-0.0001539` | `-0.0001539` | $2.83 \times 10^{-8}$ | **PASS** |
| `b_r` (Reset Bias) | `[0, 0]` | `-0.0002414` | `-0.0002414` | $5.52 \times 10^{-9}$ | **PASS** |
| `W_xz` (Update Input) | `[1, 0]` | `-0.0229573` | `-0.0229573` | $8.75 \times 10^{-10}$ | **PASS** |
| `W_hz` (Update Recurrent) | `[0, 2]` | `+0.0005481` | `+0.0005481` | $1.96 \times 10^{-9}$ | **PASS** |
| `b_z` (Update Bias) | `[0, 1]` | `+0.0307266` | `+0.0307266` | $3.51 \times 10^{-10}$ | **PASS** |
| `W_xh` (Candidate Input) | `[0, 2]` | `-0.0338336` | `-0.0338336` | $2.50 \times 10^{-10}$ | **PASS** |
| `W_hh` (Candidate Recurrent) | `[1, 1]` | `-0.0166886` | `-0.0166886` | $4.49 \times 10^{-10}$ | **PASS** |
| `b_h` (Candidate Bias) | `[0, 1]` | `-0.2357425` | `-0.2357425` | $1.11 \times 10^{-11}$ | **PASS** |
| `W_hy` (Readout Weights) | `[0, 0]` | `-0.0616777` | `-0.0616777` | $4.26 \times 10^{-11}$ | **PASS** |
| `b_y` (Readout Bias) | `[0, 1]` | `-0.4372450` | `-0.4372450` | $3.21 \times 10^{-11}$ | **PASS** |

*Batched gradient scaling verified: for a duplicated $B = 2$ batch, analytical gradients match single-sample values within $10^{-10}$ and numerical gradients within $10^{-6}$.*

### Checkpoint 3: Long-Range Bit Memory Demonstration ($T = 20$)

Benchmarking memory retention over 20 timesteps (signal at $t = 0$, 19 noise steps):

```text
Synthetic Dataset: 30 sequences, length T = 20. Target supervised at every timestep.
Simple RNN: Stagnates at 53.3% accuracy (loss ~186.98) due to exponential Jacobian decay (0.4^19 ≈ 2.7e-8).
GRU:        Converges to 100.0% accuracy (loss 0.0000) via the linear retention highway.
```

Gate dynamics inspection confirms why the GRU succeeds:
- At signal step $t = 0$: Update gate $z_0 = 0.125$ writes candidate information into memory.
- At noise steps $t = 10$: Update gate $z_{10} = 0.125$ suppresses incoming noise, keeping the linear retention highway $1 - z_{10} = 0.875$ open across time.

## Run the project

Run the complete validation suite from the repository root:

```bash
python3 10-gru/src/gru.py
```

Or execute directly from the `10-gru/src` directory:

```bash
python3 gru.py
```

## Key takeaways

1. **The Reset-Before Formulation**: Cho et al. (2014) masks prior memory $h_{t-1}$ *before* the recurrent matrix projection $W_{hh}$ (`(r_t ⊙ h_{t-1}) @ W_hh + b_h`). This maintains a single bias vector per gate (3 total), whereas PyTorch maintains double biases (6 total) for fused cuDNN projection.
2. **The Linear State Interpolation Gradient Highway**: The convex combination $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ provides a direct linear path for error propagation ($\partial h_t / \partial h_{t-1} \approx I$ when $z_t \to 0$).
3. **Parameter Efficiency**: By coupling memory retention and candidate assimilation into a single update gate $z_t$ and eliminating the separate cell state $C_t$, the GRU achieves comparable temporal retention to an LSTM with ~25% fewer parameters and higher computational throughput.
4. **Batch-Mean Consistency**: Defining output error as $\delta_o = (P - Y) / B$ aligns BPTT gradients exactly with the reported batch-mean cross-entropy loss, ensuring stability across varying mini-batch sizes.

## Next steps

1. Build `11-sequence-prediction` applying recurrent gating to Autoregressive Language Modeling and text generation.
2. Implement bidirectional GRUs (BiGRU) and stacked multi-layer recurrent encoders for sequence-to-sequence learning.
3. Benchmark against PyTorch (`torch.nn.GRU`), verifying weight transposition layouts and parameter parity.
