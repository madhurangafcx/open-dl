# 05 — CNN From Scratch

## Goal

Build and understand a Convolutional Neural Network (CNN) from first principles
with NumPy. Unlike a fully connected Artificial Neural Network (built in `02-ann-from-scratch`),
a CNN preserves 2D spatial topology, drastically reduces parameter counts through weight sharing,
and learns translation-equivariant visual features (edges, textures, contours, and shapes).

The network implements the canonical computer vision forward pipeline:

```text
Z      = conv2d(image, kernel, bias, stride)
A      = relu(Z)
P_map  = max_pool2d(A, pool_size=(2, 2), stride=1)
a_flat = flatten(P_map)
logits = a_flat @ W_dense + b_dense
probs  = softmax(logits)
```

Where:

- `image` is the single-channel or multi-channel input image (shape `(H, W)` or `(C_in, H, W)`).
- `kernel` contains the learnable spatial filters (shape `(K_h, K_w)` or `(C_out, C_in, K_h, K_w)`).
- `bias` is the additive scalar or per-channel offset (shape `(1,)` or `(C_out,)`).
- `Z` is the linear cross-correlated pre-activation feature map (shape `(H_out, W_out)`).
- `A` is the activated feature map after ReLU (`np.maximum(0, Z)`).
- `P_map` is the spatially downsampled feature representation after Max Pooling.
- `a_flat` is the unrolled 1D activation vector fed into the dense classification head.
- `W_dense` and `b_dense` are linear projection weights and biases for class scoring.
- `probs` contains normalized class probabilities summing to 1.0.

```text
Input Image (5 × 5)
       │
       ▼
Conv2D (Kernel: 3 × 3, Stride: 1, Valid padding)
       │
       ▼  Feature Map Z (3 × 3)
     ReLU
       │
       ▼  Activated Map A (3 × 3)
MaxPool2D (Window: 2 × 2, Stride: 1)
       │
       ▼  Pooled Map P_map (2 × 2)
    Flatten
       │
       ▼  Vector a_flat (4 × 1)
  Dense Head (W: 4 × 2, b: 1 × 2)
       │
       ▼  Logits (2 classes)
    Softmax
       │
       ▼  Probabilities P (1 × 2)
```

## Project structure

```text
05-cnn-from-scratch/
├── README.md
├── src/
│   └── cnn.py                    # Pure NumPy 2D convolution and forward pipeline
└── steps/
    └── step-01-mathematics.md    # Complete mathematical foundations (Q1–Q12)
```

## Learning workflow

Every project in this repository adheres to the standard 10-step sequence:

| Step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Mathematics | Complete | `steps/step-01-mathematics.md` (Q1–Q12: discrete cross-correlation, spatial geometry, 9-patch manual calculations, 25-patch padded calculations, receptive field, pooling suite, full backpropagation calculus, and complete cnn.py mapping) |
| 2. NumPy implementation | In Progress | `src/cnn.py`: `conv2d` with universal stride support (`int`, `tuple`, `np.ndarray`), pre-allocated outputs, and sliding receptive fields |
| 3. Understand forward pass | Complete | Verified exact feature map values matching manual derivation for all 9 cells |
| 4. Understand loss | In Progress | Multi-class categorical cross-entropy loss with numerical clipping |
| 5. Derive gradients | Complete | Derived `dL/db`, `dL/dK` (cross-correlation with input), and `dL/dX` (full convolution with 180° rotated filter) in Q11 |
| 6. Implement backpropagation | Pending | Next: implement `conv2d_backward`, `max_pool2d_backward`, and `relu_backward` |
| 7. Train model | Pending | Train scratch CNN on image classification |
| 8. Debug & analyze | Pending | Validate gradient correctness via numerical gradient checking |
| 9. PyTorch implementation | Pending | Equivalent pipeline using `torch.nn.Conv2d`, `torch.nn.MaxPool2d`, and `torch.nn.Linear` |
| 10. Compare results | Pending | Compare NumPy vs PyTorch forward outputs, gradient tensors, and runtime |

## Parameters and tensor shapes

For the baseline single-channel verification example in `src/cnn.py`:

| Layer / Tensor | Shape | Parameters | Function / Operation |
| :--- | :--- | :--- | :--- |
| `image` (Input) | (5, 5) | 0 | Raw grayscale pixel matrix |
| `kernel` (Weights) | (3, 3) | 9 | Spatial feature detector |
| `bias` (Offset) | (1,) | 1 | Additive scalar offset |
| `output` (Feature map `Z`) | (3, 3) | 0 | `(5 - 3) // 1 + 1 = 3` |
| `stride` (Step size) | `int` / `np.ndarray` | — | Supports scalar `1`, tuple `(s_h, s_w)`, or array `np.array([s_h, s_w])` |

### General spatial dimension formula

For any 2D input with height `H_in`, width `W_in`, filter size `(K_h, K_w)`, padding `(P_h, P_w)`, and stride `(S_h, S_w)`:

```text
H_out = floor((H_in - K_h + 2 * P_h) / S_h) + 1
W_out = floor((W_in - K_w + 2 * P_w) / S_w) + 1
```

## NumPy implementation details

`src/cnn.py` implements the core building block of the network:

### Flexible stride parsing

The function accepts stride as an integer scalar, tuple, list, or arbitrary `np.ndarray` using flattened dimension resolution:

```python
if isinstance(stride, (int, float)):
    s_h = s_w = int(stride)
else:
    stride_arr = np.asarray(stride).flatten()
    if stride_arr.size == 1:
        s_h = s_w = int(stride_arr[0])
    elif stride_arr.size == 2:
        s_h, s_w = int(stride_arr[0]), int(stride_arr[1])
    else:
        raise ValueError(f"stride must contain 1 or 2 values, got {stride_arr}")
```

### Sliding window convolution

```python
out_h = (img_h - k_h) // s_h + 1
out_w = (img_w - k_w) // s_w + 1
feature_map = np.zeros((out_h, out_w), dtype=np.float32)

for i in range(out_h):
    for j in range(out_w):
        r_start = i * s_h
        c_start = j * s_w
        patch = image[r_start : r_start + k_h, c_start : c_start + k_w]
        feature_map[i, j] = np.sum(patch * kernel) + bias
```

## Numerical verification & results

Running `src/cnn.py` verifies the exact numerical calculations detailed in `steps/step-01-mathematics.md`:

### Input Image (5 × 5):
```text
[[3. 3. 2. 1. 0.]
 [0. 0. 1. 3. 1.]
 [3. 1. 2. 2. 3.]
 [2. 0. 0. 2. 2.]
 [2. 0. 0. 0. 1.]]
```

### Filter Kernel (3 × 3):
```text
[[0. 1. 2.]
 [2. 2. 0.]
 [0. 1. 2.]]
```

### Convolved Feature Map with padding=0 (3 × 3):
```text
[[12. 12. 17.]
 [10. 17. 19.]
 [ 9.  6. 14.]]
```

### Convolved Feature Map with padding=1 (5 × 5):
```text
[[ 6. 14. 17. 11.  3.]
 [14. 12. 12. 17. 11.]
 [ 8. 10. 17. 19. 13.]
 [11.  9.  6. 14. 12.]
 [ 6.  4.  4.  6.  4.]]
```

### Downsampled Pooling Comparison on 5 × 5 Map (pool_size=2, stride=2):

| Pooling Mode | Output Map (2 × 2) | Arithmetic Operation |
| :--- | :--- | :--- |
| **Max Pooling** | `[[14.0, 17.0], [11.0, 19.0]]` | `np.max(window)` (peaks/edges) |
| **Average Pooling** | `[[11.5, 14.25], [9.5, 14.0]]` | `np.mean(window)` (smooth context) |
| **Min Pooling** | `[[6.0, 11.0], [8.0, 6.0]]` | `np.min(window)` (darkest/lowest) |

Notice the invariant relationship verified across every window:
```text
Min (6.0)  <=  Avg (11.5)  <=  Max (14.0)
Min (11.0) <=  Avg (14.25) <=  Max (17.0)
Min (8.0)  <=  Avg (9.5)   <=  Max (11.0)
Min (6.0)  <=  Avg (14.0)  <=  Max (19.0)
```

## Run the project

Run the script from the terminal:

```bash
python3 05-cnn-from-scratch/src/cnn.py
```

Or execute directly from the `05-cnn-from-scratch/src` directory:

```bash
python3 cnn.py
```

## Key takeaways

1. **Cross-correlation vs. True convolution**: Deep learning implements discrete 2D cross-correlation in the forward pass. True mathematical convolution (flipping the kernel 180°) emerges naturally in backpropagation when calculating the error gradient with respect to input activations `dX`.
2. **Inductive bias**: By enforcing local connectivity and weight tying, CNNs achieve translation equivariance while slashing parameter counts by over 99% compared to dense layers.
3. **Padding preservation**: `padding = 0` (valid) strips `K - 1` spatial pixels; `padding = (K - 1) // 2` (same) adds zero borders to keep spatial resolution constant.
4. **Pooling mechanics**: `max_pool2d`, `avg_pool2d`, and `min_pool2d` contain zero learnable parameters, downsampling spatial grids while extracting distinct signal characteristics (peaks vs. averages vs. minima).

## Next steps

1. Extend `conv2d` to multi-channel input tensors `(C_in, H, W)` and multiple filters `(C_out, C_in, K_h, K_w)`.
2. Implement `relu` activation and `flatten` layers in `src/cnn.py`.
3. Implement backward passes (`conv2d_backward`, `max_pool2d_backward`, `avg_pool2d_backward`, `relu_backward`) derived in `steps/step-01-mathematics.md`.

