# Step 1 — Mathematics for a Convolutional Neural Network

Goal: understand the complete mathematics of a Convolutional Neural Network (CNN)
from first principles. This document provides the rigorous theoretical foundations,
spatial geometry, tensor dimensions, forward calculations, and backpropagation calculus
that will be implemented with pure NumPy from scratch.

---

## Q1 — Why Convolutional Neural Networks? (The Computer Vision Problem)

In traditional Multi-Layer Perceptrons (MLP / ANN, as built in `02-ann-from-scratch`),
every input feature connects to every neuron in the subsequent layer via a dense matrix
multiplication:

```math
Z = X W + b
```

While dense layers succeed on low-dimensional tabular data and toy problems (such as XOR),
they fundamentally break down when applied to digital images for three mathematical and
structural reasons:

### 1. Parameter explosion

Consider a modest digital image of size 256 × 256 pixels with 3 color channels (RGB):

```text
Total input features = 256 × 256 × 3 = 196,608 features
```

If the first hidden layer contains 1,000 neurons, the weight matrix `W` requires:

```text
Weights = 196,608 × 1,000 = 196,608,000 parameters (~196.6 million floats)
```

At 4 bytes per 32-bit float (`np.float32`), this single layer demands approximately
786 MB of memory for parameters alone, excluding activations and gradients. Training
hundreds of millions of weights causes severe overfitting, extreme memory exhaustion,
and slow convergence.

### 2. Destruction of spatial 2D topology

A dense layer requires flattening the 2D grid into a 1D column vector:

```text
Image (H, W, C) ──► Flatten ──► Vector (H × W × C, 1)
```

In an image, pixels that are spatially close together (vertically, horizontally, or
diagonally) share high mutual information (e.g., edges, contours, textures). Flattening
destroys this 2D geometry: pixel `(0, 0)` is placed adjacent to `(0, 1)`, but pixel
`(0, 0)` is separated from its vertical neighbor `(1, 0)` by `W` elements. A dense
layer treats distant pixels with the exact same initial weight capacity as neighboring
pixels.

### 3. Lack of translation equivariance and invariance

In an MLP, each weight is dedicated to a specific coordinate `(i, j)`. If a feature
(e.g., an eye, a corner, or an edge) shifts by even a few pixels, an entirely different
set of weights must be trained to recognize it.

### The CNN solution: Three fundamental inductive biases

Convolutional Neural Networks resolve these shortcomings through three core mathematical
principles:

1. **Local connectivity (receptive fields)**: Neurons in a convolutional layer connect
   only to a small local patch of the input (e.g., 3 × 3 or 5 × 5 pixels), rather than
   the entire image.
2. **Parameter sharing (weight tying)**: The exact same kernel (weights and bias) slides
   across every spatial position of the image. If an edge detector is useful at the top-left,
   it is equally useful at the bottom-right.
3. **Translation equivariance**: Shifting an object in the input image by `(Δx, Δy)`
   shifts the response in the output feature map by `(Δx, Δy)`. Formally, for transformation
   operator `T`:

```math
\mathrm{Conv}(T(X)) = T(\mathrm{Conv}(X))
```

---

## Q2 — Network architecture and tensor shapes

The baseline CNN architecture in this project implements the canonical feedforward
feature-extraction pipeline:

```text
Input Image (5 × 5)
       │
       ▼
Conv2D (Kernel: 3 × 3, Stride: 1, Valid padding)
       │
       ▼  Feature Map (3 × 3)
     ReLU
       │
       ▼  Activated Map (3 × 3)
MaxPool2D (Window: 2 × 2, Stride: 1)
       │
       ▼  Pooled Map (2 × 2)
    Flatten
       │
       ▼  Vector (4 × 1)
 Dense Head (W: 4 × 2, b: 1 × 2)
       │
       ▼  Logits (2 classes)
    Softmax
       │
       ▼  Probabilities P
```

### Parameter and shape summary

| Component | Tensor Name | Dimensions | Number of Parameters | Description |
| :--- | :--- | :--- | :--- | :--- |
| Input Image | `X` | (5, 5) | 0 | Single-channel 2D grayscale image |
| Convolutional Filter | `K` | (3, 3) | 3 × 3 = 9 | Learnable spatial filter |
| Convolutional Bias | `b` | (1,) | 1 | Learnable additive scalar bias |
| Pre-activation Feature Map | `Z` | (3, 3) | 0 | Raw convolved linear response |
| Non-linear Activation | `A` | (3, 3) | 0 | `ReLU(Z) = max(0, Z)` |
| Downsampled Pooling Map | `P_map` | (2, 2) | 0 | Max pooling spatial reduction |
| Flattened Vector | `a_flat` | (4, 1) | 0 | Unrolled 1D activation vector |
| Dense Layer Weights | `W_dense` | (4, 2) | 4 × 2 = 8 | Linear classification projection |
| Dense Layer Bias | `b_dense` | (1, 2) | 2 | Classification class offsets |
| Output Logits | `Z_out` | (1, 2) | 0 | Unnormalized class scores |
| Final Probabilities | `P` | (1, 2) | 0 | `Softmax(Z_out)` |

---

## Q3 — Mathematical foundations: Continuous convolution, discrete convolution, and cross-correlation

The term "convolution" in deep learning is technically a slight misnomer: virtually all
deep learning frameworks (NumPy, PyTorch, TensorFlow, JAX) implement **cross-correlation**.
Understanding the distinction is vital for deriving backpropagation.

### Continuous 2D convolution

In continuous mathematics and signal processing, the convolution of an input function
`f(x, y)` with a filter kernel `g(x, y)` is defined as:

```math
(f * g)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta) \, g(x - \tau, y - \eta) \, \mathrm{d}\tau \, \mathrm{d}\eta
```

Notice the negative signs `- τ` and `- η`: the kernel `g` is inverted (flipped horizontally
and vertically by 180 degrees) before sliding over `f`. This flip gives continuous
convolution its essential **commutative property**:

```math
f * g = g * f
```

### Discrete 2D convolution

For a discrete 2D digital image `I` and a discrete kernel `K` of size `K_h × K_w`:

```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) \, K(m, n)
```

Flipping the kernel:

```math
K^{\mathrm{rot}}(m, n) = K(K_h - 1 - m, K_w - 1 - n)
```

### Discrete 2D cross-correlation (Deep learning "convolution")

In deep learning, the kernel is **not** flipped prior to sliding across the image.
Instead, indices are added directly:

```math
(I \star K)(i, j) = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} I(i + m, j + n) \, K(m, n)
```

With an additive scalar bias `b`:

```math
Z_{i, j} = \left( \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} I(i + m, j + n) \, K(m, n) \right) + b
```

### Why does deep learning use cross-correlation?

1. **Parameters are learned**: In traditional signal processing, filters are hand-crafted
   (e.g., Sobel, Gaussian, Laplacian), so flipping is necessary for theoretical consistency.
   In deep learning, the kernel weights `K` are randomly initialized and learned via gradient
   descent. If a flipped kernel is optimal, the network simply learns the flipped weights
   directly.
2. **Computational efficiency**: Avoiding the 180-degree array flip during the forward
   pass eliminates unnecessary memory indexing operations.
3. **Where true convolution reappears**: During **backpropagation**, computing the gradient
   of the loss with respect to the input activations `X` mathematically requires true
   convolution with the 180-degree flipped kernel!

---

## Q4 — Forward pass: Step-by-step numerical walkthrough (Matching cnn.py)

To achieve absolute mathematical transparency, we perform the exact numerical forward pass
matching `05-cnn-from-scratch/src/cnn.py`.

### 1. Input definitions

Input image `image` (size 5 × 5, single channel):

```text
         Col 0   Col 1   Col 2   Col 3   Col 4
Row 0  [   3       3       2       1       0   ]
Row 1  [   0       0       1       3       1   ]
Row 2  [   3       1       2       2       3   ]
Row 3  [   2       0       0       2       2   ]
Row 4  [   2       0       0       0       1   ]
```

Filter kernel `kernel` (size 3 × 3):

```text
         Col 0   Col 1   Col 2
Row 0  [   0       1       2   ]
Row 1  [   2       2       0   ]
Row 2  [   0       1       2   ]
```

Bias term:

```text
bias = 0.0
```

Stride:

```text
stride = 1  (s_h = 1, s_w = 1)
```

### 2. Output dimensions calculation

```math
H_{\mathrm{out}} = \frac{5 - 3}{1} + 1 = 3
```

```math
W_{\mathrm{out}} = \frac{5 - 3}{1} + 1 = 3
```

The output feature map `Z` is a 3 × 3 matrix pre-allocated with zeros:

```text
Z = np.zeros((3, 3), dtype=np.float32)
```

---

### 3. Step-by-step calculation of all 9 output cells

At each output location `(i, j)`, the algorithm performs:
1. **Receptive field extraction**: Extract the 3 × 3 patch starting at `r_start = i * 1`, `c_start = j * 1`:
   `patch = image[r_start : r_start + 3, c_start : c_start + 3]`
2. **Hadamard product**: Compute element-wise multiplication `patch ⊙ kernel`.
3. **Summation and bias**: Sum all 9 products and add `bias = 0.0`.

```math
Z_{i, j} = \sum_{m=0}^{2} \sum_{n=0}^{2} \left( \mathrm{patch}_{m, n} \cdot K_{m, n} \right) + b
```

---

#### Cell (0, 0) — Top-Left

- **Slice**: `image[0:3, 0:3]`
- **Patch**:
  ```text
  [3, 3, 2]
  [0, 0, 1]
  [3, 1, 2]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (3 × 0) = 0    (3 × 1) = 3    (2 × 2) = 4
  (0 × 2) = 0    (0 × 2) = 0    (1 × 0) = 0
  (3 × 0) = 0    (1 × 1) = 1    (2 × 2) = 4
  ```
- **Sum**:
  ```math
  Z_{0, 0} = (0 + 3 + 4) + (0 + 0 + 0) + (0 + 1 + 4) + 0.0 = 12.0
  ```

---

#### Cell (0, 1) — Top-Middle

- **Slice**: `image[0:3, 1:4]`
- **Patch**:
  ```text
  [3, 2, 1]
  [0, 1, 3]
  [1, 2, 2]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (3 × 0) = 0    (2 × 1) = 2    (1 × 2) = 2
  (0 × 2) = 0    (1 × 2) = 2    (3 × 0) = 0
  (1 × 0) = 0    (2 × 1) = 2    (2 × 2) = 4
  ```
- **Sum**:
  ```math
  Z_{0, 1} = (0 + 2 + 2) + (0 + 2 + 0) + (0 + 2 + 4) + 0.0 = 12.0
  ```

---

#### Cell (0, 2) — Top-Right

- **Slice**: `image[0:3, 2:5]`
- **Patch**:
  ```text
  [2, 1, 0]
  [1, 3, 1]
  [2, 2, 3]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (2 × 0) = 0    (1 × 1) = 1    (0 × 2) = 0
  (1 × 2) = 2    (3 × 2) = 6    (1 × 0) = 0
  (2 × 0) = 0    (2 × 1) = 2    (3 × 2) = 6
  ```
- **Sum**:
  ```math
  Z_{0, 2} = (0 + 1 + 0) + (2 + 6 + 0) + (0 + 2 + 6) + 0.0 = 17.0
  ```

---

#### Cell (1, 0) — Middle-Left

- **Slice**: `image[1:4, 0:3]`
- **Patch**:
  ```text
  [0, 0, 1]
  [3, 1, 2]
  [2, 0, 0]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (0 × 0) = 0    (0 × 1) = 0    (1 × 2) = 2
  (3 × 2) = 6    (1 × 2) = 2    (2 × 0) = 0
  (2 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  ```
- **Sum**:
  ```math
  Z_{1, 0} = (0 + 0 + 2) + (6 + 2 + 0) + (0 + 0 + 0) + 0.0 = 10.0
  ```

---

#### Cell (1, 1) — Center

- **Slice**: `image[1:4, 1:4]`
- **Patch**:
  ```text
  [0, 1, 3]
  [1, 2, 2]
  [0, 0, 2]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (0 × 0) = 0    (1 × 1) = 1    (3 × 2) = 6
  (1 × 2) = 2    (2 × 2) = 4    (2 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (2 × 2) = 4
  ```
- **Sum**:
  ```math
  Z_{1, 1} = (0 + 1 + 6) + (2 + 4 + 0) + (0 + 0 + 4) + 0.0 = 17.0
  ```

---

#### Cell (1, 2) — Middle-Right

- **Slice**: `image[1:4, 2:5]`
- **Patch**:
  ```text
  [1, 3, 1]
  [2, 2, 3]
  [0, 2, 2]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (1 × 0) = 0    (3 × 1) = 3    (1 × 2) = 2
  (2 × 2) = 4    (2 × 2) = 4    (3 × 0) = 0
  (0 × 0) = 0    (2 × 1) = 2    (2 × 2) = 4
  ```
- **Sum**:
  ```math
  Z_{1, 2} = (0 + 3 + 2) + (4 + 4 + 0) + (0 + 2 + 4) + 0.0 = 19.0
  ```

---

#### Cell (2, 0) — Bottom-Left

- **Slice**: `image[2:5, 0:3]`
- **Patch**:
  ```text
  [3, 1, 2]
  [2, 0, 0]
  [2, 0, 0]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (3 × 0) = 0    (1 × 1) = 1    (2 × 2) = 4
  (2 × 2) = 4    (0 × 2) = 0    (0 × 0) = 0
  (2 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  ```
- **Sum**:
  ```math
  Z_{2, 0} = (0 + 1 + 4) + (4 + 0 + 0) + (0 + 0 + 0) + 0.0 = 9.0
  ```

---

#### Cell (2, 1) — Bottom-Middle

- **Slice**: `image[2:5, 1:4]`
- **Patch**:
  ```text
  [1, 2, 2]
  [0, 0, 2]
  [0, 0, 0]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (1 × 0) = 0    (2 × 1) = 2    (2 × 2) = 4
  (0 × 2) = 0    (0 × 2) = 0    (2 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  ```
- **Sum**:
  ```math
  Z_{2, 1} = (0 + 2 + 4) + (0 + 0 + 0) + (0 + 0 + 0) + 0.0 = 6.0
  ```

---

#### Cell (2, 2) — Bottom-Right

- **Slice**: `image[2:5, 2:5]`
- **Patch**:
  ```text
  [2, 2, 3]
  [0, 2, 2]
  [0, 0, 1]
  ```
- **Element-wise multiplication with kernel**:
  ```text
  (2 × 0) = 0    (2 × 1) = 2    (3 × 2) = 6
  (0 × 2) = 0    (2 × 2) = 4    (2 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (1 × 2) = 2
  ```
- **Sum**:
  ```math
  Z_{2, 2} = (0 + 2 + 6) + (0 + 4 + 0) + (0 + 0 + 2) + 0.0 = 14.0
  ```

---

### Final verified feature map (padding = 0, unpadded)

Collecting all 9 values yields the exact matrix produced by `conv2d_unpadded` (or `conv2d` with `padding=0`) in `src/cnn.py`:

```text
Z_unpadded = np.array([
    [12.0, 12.0, 17.0],
    [10.0, 17.0, 19.0],
    [ 9.0,  6.0, 14.0]
], dtype=np.float32)
```

```text
Z_unpadded.shape = (3, 3)
```

---

### 4. Forward pass with padding = 1 ("Same" padding) — Matching cnn.py

When `padding = 1`, a 1-pixel border of zeros is added around the entire 5 × 5 image.

#### Padded Canvas (7 × 7):

```math
H_{\mathrm{padded}} = H_{\mathrm{in}} + 2 P = 5 + 2(1) = 7
```

```math
W_{\mathrm{padded}} = W_{\mathrm{in}} + 2 P = 5 + 2(1) = 7
```

```text
         Col 0  Col 1  Col 2  Col 3  Col 4  Col 5  Col 6
Row 0  [   0      0      0      0      0      0      0   ]  <- Top zero border
Row 1  [   0    ( 3      3      2      1      0 )    0   ]  <- Original Row 0
Row 2  [   0    ( 0      0      1      3      1 )    0   ]  <- Original Row 1
Row 3  [   0    ( 3      1      2      2      3 )    0   ]  <- Original Row 2
Row 4  [   0    ( 2      0      0      2      2 )    0   ]  <- Original Row 3
Row 5  [   0    ( 2      0      0      0      1 )    0   ]  <- Original Row 4
Row 6  [   0      0      0      0      0      0      0   ]  <- Bottom zero border
```

#### Output dimensions with padding = 1:

```math
H_{\mathrm{out}} = \left\lfloor \frac{7 - 3}{1} \right\rfloor + 1 = 5
```

```math
W_{\mathrm{out}} = \left\lfloor \frac{7 - 3}{1} \right\rfloor + 1 = 5
```

The output feature map preserves the exact 5 × 5 spatial dimensions of the original image!

#### The Three Topological Regions of the 5 × 5 Padded Feature Map

When sliding a 3 × 3 kernel over the 7 × 7 zero-padded canvas, the 25 output positions partition into three distinct mathematical regions based on how many zero-padded boundary elements enter the receptive field:

```text
       Col 0      Col 1      Col 2      Col 3      Col 4
Row 0 [ Corner ] [  Edge  ] [  Edge  ] [  Edge  ] [ Corner ]
Row 1 [  Edge  ] [Interior] [Interior] [Interior] [  Edge  ]
Row 2 [  Edge  ] [Interior] [Interior] [Interior] [  Edge  ]
Row 3 [  Edge  ] [Interior] [Interior] [Interior] [  Edge  ]
Row 4 [ Corner ] [  Edge  ] [  Edge  ] [  Edge  ] [ Corner ]
```

1. **Corner Cells (4 positions: `(0,0)`, `(0,4)`, `(4,0)`, `(4,4)`)**:
   - The 3 × 3 receptive field overlaps **2 padding borders** simultaneously (e.g. top and left).
   - Exactly **5 elements** in the receptive field are zeros, and only **4 elements** come from the original image.
2. **Edge Cells (12 positions)**:
   - The receptive field overlaps **1 padding border** (top, bottom, left, or right).
   - Exactly **3 elements** are zeros, and **6 elements** come from the original image.
3. **Interior Cells (9 positions: rows 1 to 3, cols 1 to 3)**:
   - The receptive field is completely within the original image boundary (**0 zeros**).
   - All **9 elements** are original image pixels.
   - Consequently, this 3 × 3 interior sub-matrix is **identical** to the unpadded convolution result `Z_unpadded`!

---

#### Step-by-Step Arithmetic for Key Representative Cells

#### 1. Top-Left Corner Cell (0, 0) — Overlaps Top & Left Zero Borders

- **Receptive Field Slice**: `padded[0:3, 0:3]`
  ```text
  [ 0,  0,  0 ]   <- Top zero border
  [ 0,  3,  3 ]   <- Left zero border + original pixels (0,0) and (0,1)
  [ 0,  0,  0 ]   <- Left zero border + original pixels (1,0) and (1,1)
  ```
- **Hadamard product with kernel**:
  ```text
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  (0 × 2) = 0    (3 × 2) = 6    (3 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  ```
- **Sum & Bias**:
  ```math
  Z_{\mathrm{padded}, 0, 0} = 0 + 0 + 0 + 0 + 6 + 0 + 0 + 0 + 0 + 0.0 = 6.0
  ```

---

#### 2. Top-Right Corner Cell (0, 4) — Overlaps Top & Right Zero Borders

- **Receptive Field Slice**: `padded[0:3, 4:7]`
  ```text
  [ 0,  0,  0 ]   <- Top zero border
  [ 1,  0,  0 ]   <- Original pixels (0,3), (0,4) + Right zero border
  [ 3,  1,  0 ]   <- Original pixels (1,3), (1,4) + Right zero border
  ```
- **Hadamard product with kernel**:
  ```text
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  (1 × 2) = 2    (0 × 2) = 0    (0 × 0) = 0
  (3 × 0) = 0    (1 × 1) = 1    (0 × 2) = 0
  ```
- **Sum & Bias**:
  ```math
  Z_{\mathrm{padded}, 0, 4} = 0 + 0 + 0 + 2 + 0 + 0 + 0 + 1 + 0 + 0.0 = 3.0
  ```

---

#### 3. Top-Middle Edge Cell (0, 2) — Overlaps Top Zero Border Only

- **Receptive Field Slice**: `padded[0:3, 2:5]`
  ```text
  [ 0,  0,  0 ]   <- Top zero border
  [ 3,  2,  1 ]   <- Original row 0, cols 1:4
  [ 0,  1,  3 ]   <- Original row 1, cols 1:4
  ```
- **Hadamard product with kernel**:
  ```text
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  (3 × 2) = 6    (2 × 2) = 4    (1 × 0) = 0
  (0 × 0) = 0    (1 × 1) = 1    (3 × 2) = 6
  ```
- **Sum & Bias**:
  ```math
  Z_{\mathrm{padded}, 0, 2} = (0 + 0 + 0) + (6 + 4 + 0) + (0 + 1 + 6) + 0.0 = 17.0
  ```

---

#### 4. Interior Center Cell (2, 2) — Zero Padding Does Not Enter

- **Receptive Field Slice**: `padded[2:5, 2:5] = image[1:4, 1:4]`
  ```text
  [ 0,  1,  3 ]
  [ 1,  2,  2 ]
  [ 0,  0,  2 ]
  ```
- **Hadamard product with kernel**:
  ```text
  (0 × 0) = 0    (1 × 1) = 1    (3 × 2) = 6
  (1 × 2) = 2    (2 × 2) = 4    (2 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (2 × 2) = 4
  ```
- **Sum & Bias**:
  ```math
  Z_{\mathrm{padded}, 2, 2} = (0 + 1 + 6) + (2 + 4 + 0) + (0 + 0 + 4) + 0.0 = 17.0
  ```
  *(Notice this is mathematically identical to unpadded cell `(1, 1)`!)*

---

#### 5. Bottom-Right Corner Cell (4, 4) — Overlaps Bottom & Right Zero Borders

- **Receptive Field Slice**: `padded[4:7, 4:7]`
  ```text
  [ 2,  2,  0 ]   <- Original pixels (3,3), (3,4) + Right zero border
  [ 0,  1,  0 ]   <- Original pixels (4,3), (4,4) + Right zero border
  [ 0,  0,  0 ]   <- Bottom zero border
  ```
- **Hadamard product with kernel**:
  ```text
  (2 × 0) = 0    (2 × 1) = 2    (0 × 2) = 0
  (0 × 2) = 0    (1 × 2) = 2    (0 × 0) = 0
  (0 × 0) = 0    (0 × 1) = 0    (0 × 2) = 0
  ```
- **Sum & Bias**:
  ```math
  Z_{\mathrm{padded}, 4, 4} = 0 + 2 + 0 + 0 + 2 + 0 + 0 + 0 + 0 + 0.0 = 4.0
  ```

---

#### Complete 25-Cell Calculation Table (Matching `output_pad1` in `cnn.py`)

| Output Cell `(i, j)` | Region Type | Active Dot-Product Terms `(patch * kernel)` | Sum + Bias | Final Value |
| :--- | :--- | :--- | :--- | :--- |
| `(0, 0)` | Corner | `(3*2)` | `6 + 0.0` | **6.0** |
| `(0, 1)` | Edge | `(3*2) + (3*2) + (1*2)` | `6 + 6 + 2 + 0.0` | **14.0** |
| `(0, 2)` | Edge | `(3*2) + (2*2) + (1*1) + (3*2)` | `6 + 4 + 1 + 6 + 0.0` | **17.0** |
| `(0, 3)` | Edge | `(2*2) + (1*2) + (3*1) + (1*2)` | `4 + 2 + 3 + 2 + 0.0` | **11.0** |
| `(0, 4)` | Corner | `(1*2) + (1*1)` | `2 + 1 + 0.0` | **3.0** |
| `(1, 0)` | Edge | `(3*1) + (3*2) + (3*1) + (1*2)` | `3 + 6 + 3 + 2 + 0.0` | **14.0** |
| `(1, 1)` | Interior | `(3*1) + (2*2) + (1*1) + (2*2)` | `3 + 4 + 1 + 4 + 0.0` | **12.0** |
| `(1, 2)` | Interior | `(2*1) + (1*2) + (1*2) + (2*1) + (2*2)` | `2 + 2 + 2 + 2 + 4 + 0.0` | **12.0** |
| `(1, 3)` | Interior | `(1*1) + (1*2) + (3*2) + (2*1) + (3*2)` | `1 + 2 + 6 + 2 + 6 + 0.0` | **17.0** |
| `(1, 4)` | Edge | `(3*2) + (1*2) + (3*1)` | `6 + 2 + 3 + 0.0` | **11.0** |
| `(2, 0)` | Edge | `(3*2) + (2*1)` | `6 + 2 + 0.0` | **8.0** |
| `(2, 1)` | Interior | `(1*2) + (3*2) + (1*2)` | `2 + 6 + 2 + 0.0` | **10.0** |
| `(2, 2)` | Interior | `(1*1) + (3*2) + (1*2) + (2*2) + (2*2)` | `1 + 6 + 2 + 4 + 4 + 0.0` | **17.0** |
| `(2, 3)` | Interior | `(3*1) + (1*2) + (2*2) + (2*2) + (2*1) + (2*2)` | `3 + 2 + 4 + 4 + 2 + 4 + 0.0` | **19.0** |
| `(2, 4)` | Edge | `(1*1) + (2*2) + (3*2) + (2*1)` | `1 + 4 + 6 + 2 + 0.0` | **13.0** |
| `(3, 0)` | Edge | `(3*1) + (1*2) + (2*2) + (2*1)` | `3 + 2 + 4 + 2 + 0.0` | **11.0** |
| `(3, 1)` | Interior | `(1*1) + (2*2) + (2*2)` | `1 + 4 + 4 + 0.0` | **9.0** |
| `(3, 2)` | Interior | `(2*1) + (2*2)` | `2 + 4 + 0.0` | **6.0** |
| `(3, 3)` | Interior | `(2*1) + (3*2) + (2*2) + (1*2)` | `2 + 6 + 4 + 2 + 0.0` | **14.0** |
| `(3, 4)` | Edge | `(3*1) + (2*2) + (2*2) + (1*1)` | `3 + 4 + 4 + 1 + 0.0` | **12.0** |
| `(4, 0)` | Corner | `(2*1) + (2*2)` | `2 + 4 + 0.0` | **6.0** |
| `(4, 1)` | Edge | `(2*2)` | `4 + 0.0` | **4.0** |
| `(4, 2)` | Edge | `(2*2)` | `4 + 0.0` | **4.0** |
| `(4, 3)` | Edge | `(2*1) + (2*2)` | `2 + 4 + 0.0` | **6.0** |
| `(4, 4)` | Corner | `(2*1) + (1*2)` | `2 + 2 + 0.0` | **4.0** |

---

#### Full 5 × 5 convolved feature map (`output_pad1`):

```text
Z_padded = np.array([
    [ 6.0, 14.0, 17.0, 11.0,  3.0],
    [14.0, 12.0, 12.0, 17.0, 11.0],
    [ 8.0, 10.0, 17.0, 19.0, 13.0],
    [11.0,  9.0,  6.0, 14.0, 12.0],
    [ 6.0,  4.0,  4.0,  6.0,  4.0]
], dtype=np.float32)
```

#### Interior Preservation Property:

Notice that extracting rows 1:4 and cols 1:4 from `Z_padded` yields the exact unpadded 3 × 3 feature map:

```text
Z_padded[1:4, 1:4] = np.array([
    [12.0, 12.0, 17.0],
    [10.0, 17.0, 19.0],
    [ 9.0,  6.0, 14.0]
])
```

---

#### Boundary Effect Analysis: Why `padding = 0` produces 12.0 while `padding = 1` produces 6.0 at [0, 0]

In `src/cnn.py`, running the script produces:
- `padding=0 at [0, 0]: 12.0`
- `padding=1 at [0, 0]: 6.0`

Why does adding zero-padding change the output at coordinate `[0, 0]`?

1. **In Unpadded Convolution (`padding = 0`)**:
   - The top-left corner of the kernel `K[0, 0]` aligns with image pixel `X[0, 0] = 3`.
   - The entire 3 × 3 kernel lies on active image data `image[0:3, 0:3]`.
   - All 9 elements contribute non-zero products, summing to `12.0`.

2. **In Same-Padded Convolution (`padding = 1`)**:
   - The **center** of the kernel `K[1, 1]` aligns with image pixel `X[0, 0] = 3`.
   - The top row of the kernel and the left column of the kernel lie outside the image on the zero border.
   - 5 of the 9 kernel weights multiply `0.0`:
     ```math
     0 \cdot K_{0,0} + 0 \cdot K_{0,1} + 0 \cdot K_{0,2} + 0 \cdot K_{1,0} + 0 \cdot K_{2,0} = 0
     ```
   - Only 4 kernel weights multiply real image pixels, producing `3 * K[1, 1] = 3 * 2 = 6.0`.

> [!NOTE]
> Same padding guarantees that every pixel of the original image serves as the **center** of at least one receptive field. However, pixels near the boundary have smaller effective receptive fields (fewer non-zero neighbors), causing natural edge attenuation.

---


## Q5 — Spatial arithmetic: Output dimensions, stride, and padding

The dimensions of the feature map are governed by the input dimensions `(H_in, W_in)`,
the filter size `(K_h, K_w)`, the zero-padding amount `(P_h, P_w)`, and the stride
`(S_h, S_w)`.

### The universal output dimension formula

```math
H_{\mathrm{out}} = \left\lfloor \frac{H_{\mathrm{in}} - K_h + 2 P_h}{S_h} \right\rfloor + 1
```

```math
W_{\mathrm{out}} = \left\lfloor \frac{W_{\mathrm{in}} - K_w + 2 P_w}{S_w} \right\rfloor + 1
```

Where:
- `H_in, W_in`: Height and width of the input image or feature map.
- `K_h, K_w`: Height and width of the convolutional filter.
- `P_h, P_w`: Number of padded zero-pixels added symmetrically to the borders (top/bottom and left/right).
- `S_h, S_w`: Stride step size along the vertical and horizontal dimensions.
- `⌊ · ⌋`: Floor operator (integer division `//` in Python).

### Derivation of the formula

Why do we subtract `K`, divide by `S`, and add `1`?

1. **Initial placement (`offset = 0`)**: The filter is placed at the top-left corner.
   This initial placement consumes `K` pixels of the input and accounts for the first
   valid output pixel. This explains the `+ 1` term.
2. **Remaining traverse distance**: After the initial placement, the remaining distance
   available for the filter to shift across is:
   ```text
   Remaining Distance = (Dimension + 2P) - K
   ```
3. **Step size quantization**: Each slide moves by `S` pixels. The total number of shifts
   allowed without exceeding the boundary is:
   ```text
   Number of valid shifts = ⌊ (Remaining Distance) / S ⌋
   ```
4. **Total output positions**:
   ```text
   Total positions = (Initial position) + (Number of shifts)
                   = 1 + ⌊ (Dimension + 2P - K) / S ⌋
   ```

### Padding modes: Valid vs. Same

#### 1. "Valid" padding (No padding, `P = 0`)

The filter operates only within the strict boundaries of the original image:

```math
H_{\mathrm{out}} = \left\lfloor \frac{H_{\mathrm{in}} - K_h}{S_h} \right\rfloor + 1
```

- **Effect**: Spatial dimensions shrink with every convolutional layer. For a 3 × 3
  kernel, each layer strips 2 pixels from height and width.
- **Border loss**: Pixels on the extreme perimeter participate in far fewer receptive
  fields than center pixels, causing edge information loss.

#### 2. "Same" padding (Preserve spatial dimensions)

Zero-padding is added around the border so that for stride `S = 1`, the output size
exactly equals the input size:

```math
H_{\mathrm{out}} = H_{\mathrm{in}} \implies H_{\mathrm{in}} = H_{\mathrm{in}} - K_h + 2P_h + 1
```

Rearranging for `P_h`:

```math
2P_h = K_h - 1 \implies P_h = \frac{K_h - 1}{2}
```

For odd filter sizes:
- Filter `3 × 3`: `P = (3 - 1) / 2 = 1` pixel of zero-padding on all four borders.
- Filter `5 × 5`: `P = (5 - 1) / 2 = 2` pixels of zero-padding on all four borders.
- Filter `7 × 7`: `P = (7 - 1) / 2 = 3` pixels of zero-padding on all four borders.

### Stride mechanics and flexible array handling

Stride determines how many pixels the filter jumps between consecutive receptive fields:

```text
r_start = i * s_h
c_start = j * s_w
```

When stride `S > 1`, the convolution simultaneously performs feature extraction and
**spatial downsampling**, halving the resolution when `S = 2`.

#### Arbitrary array input in `conv2d`

In real-world architectures, stride can be specified as a scalar integer (`stride = 2`),
a tuple (`stride = (2, 2)`), or a NumPy array (`stride = np.array([2, 2])`).
Using `np.asarray(stride).flatten()` provides seamless compatibility across all representations:

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
        raise ValueError("stride must contain 1 or 2 values")
```

### Numerical walkthrough: Strided unpadded convolution (stride = 2) — Matching cnn.py

In `src/cnn.py` (lines 55–58), the comments explicitly discuss strided convolution:
```python
# For a 5x5 image, 3x3 kernel, and stride (2, 2):
# out_h = (5 - 3) // 2 + 1 = 2 rows
# out_w = (5 - 3) // 2 + 1 = 2 columns
```

Let us work through the exact mathematical forward pass for this configuration:

#### Output dimensions calculation:

```math
H_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = \lfloor 1.0 \rfloor + 1 = 2
```

```math
W_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = \lfloor 1.0 \rfloor + 1 = 2
```

The feature map shrinks from 3 × 3 (with `stride=1`) down to **2 × 2** (with `stride=2`).

#### Step-by-step arithmetic for all 4 output positions:

1. **Cell (0, 0)**:
   - Starts at `r = 0 * 2 = 0`, `c = 0 * 2 = 0`
   - Receptive field: `image[0:3, 0:3]`
   - Calculation:
     ```math
     Z_{0, 0} = (3 \cdot 0 + 3 \cdot 1 + 2 \cdot 2) + (0 \cdot 2 + 0 \cdot 2 + 1 \cdot 0) + (3 \cdot 0 + 1 \cdot 1 + 2 \cdot 2) + 0.0 = 12.0
     ```

2. **Cell (0, 1)**:
   - Starts at `r = 0 * 2 = 0`, `c = 1 * 2 = 2`
   - Receptive field: `image[0:3, 2:5]` (jumps over column 1)
   - Calculation:
     ```math
     Z_{0, 1} = (2 \cdot 0 + 1 \cdot 1 + 0 \cdot 2) + (1 \cdot 2 + 3 \cdot 2 + 1 \cdot 0) + (2 \cdot 0 + 2 \cdot 1 + 3 \cdot 2) + 0.0 = 17.0
     ```

3. **Cell (1, 0)**:
   - Starts at `r = 1 * 2 = 2`, `c = 0 * 2 = 0` (jumps over row 1)
   - Receptive field: `image[2:5, 0:3]`
   - Calculation:
     ```math
     Z_{1, 0} = (3 \cdot 0 + 1 \cdot 1 + 2 \cdot 2) + (2 \cdot 2 + 0 \cdot 2 + 0 \cdot 0) + (2 \cdot 0 + 0 \cdot 1 + 0 \cdot 2) + 0.0 = 9.0
     ```

4. **Cell (1, 1)**:
   - Starts at `r = 1 * 2 = 2`, `c = 1 * 2 = 2` (jumps over row 1 and column 1)
   - Receptive field: `image[2:5, 2:5]`
   - Calculation:
     ```math
     Z_{1, 1} = (2 \cdot 0 + 2 \cdot 1 + 3 \cdot 2) + (0 \cdot 2 + 2 \cdot 2 + 2 \cdot 0) + (0 \cdot 0 + 0 \cdot 1 + 1 \cdot 2) + 0.0 = 14.0
     ```

#### Resulting 2 × 2 Strided Feature Map:

```text
Z_strided2 = np.array([
    [12.0, 17.0],
    [ 9.0, 14.0]
], dtype=np.float32)
```

#### The Subsampling Relationship:

Notice that striding by `S = 2` exactly subsamples the unit-stride unpadded feature map `Z_unpadded` at even row and column indices:

```math
Z_{\mathrm{stride}=2}[i, j] = Z_{\mathrm{stride}=1}[2i, \, 2j]
```

```text
Z_unpadded[0, 0] = 12.0  ──►  Z_strided2[0, 0] = 12.0
Z_unpadded[0, 2] = 17.0  ──►  Z_strided2[0, 1] = 17.0
Z_unpadded[2, 0] =  9.0  ──►  Z_strided2[1, 0] =  9.0
Z_unpadded[2, 2] = 14.0  ──►  Z_strided2[1, 1] = 14.0
```

---

## Q6 — Multi-channel and multi-filter convolution (4D tensors)

Real-world computer vision tasks do not operate on single 2D matrices. Input images
have color channels (RGB, `C_in = 3`), and intermediate layers produce multi-channel
feature volumes (`C_in = 16, 32, 64, 128, ...`).

### The 4D tensor structure

```text
Input Tensor X:   (N, C_in,  H_in,  W_in)
Kernel Tensor W:  (C_out, C_in, K_h, K_w)
Bias Vector b:    (C_out,)
Output Tensor Z:  (N, C_out, H_out, W_out)
```

Where:
- `N`: Batch size (number of images processed simultaneously).
- `C_in`: Number of input channels (e.g., 3 for RGB image, 1 for grayscale).
- `C_out`: Number of filters in the layer (number of output feature channels).
- `K_h, K_w`: Spatial height and width of each filter.

### The multi-channel mathematical equation

To compute a single scalar value at position `(i, j)` in output channel `c_out` for
example `n`:

```math
Z_{n, c_{\mathrm{out}}, i, j} = b_{c_{\mathrm{out}}} + \sum_{c_{\mathrm{in}}=0}^{C_{\mathrm{in}} - 1} \sum_{m=0}^{K_h - 1} \sum_{n'=0}^{K_w - 1} X_{n, c_{\mathrm{in}}, i \cdot S_h + m, j \cdot S_w + n'} \cdot W_{c_{\mathrm{out}}, c_{\mathrm{in}}, m, n'}
```

### Visualizing multi-channel reduction

```text
Input Volume (C_in = 3):        Filter c_out (3 slices):
┌───────────┐                   ┌───────┐
│ Channel 0 │  (Red)     *      │  W_0  │  ──┐
├───────────┤                   ├───────┤    │
│ Channel 1 │  (Green)   *      │  W_1  │  ──┼─► [ Sum of 3 slices ] + bias ──► Output Feature Map (c_out)
├───────────┤                   ├───────┤    │
│ Channel 2 │  (Blue)    *      │  W_2  │  ──┘
└───────────┘                   └───────┘
```

Each filter `c_out` consists of `C_in` separate 2D kernels. The 2D convolution is
performed on each channel independently, and their results are **summed element-wise across
all input channels**, followed by adding the scalar bias `b[c_out]`.

### Parameter count efficiency: CNN vs. Dense

Consider convolving an RGB image (`C_in = 3`) into 32 feature maps using 3 × 3 filters:

```text
Parameters per filter = (3 channels × 3 × 3) + 1 bias = 28 parameters
Total parameters for 32 filters = 32 × 28 = 896 parameters
```

In contrast, connecting a 32 × 32 × 3 input image (3,072 values) to 32 neurons in a dense
layer requires:

```text
Dense parameters = (3,072 × 32) + 32 = 98,336 parameters
```

The convolutional layer uses **less than 1%** of the parameters while preserving 2D
spatial layout and acquiring translation equivariance.

---

## Q7 — Non-linear activation: ReLU (Rectified Linear Unit)

Convolution is a purely linear operation: a linear combination of inputs plus a bias.
Stacking multiple convolutional layers without an intervening non-linear activation function
algebraically collapses into a single larger linear convolution:

```math
\mathrm{Conv}_2(\mathrm{Conv}_1(X)) = \mathrm{Conv}_{\mathrm{combined}}(X)
```

To learn complex, non-linear visual representations (corners, curves, object parts),
an element-wise activation function must be applied to every pre-activation value `Z`.

### Mathematical definition of ReLU

```math
\mathrm{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
```

```text
         ReLU(z)
            ▲
            │       /
            │      /
            │     /
            │    /
────────────┼───/──────► z
            │  0
```

### Applying ReLU to our verified feature map `Z`

From Q4, the convolved feature map `Z` is:

```text
Z = [[12.0, 12.0, 17.0],
     [10.0, 17.0, 19.0],
     [ 9.0,  6.0, 14.0]]
```

Since all values are positive (`Z > 0`):

```math
A = \mathrm{ReLU}(Z) = Z = \begin{bmatrix} 12.0 & 12.0 & 17.0 \\ 10.0 & 17.0 & 19.0 \\ 9.0 & 6.0 & 14.0 \end{bmatrix}
```

### Derivative for backpropagation

```math
\frac{\mathrm{d}\,\mathrm{ReLU}(z)}{\mathrm{d}z} = \mathbb{I}(z > 0) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
```

ReLU avoids the vanishing gradient problem inherent in sigmoid and tanh activations
because its derivative is exactly `1.0` for all positive inputs.

---

## Q8 — Pooling and downsampling mathematics (Max Pooling, Average Pooling & Min Pooling)

A pooling layer reduces the spatial dimensions `(H, W)` of feature maps while retaining
the most dominant visual features or contextual summaries.

### Why pooling is necessary
1. **Dimension reduction**: Reduces memory consumption and computational cost in downstream
   layers (e.g. cutting spatial resolution in half reduces dense parameters by 75%).
2. **Translation invariance**: If a feature activates at `(i, j)` or slightly shifted at
   `(i+1, j)`, the pooled response over the local window remains invariant.
3. **Receptive field enlargement**: Downsampling allows deeper layers to view a wider spatial
   region of the original input image without requiring larger filters.
4. **Zero learnable parameters**: Pooling layers contain **no weights and no biases**
   (`weights = 0`, `biases = 0`). They execute a fixed, deterministic routing or reduction
   operation.

---

### Mathematical definitions of the three pooling operations

For an input activation map `A`, a pooling window of size `P_h × P_w`, and stride `(S_h, S_w)`:

#### 1. Max Pooling (`max_pool2d`)
Selects the single largest element in each window:

```math
\mathrm{MaxPool}(A)_{i, j} = \max_{m=0}^{P_h - 1} \max_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

- **What it captures**: Peak feature presence (sharp edges, high-contrast textures, foreground signals).
- **Core question**: *"Did the target feature appear anywhere inside this local window?"*

#### 2. Average Pooling (`avg_pool2d`)
Calculates the arithmetic mean across all elements in each window:

```math
\mathrm{AvgPool}(A)_{i, j} = \frac{1}{P_h \cdot P_w} \sum_{m=0}^{P_h - 1} \sum_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

- **What it captures**: Overall region context, background brightness, smooth spatial statistics.
- **Core question**: *"What is the general, average activation level across this entire region?"*

#### 3. Min Pooling (`min_pool2d`)
Selects the single smallest element in each window:

```math
\mathrm{MinPool}(A)_{i, j} = \min_{m=0}^{P_h - 1} \min_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

- **What it captures**: Dark features on bright backgrounds (black text/ink, cracks, shadows, closest depth values in LiDAR).
- **Core question**: *"What is the minimum activation or closest proximity within this local window?"*

---

### Fundamental mathematical inequality

For any real-valued window of numbers, the minimum is always less than or equal to the
average, which is always less than or equal to the maximum:

```math
\mathrm{MinPool}(A)_{i, j} \le \mathrm{AvgPool}(A)_{i, j} \le \mathrm{MaxPool}(A)_{i, j}
```

---

### Numerical demonstration on the 5 × 5 padded feature map (Matching cnn.py)

From Q4, the convolved feature map `output_pad1` obtained from our 5 × 5 input image is:

```text
A_padded = np.array([
    [ 6.0, 14.0, 17.0, 11.0,  3.0],
    [14.0, 12.0, 12.0, 17.0, 11.0],
    [ 8.0, 10.0, 17.0, 19.0, 13.0],
    [11.0,  9.0,  6.0, 14.0, 12.0],
    [ 6.0,  4.0,  4.0,  6.0,  4.0]
], dtype=np.float32)
```

We apply a **2 × 2 pooling window** (`P_h = 2, P_w = 2`) with **stride 2** (`S_h = 2, S_w = 2`).

#### Output dimensions calculation:

```math
H_{\mathrm{pool}} = \left\lfloor \frac{5 - 2}{2} \right\rfloor + 1 = \lfloor 1.5 \rfloor + 1 = 2
```

```math
W_{\mathrm{pool}} = \left\lfloor \frac{5 - 2}{2} \right\rfloor + 1 = \lfloor 1.5 \rfloor + 1 = 2
```

The output for all three pooling operations is a **2 × 2** matrix.

#### Boundary truncation analysis (Why the 5th row and 5th column are excluded):

With an input dimension of 5, a pool size of `P = 2`, and non-overlapping stride `S = 2`:
- **Window 0** covers coordinate indices `0` and `1`.
- **Window 1** covers coordinate indices `2` and `3`.
- **Coordinate index 4** (the 5th row `[6.0, 4.0, 4.0, 6.0, 4.0]` and 5th column `[3.0, 11.0, 13.0, 12.0, 4.0]`) cannot form a valid 2-element window because coordinate `5` is outside the feature map bounds.

Mathematically, the floor division operator in:

```math
H_{\mathrm{pool}} = \left\lfloor \frac{5 - 2}{2} \right\rfloor + 1 = \lfloor 1.5 \rfloor + 1 = 1 + 1 = 2
```

strictly truncates the fractional remainder `0.5`, discarding incomplete boundary windows. In standard deep learning frameworks (such as PyTorch's `nn.MaxPool2d`), this is the universal default behavior (`ceil_mode=False`).

---

#### Step-by-step arithmetic for all 4 pooling windows:

#### Window (0, 0) — Top-Left: rows 0:2, cols 0:2
```text
Slice:
[  6.0, 14.0 ]
[ 14.0, 12.0 ]
```
- **Max Pooling**:
  ```math
  \mathrm{MaxPool}_{0, 0} = \max(6.0, 14.0, 14.0, 12.0) = \mathbf{14.0}
  ```
- **Average Pooling**:
  ```math
  \mathrm{AvgPool}_{0, 0} = \frac{6.0 + 14.0 + 14.0 + 12.0}{4} = \frac{46.0}{4} = \mathbf{11.5}
  ```
- **Min Pooling**:
  ```math
  \mathrm{MinPool}_{0, 0} = \min(6.0, 14.0, 14.0, 12.0) = \mathbf{6.0}
  ```
- **Inequality Check**: `6.0 <= 11.5 <= 14.0` (Min <= Avg <= Max verified).

---

#### Window (0, 1) — Top-Right: rows 0:2, cols 2:4
```text
Slice:
[ 17.0, 11.0 ]
[ 12.0, 17.0 ]
```
- **Max Pooling**:
  ```math
  \mathrm{MaxPool}_{0, 1} = \max(17.0, 11.0, 12.0, 17.0) = \mathbf{17.0}
  ```
- **Average Pooling**:
  ```math
  \mathrm{AvgPool}_{0, 1} = \frac{17.0 + 11.0 + 12.0 + 17.0}{4} = \frac{57.0}{4} = \mathbf{14.25}
  ```
- **Min Pooling**:
  ```math
  \mathrm{MinPool}_{0, 1} = \min(17.0, 11.0, 12.0, 17.0) = \mathbf{11.0}
  ```
- **Inequality Check**: `11.0 <= 14.25 <= 17.0` (Min <= Avg <= Max verified).

---

#### Window (1, 0) — Bottom-Left: rows 2:4, cols 0:2
```text
Slice:
[  8.0, 10.0 ]
[ 11.0,  9.0 ]
```
- **Max Pooling**:
  ```math
  \mathrm{MaxPool}_{1, 0} = \max(8.0, 10.0, 11.0, 9.0) = \mathbf{11.0}
  ```
- **Average Pooling**:
  ```math
  \mathrm{AvgPool}_{1, 0} = \frac{8.0 + 10.0 + 11.0 + 9.0}{4} = \frac{38.0}{4} = \mathbf{9.5}
  ```
- **Min Pooling**:
  ```math
  \mathrm{MinPool}_{1, 0} = \min(8.0, 10.0, 11.0, 9.0) = \mathbf{8.0}
  ```
- **Inequality Check**: `8.0 <= 9.5 <= 11.0` (Min <= Avg <= Max verified).

---

#### Window (1, 1) — Bottom-Right: rows 2:4, cols 2:4
```text
Slice:
[ 17.0, 19.0 ]
[  6.0, 14.0 ]
```
- **Max Pooling**:
  ```math
  \mathrm{MaxPool}_{1, 1} = \max(17.0, 19.0, 6.0, 14.0) = \mathbf{19.0}
  ```
- **Average Pooling**:
  ```math
  \mathrm{AvgPool}_{1, 1} = \frac{17.0 + 19.0 + 6.0 + 14.0}{4} = \frac{56.0}{4} = \mathbf{14.0}
  ```
- **Min Pooling**:
  ```math
  \mathrm{MinPool}_{1, 1} = \min(17.0, 19.0, 6.0, 14.0) = \mathbf{6.0}
  ```
- **Inequality Check**: `6.0 <= 14.0 <= 19.0` (Min <= Avg <= Max verified).

---

### Final verified pooling matrices (Matching cnn.py)

#### 1. Max Pooled Map (2 × 2):
```text
P_max = np.array([
    [14.0, 17.0],
    [11.0, 19.0]
], dtype=np.float32)
```

#### 2. Average Pooled Map (2 × 2):
```text
P_avg = np.array([
    [11.5,  14.25],
    [ 9.5,  14.0 ]
], dtype=np.float32)
```

#### 3. Min Pooled Map (2 × 2):
```text
P_min = np.array([
    [ 6.0, 11.0],
    [ 8.0,  6.0]
], dtype=np.float32)
```

---

### Comparison of practical use cases in deep learning

| Property | Max Pooling | Average Pooling | Min Pooling |
| :--- | :--- | :--- | :--- |
| **Formula** | `max(window)` | `mean(window)` | `min(window)` |
| **Sensitivity** | High (sensitive to peaks/edges) | Medium (smoothes out spikes) | High (sensitive to valleys/shadows) |
| **Preserved Signal** | Foreground features, sharp lines | Background tone, smooth texture | Dark pixels, holes, cracks, proximity |
| **Modern Application** | Feature extractor in early/middle CNN layers | **Global Average Pooling (GAP)** at final classifier head | Depth sensing (LiDAR), OCR, morphological erosion |
| **Post-ReLU Behavior** | Robust (preserves active activations) | Preserves positive energy | Collapses to `0.0` if any zero exists in window |

---


## Q9 — Receptive field mathematics

The **Theoretical Receptive Field (RF)** is the spatial region in the input image that
directly influences the activation of a specific neuron in layer `l`.

### Layer-by-layer growth equation

For layer `l` with kernel size `K_l` and stride `S_l`:

```math
RF_l = RF_{l-1} + (K_l - 1) \cdot J_{l-1}
```

Where `J_l` is the cumulative stride (jump) up to layer `l`:

```math
J_l = J_{l-1} \cdot S_l \quad \text{with} \quad J_0 = 1, \; RF_0 = 1
```

### The VGG principle: Stacking 3 × 3 convolutions

Why did VGGNet (and modern architectures like ResNet and ConvNeXt) standardize on small
3 × 3 filters instead of 5 × 5 or 7 × 7 filters?

Consider two consecutive 3 × 3 convolutions with stride 1:
- Layer 1 (`K_1 = 3`, `S_1 = 1`):
  ```text
  RF_1 = 1 + (3 - 1) × 1 = 3 × 3 pixels
  J_1 = 1 × 1 = 1
  ```
- Layer 2 (`K_2 = 3`, `S_2 = 1`):
  ```text
  RF_2 = 3 + (3 - 1) × 1 = 5 × 5 pixels
  ```

Two stacked 3 × 3 convolutions cover the exact same 5 × 5 receptive field as a single
5 × 5 convolution!

#### Mathematical comparison:

| Metric | One 5 × 5 Convolution | Two Stacked 3 × 3 Convolutions |
| :--- | :--- | :--- |
| Receptive Field | 5 × 5 pixels | 5 × 5 pixels |
| Weights per channel | 5 × 5 = 25 weights | 2 × (3 × 3) = 18 weights |
| Parameter Savings | Baseline (100%) | **28% reduction** |
| Non-linear Activations | 1 ReLU | **2 ReLUs** (greater expressive power) |

---

## Q10 — Flattening and the dense classification head (Softmax & Cross-Entropy)

To make a final categorical classification (e.g., Cat vs. Dog, or XOR binary class),
the spatial 2D feature map must be transformed into a 1D vector and passed into a
fully connected output layer.

### 1. Flattening

The pooled map `P_map` of shape `(2, 2)` is unrolled row-by-row into a 1D column vector:

```math
\mathrm{P\_map} = \begin{bmatrix} 17.0 & 19.0 \\ 17.0 & 19.0 \end{bmatrix} \implies a_{\mathrm{flat}} = \begin{bmatrix} 17.0 \\ 19.0 \\ 17.0 \\ 19.0 \end{bmatrix}
```

```text
a_flat.shape = (4, 1)   (or (1, 4) for a single example batch)
```

### 2. Dense layer forward projection

Let the dense output layer connect the 4 flattened features to 2 output classes:

```math
Z_{\mathrm{dense}} = a_{\mathrm{flat}} W_{\mathrm{dense}} + b_{\mathrm{dense}}
```

Where:
- `a_flat`: shape `(1, 4)`
- `W_dense`: shape `(4, 2)`
- `b_dense`: shape `(1, 2)`
- `Z_dense`: shape `(1, 2)` (class logits `[z_0, z_1]`)

### 3. Softmax activation

The Softmax function converts unnormalized real-valued logits `Z_dense` into valid
probabilities `P` that sum to `1.0`:

```math
P_k = \frac{e^{Z_{\mathrm{dense}, k} - \max(Z_{\mathrm{dense}})}}{\sum_{j=0}^{C-1} e^{Z_{\mathrm{dense}, j} - \max(Z_{\mathrm{dense}})}}
```

> [!TIP]
> Subtracting `max(Z)` is the standard numerical stability trick to prevent floating-point
> overflow when exponentiating large logits.

### 4. Categorical Cross-Entropy Loss

For true one-hot target vector `Y = [y_0, y_1]`:

```math
L = - \sum_{k=0}^{C-1} Y_k \ln(P_k + \epsilon)
```

Where `ε = 1e-15` guarantees numerical safety against `ln(0) = -inf`.

---

## Q11 — CNN backpropagation derivations and equations (The complete calculus)

Backpropagation in a CNN requires computing how the scalar loss `L` changes with respect
to every learnable parameter and input activation:

```text
dL/dZ_out ──► dL/dW_dense, dL/db_dense ──► dL/da_flat ──► dL/dP_map ──► dL/dA ──► dL/dZ ──► dL/dK, dL/db, dL/dX
```

Let upstream gradient arriving at feature map `Z` be denoted:

```math
\delta_{i, j} = \frac{\partial L}{\partial Z_{i, j}} \in \mathbb{R}^{H_{\mathrm{out}} \times W_{\mathrm{out}}}
```

---

### 1. Gradient with respect to Convolutional Bias `b`

Recall the forward definition of each output cell:

```math
Z_{i, j} = \left( \sum_{m} \sum_{n} X_{i \cdot S_h + m, \, j \cdot S_w + n} K_{m, n} \right) + b
```

Taking the partial derivative of `Z_{i, j}` with respect to `b`:

```math
\frac{\partial Z_{i, j}}{\partial b} = 1
```

By the multivariable chain rule, the bias contributes to every cell `(i, j)` in the
output feature map:

```math
\frac{\partial L}{\partial b} = \sum_{i=0}^{H_{\mathrm{out}} - 1} \sum_{j=0}^{W_{\mathrm{out}} - 1} \frac{\partial L}{\partial Z_{i, j}} \cdot \frac{\partial Z_{i, j}}{\partial b} = \sum_{i=0}^{H_{\mathrm{out}} - 1} \sum_{j=0}^{W_{\mathrm{out}} - 1} \delta_{i, j}
```

In NumPy:

```python
db = np.sum(dZ)
```

For multi-channel feature maps with batch size `N`, sum across `axis=(0, 2, 3)`:

```python
db = np.sum(dZ, axis=(0, 2, 3), keepdims=True)
```

---

### 2. Gradient with respect to Kernel Weights `K`

How does a single filter weight `K_{m, n}` affect the loss?
From the forward equation, the partial derivative is:

```math
\frac{\partial Z_{i, j}}{\partial K_{m, n}} = X_{i \cdot S_h + m, \, j \cdot S_w + n}
```

Applying the multivariable chain rule across all output positions `(i, j)` where `K_{m, n}`
was used:

```math
\frac{\partial L}{\partial K_{m, n}} = \sum_{i=0}^{H_{\mathrm{out}} - 1} \sum_{j=0}^{W_{\mathrm{out}} - 1} \frac{\partial L}{\partial Z_{i, j}} \cdot \frac{\partial Z_{i, j}}{\partial K_{m, n}} = \sum_{i=0}^{H_{\mathrm{out}} - 1} \sum_{j=0}^{W_{\mathrm{out}} - 1} \delta_{i, j} \cdot X_{i \cdot S_h + m, \, j \cdot S_w + n}
```

#### The fundamental insight:
Computing the gradient `∂L/∂K` is **itself a cross-correlation** between the input
patch `X` and the upstream error gradient `δ`!

In NumPy:

```python
dK = np.zeros_like(kernel)
for i in range(out_h):
    for j in range(out_w):
        patch = image[i * s_h : i * s_h + k_h, j * s_w : j * s_w + k_w]
        dK += dZ[i, j] * patch
```

---

### 3. Gradient with respect to Input Activations `X` (Backpropagating to earlier layers)

To allow gradients to flow back to preceding layers, we must compute `∂L/∂X_{r, c}` for
every input pixel `(r, c)`.

An input pixel `X_{r, c}` affects only those output cells `Z_{i, j}` whose receptive field
covers `(r, c)`:

```math
i \cdot S_h \le r < i \cdot S_h + K_h \quad \text{and} \quad j \cdot S_w \le c < j \cdot S_w + K_w
```

For unit stride (`S = 1`):

```math
\frac{\partial L}{\partial X_{r, c}} = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} \delta_{r - m, \, c - n} \cdot K_{m, n}
```

#### The emergence of true mathematical convolution:
Let us substitute variables with the 180-degree flipped kernel:

```math
K^{\mathrm{rot}}_{m', n'} = K_{K_h - 1 - m', \, K_w - 1 - n'}
```

Then:

```math
\frac{\partial L}{\partial X} = \delta *_{\mathrm{full}} \mathrm{rot}_{180}(K)
```

> [!IMPORTANT]
> This is the mathematical reason why true convolution with a flipped kernel exists in
> deep learning: **The backward pass of cross-correlation is a full convolution with the
> 180-degree rotated filter.**

---

### 4. Backpropagation through ReLU

ReLU acts as an element-wise switch:

```math
\frac{\partial L}{\partial Z_{i, j}} = \frac{\partial L}{\partial A_{i, j}} \cdot \mathbb{I}(Z_{i, j} > 0)
```

In NumPy:

```python
dZ = dA * (Z > 0)
```

---

### 5. Backpropagation through Pooling Layers (Max, Average & Min Pooling)

Because pooling operations do not contain learnable parameters, their backpropagation
step only requires routing the upstream gradient `dP` back to the input activation `dA`.

#### 5.1 Backpropagation through Max Pooling (The Argmax Mask)
During the forward pass of Max Pooling, only one element in each pooling window achieved
the maximum. Therefore, during backpropagation, the gradient flows **exclusively** to the
single neuron that achieved the maximum. All other neurons in that window receive `0.0`.

```math
\frac{\partial L}{\partial A_{r, c}} = \begin{cases} \delta^{\mathrm{pool}}_{i, j} & \text{if } (r, c) = \operatorname{argmax}(\mathrm{window}_{i, j}) \\ 0 & \text{otherwise} \end{cases}
```

In NumPy:
```python
dA = np.zeros_like(A)
for i in range(out_h):
    for j in range(out_w):
        window = A[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w]
        mask = (window == np.max(window))
        dA[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += mask * dP[i, j]
```

#### 5.2 Backpropagation through Average Pooling (Uniform Gradient Distribution)
In Average Pooling, each output neuron is the arithmetic mean of all elements in the pooling window:

```math
P_{i, j} = \frac{1}{P_h \cdot P_w} \sum_{m=0}^{P_h - 1} \sum_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

Therefore, the partial derivative with respect to each individual element in the window is constant:

```math
\frac{\partial P_{i, j}}{\partial A_{r, c}} = \frac{1}{P_h \cdot P_w}
```

By the chain rule, the upstream error gradient `\delta_{i, j}` is distributed **equally and uniformly**
across every neuron in the pooling window:

```math
\frac{\partial L}{\partial A_{r, c}} = \frac{1}{P_h \cdot P_w} \cdot \delta^{\mathrm{pool}}_{i, j}
```

In NumPy:
```python
dA = np.zeros_like(A)
for i in range(out_h):
    for j in range(out_w):
        dA[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += dP[i, j] / (p_h * p_w)
```

#### 5.3 Backpropagation through Min Pooling (The Argmin Mask)
Just like Max Pooling, Min Pooling routes the upstream gradient **exclusively** to the
single neuron that achieved the minimum:

```math
\frac{\partial L}{\partial A_{r, c}} = \begin{cases} \delta^{\mathrm{pool}}_{i, j} & \text{if } (r, c) = \operatorname{argmin}(\mathrm{window}_{i, j}) \\ 0 & \text{otherwise} \end{cases}
```

In NumPy:
```python
dA = np.zeros_like(A)
for i in range(out_h):
    for j in range(out_w):
        window = A[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w]
        mask = (window == np.min(window))
        dA[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += mask * dP[i, j]
```

---

## Q12 — Complete mathematical mapping to cnn.py implementation

This section provides an exhaustive, topic-by-topic mathematical cross-reference directly linking every line of code, formula, and numerical calculation in [`cnn.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py) to its underlying mathematical proof.

---

### Topic 1: Numerical Precision and Memory Footprint (`np.float32`)
- **Code reference**: [`cnn.py:L18-26`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L18-L26)

#### Mathematical Representation (IEEE 754):
Every floating-point scalar is stored in 32-bit single precision:
- **1 sign bit** (`s`)
- **8 exponent bits** (`e`), bias = 127
- **23 fraction/mantissa bits** (`m`)

```math
v = (-1)^s \times 2^{e - 127} \times \left( 1 + \sum_{i=1}^{23} b_{23 - i} 2^{-i} \right)
```

#### Memory Footprint Equations:

```math
\mathrm{Memory}_{\mathrm{FP32}} = N_{\mathrm{elements}} \times 4 \text{ bytes}
```

```math
\mathrm{Memory}_{\mathrm{FP64}} = N_{\mathrm{elements}} \times 8 \text{ bytes}
```

For our 5 × 5 image:
- FP32: 25 × 4 = 100 bytes
- FP64: 25 × 8 = 200 bytes (100% overhead)

#### Hardware & Computational Efficiency:
1. **Memory Bandwidth**: Deep learning workloads are predominantly memory-bandwidth bound. Halving memory footprint doubles throughput across memory buses.
2. **SIMD & Tensor Cores**: Standard modern processors (Intel AVX2, ARM NEON, NVIDIA Tensor Cores) pack twice as many FP32 operations into a single clock cycle compared to FP64 (e.g. 8 floats per 256-bit AVX register vs. 4 doubles).
3. **Sufficient Precision**: Neural network weights and activations are inherently stochastic; the 23-bit mantissa provides ~7 decimal digits of precision, which is more than sufficient for gradient descent convergence.

---

### Topic 2: Baseline Unpadded Discrete Convolution (`conv2d_unpadded`)
- **Code reference**: [`cnn.py:L35-87`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L35-L87)

#### 1. Universal Output Dimension Formula (Valid Padding, P = 0):

```math
H_{\mathrm{out}} = \left\lfloor \frac{H - K_h}{S_h} \right\rfloor + 1, \quad W_{\mathrm{out}} = \left\lfloor \frac{W - K_w}{S_w} \right\rfloor + 1
```

- **For unit stride (S = 1)**:
  ```math
  H_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{1} \right\rfloor + 1 = 3, \quad W_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{1} \right\rfloor + 1 = 3
  ```
- **For stride (2, 2)**:
  ```math
  H_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = 2, \quad W_{\mathrm{out}} = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = 2
  ```

#### 2. Memory Pre-Allocation Architecture:
- `np.zeros((H_out, W_out), dtype=np.float32)` allocates a contiguous block of `H_out * W_out * 4` bytes. In CPython, arrays use row-major contiguous memory indexing:
  ```math
  \mathrm{Address}(i, j) = \mathrm{BaseAddress} + (i \cdot W_{\mathrm{out}} + j) \times 4
  ```
- Python nested lists create pointer fragmentation: an outer list of pointers to inner lists of pointers to 28-byte `PyFloatObject` instances.

#### 3. Sliding Window Slicing and Hadamard Reduction:
For each coordinate `(i, j)`:
```math
r_{\mathrm{start}} = i \cdot S_h, \quad c_{\mathrm{start}} = j \cdot S_w
```

```math
\mathrm{patch}_{i, j} = X[r_{\mathrm{start}} : r_{\mathrm{start}} + K_h, \, c_{\mathrm{start}} : c_{\mathrm{start}} + K_w]
```

```math
Z_{i, j} = \left( \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} \mathrm{patch}_{m, n} \cdot K_{m, n} \right) + b
```

#### Verified Numerical Output (3 × 3):
```text
[[12.0, 12.0, 17.0],
 [10.0, 17.0, 19.0],
 [ 9.0,  6.0, 14.0]]
```

---

### Topic 3: Zero-Padded Discrete Convolution (`conv2d`)
- **Code reference**: [`cnn.py:L92-152`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L92-L152)

#### 1. Canvas Allocation and Insertion:
When `padding = 1`, the dimensions expand:
```math
H_{\mathrm{padded}} = H_{\mathrm{in}} + 2 P_h = 5 + 2(1) = 7
```
```math
W_{\mathrm{padded}} = W_{\mathrm{in}} + 2 P_w = 5 + 2(1) = 7
```

Mapping the input image into the center of the zero canvas:
```math
X_{\mathrm{padded}}[P_h : P_h + H_{\mathrm{in}}, \, P_w : P_w + W_{\mathrm{in}}] = X
```

#### 2. Mathematical Neutrality of Zero-Padding:
Why must the border be padded with `0.0` and not `1.0`?
```math
0.0 \cdot K_{m, n} = 0.0 \quad (\text{Additive identity in }\mathbb{R})
```
Padding with `0` contributes zero energy to the linear sum. If padded with `1.0`, the border pixels would contribute `1.0 * K_{m, n} = K_{m, n}`, creating artificial high-energy boundary artifacts that corrupt edge detectors.

#### 3. Spatial Dimension Preservation ("Same" Padding):
```math
H_{\mathrm{out}} = \left\lfloor \frac{7 - 3}{1} \right\rfloor + 1 = 5, \quad W_{\mathrm{out}} = \left\lfloor \frac{7 - 3}{1} \right\rfloor + 1 = 5
```

#### 4. Verified Numerical Output (`output_pad1`, 5 × 5):
```text
[[ 6.0, 14.0, 17.0, 11.0,  3.0],
 [14.0, 12.0, 12.0, 17.0, 11.0],
 [ 8.0, 10.0, 17.0, 19.0, 13.0],
 [11.0,  9.0,  6.0, 14.0, 12.0],
 [ 6.0,  4.0,  4.0,  6.0,  4.0]]
```

---

### Topic 4: Max Pooling Layer (`max_pool2d`)
- **Code reference**: [`cnn.py:L154-180`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L154-L180)

#### 1. Downsampling Dimension Formula:
For input `5 × 5`, window `2 × 2`, and stride `2`:
```math
H_{\mathrm{pool}} = \left\lfloor \frac{5 - 2}{2} \right\rfloor + 1 = 2, \quad W_{\mathrm{pool}} = \left\lfloor \frac{5 - 2}{2} \right\rfloor + 1 = 2
```

#### 2. Boundary Truncation:
The 5th row (`[6.0, 4.0, 4.0, 6.0, 4.0]`) and 5th column (`[3.0, 11.0, 13.0, 12.0, 4.0]`) are dropped because non-overlapping windows of size 2 cannot exceed the coordinate boundary:
- Window 0: coordinates `[0, 1]`
- Window 1: coordinates `[2, 3]`
- Coordinate `4`: remainder truncated by `floor((5 - 2)/2)`.

#### 3. Quadrant Arithmetic:
```math
\mathrm{MaxPool}_{0, 0} = \max(6.0, 14.0, 14.0, 12.0) = \mathbf{14.0}
```
```math
\mathrm{MaxPool}_{0, 1} = \max(17.0, 11.0, 12.0, 17.0) = \mathbf{17.0}
```
```math
\mathrm{MaxPool}_{1, 0} = \max(8.0, 10.0, 11.0, 9.0) = \mathbf{11.0}
```
```math
\mathrm{MaxPool}_{1, 1} = \max(17.0, 19.0, 6.0, 14.0) = \mathbf{19.0}
```

#### Verified Numerical Output (2 × 2):
```text
[[14.0, 17.0],
 [11.0, 19.0]]
```

---

### Topic 5: Average Pooling Layer (`avg_pool2d`)
- **Code reference**: [`cnn.py:L181-205`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L181-L205)

#### 1. Mathematical Operation:
```math
\mathrm{AvgPool}(A)_{i, j} = \frac{1}{P_h \cdot P_w} \sum_{m=0}^{P_h - 1} \sum_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

#### 2. Quadrant Arithmetic:
```math
\mathrm{AvgPool}_{0, 0} = \frac{6.0 + 14.0 + 14.0 + 12.0}{4} = \frac{46.0}{4} = \mathbf{11.5}
```
```math
\mathrm{AvgPool}_{0, 1} = \frac{17.0 + 11.0 + 12.0 + 17.0}{4} = \frac{57.0}{4} = \mathbf{14.25}
```
```math
\mathrm{AvgPool}_{1, 0} = \frac{8.0 + 10.0 + 11.0 + 9.0}{4} = \frac{38.0}{4} = \mathbf{9.5}
```
```math
\mathrm{AvgPool}_{1, 1} = \frac{17.0 + 19.0 + 6.0 + 14.0}{4} = \frac{56.0}{4} = \mathbf{14.0}
```

#### Verified Numerical Output (2 × 2):
```text
[[11.5,  14.25],
 [ 9.5,  14.0 ]]
```

---

### Topic 6: Min Pooling Layer (`min_pool2d`)
- **Code reference**: [`cnn.py:L207-231`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L207-L231)

#### 1. Mathematical Operation:
```math
\mathrm{MinPool}(A)_{i, j} = \min_{m=0}^{P_h - 1} \min_{n'=0}^{P_w - 1} A_{i \cdot S_h + m, \, j \cdot S_w + n'}
```

#### 2. Quadrant Arithmetic:
```math
\mathrm{MinPool}_{0, 0} = \min(6.0, 14.0, 14.0, 12.0) = \mathbf{6.0}
```
```math
\mathrm{MinPool}_{0, 1} = \min(17.0, 11.0, 12.0, 17.0) = \mathbf{11.0}
```
```math
\mathrm{MinPool}_{1, 0} = \min(8.0, 10.0, 11.0, 9.0) = \mathbf{8.0}
```
```math
\mathrm{MinPool}_{1, 1} = \min(17.0, 19.0, 6.0, 14.0) = \mathbf{6.0}
```

#### Verified Numerical Output (2 × 2):
```text
[[ 6.0, 11.0],
 [ 8.0,  6.0]]
```

---

### Topic 7: The Cross-Pooling Invariant Inequality Proof
- **Theorem**: For any finite non-empty real-valued window `W = {w_1, w_2, ..., w_k}`:
  ```math
  \min(W) \le \frac{1}{k} \sum_{i=1}^{k} w_i \le \max(W)
  ```
- **Proof**:
  Let `m = min(W)` and `M = max(W)`. By definition:
  ```math
  \forall i \in \{1, \dots, k\}, \quad m \le w_i \le M
  ```
  Summing across all `k` elements:
  ```math
  \sum_{i=1}^{k} m \le \sum_{i=1}^{k} w_i \le \sum_{i=1}^{k} M \implies k \cdot m \le \sum_{i=1}^{k} w_i \le k \cdot M
  ```
  Dividing by scalar `k > 0`:
  ```math
  m \le \frac{1}{k} \sum_{i=1}^{k} w_i \le M
  ```
  Equality `m = mean(W) = M` holds if and only if all elements in the window are identical (`w_1 = w_2 = ... = w_k`).

#### Quadrant Verification on `output_pad1`:

| Quadrant | Window Slice | Min Value | Avg Value | Max Value | Verified Chain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `(0, 0)` | `[[6, 14], [14, 12]]` | 6.0 | 11.50 | 14.0 | `6.0 <= 11.5 <= 14.0` |
| `(0, 1)` | `[[17, 11], [12, 17]]` | 11.0 | 14.25 | 17.0 | `11.0 <= 14.25 <= 17.0` |
| `(1, 0)` | `[[8, 10], [11, 9]]` | 8.0 | 9.50 | 11.0 | `8.0 <= 9.5 <= 11.0` |
| `(1, 1)` | `[[17, 19], [6, 14]]` | 6.0 | 14.00 | 19.0 | `6.0 <= 14.0 <= 19.0` |

---

### Topic 8: Execution Script Numerical Verification Audit
- **Code reference**: [`cnn.py:L233-317`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/05-cnn-from-scratch/src/cnn.py#L233-L317)

Every numerical print statement produced by running `python cnn.py` is audited below:

```text
====================================================================================================
EXECUTION STEP               NUMERICAL OUTPUT                                       STATUS
====================================================================================================
1. Input Image (5x5)         [[3, 3, 2, 1, 0], [0, 0, 1, 3, 1], ...]                VERIFIED (Exact)
2. Kernel (3x3)              [[0, 1, 2], [2, 2, 0], [0, 1, 2]]                      VERIFIED (Exact)
3. Convolved Feature Map     3x3 matrix: [[12, 12, 17], [10, 17, 19], [9, 6, 14]]   VERIFIED (Exact)
4. Valid Padding (pad=0)     [0, 0] = 12.0 (4-step breakdown: patch*K -> sum -> +b) VERIFIED (Exact)
5. Same Padding (pad=1)      7x7 canvas, 5x5 map, [0, 0] = 6.0 (5-step breakdown)   VERIFIED (Exact)
6. Max Pooled Map (2x2)      [[14.0, 17.0], [11.0, 19.0]]                           VERIFIED (Exact)
7. Average Pooled Map (2x2)  [[11.5, 14.25], [9.5, 14.0]]                           VERIFIED (Exact)
8. Min Pooled Map (2x2)      [[6.0, 11.0], [8.0, 6.0]]                              VERIFIED (Exact)
====================================================================================================
```

---

## Summary of the complete mathematical pipeline

```text
====================================================================================================
STEP                          FORWARD OPERATION                               BACKWARD GRADIENT
====================================================================================================
1. Conv2D Layer               Z[i,j] = sum(X_patch * K) + b                   dK = sum(dZ[i,j] * X_patch)
                                                                              db = sum(dZ)
                                                                              dX = dZ *_full rot180(K)

2. Non-linear Activation      A = max(0, Z)                                   dZ = dA * (Z > 0)

3. Downsampling (Pooling):
   - Max Pooling              P_map[i,j] = max(A_window)                      dA = dP * (A_window == max)
   - Average Pooling          P_map[i,j] = mean(A_window)                     dA = dP / (P_h * P_w)
   - Min Pooling              P_map[i,j] = min(A_window)                      dA = dP * (A_window == min)

4. Spatial Unrolling          a_flat = P_map.reshape(-1, 1)                   dP_map = da_flat.reshape(P_map.shape)

5. Dense Classification Head  Z_dense = a_flat @ W_dense + b_dense            dW_dense = a_flat.T @ dZ_dense
                                                                              db_dense = sum(dZ_dense, axis=0)
                                                                              da_flat  = dZ_dense @ W_dense.T

6. Probability Normalization  P = softmax(Z_dense)                            dZ_dense = (P - Y) / m
   & Cross-Entropy Loss       L = -sum(Y * log(P))
====================================================================================================
```

This completes the foundational mathematics for Project 05. The next step is validating
each of these formulas in executable NumPy code.

