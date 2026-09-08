# Step 1 — Mathematics for a Two-Layer Neural Network

Goal: understand the complete mathematics of a small neural network that can
learn XOR. This is the model that will later be implemented with NumPy from
scratch.

---

## Q1 — The XOR problem

XOR means _exclusive OR_: the output is `1` when exactly one input is `1`.

| x₁ | x₂ | y |
| ----- | ----- | --- |
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

```text
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = [0, 1, 1, 0]
```

```text
X.shape = (4, 2)    # four examples, two features per example
y.shape = (4,)      # one class label per example
```

### Why one neuron cannot learn XOR

A single neuron creates one straight decision boundary. The positive examples,
`[0, 1]` and `[1, 0]`, lie on opposite corners; the negative examples,
`[0, 0]` and `[1, 1]`, lie on the other corners. No one straight line can
separate those two groups correctly.

```text
Project 01: 2 inputs → one neuron → output      cannot learn XOR
Project 02: 2 inputs → hidden layer → output    can learn XOR
```

---

## Q2 — Network architecture

```text
Input layer:  2 features
Hidden layer: 4 neurons with ReLU activation
Output layer: 2 neurons with softmax activation

X → Dense layer 1 → ReLU → Dense layer 2 → Softmax → P
```

The output layer has two neurons because this is a two-class problem:

```text
output 0 = score or probability for class 0
output 1 = score or probability for class 1
```

---

## Q3 — One-hot target matrix

Softmax returns one probability for every class, so each target must also have
one value for every class.

```text
class 0 → [1, 0]
class 1 → [0, 1]
```

```text
Y = [[1, 0],
     [0, 1],
     [0, 1],
     [1, 0]]

Y.shape = (4, 2)
```

For input `[0, 1]`, the correct target is `[0, 1]`. Therefore the model
should make its class-0 probability approach 0 and its class-1 probability
approach 1.

---

## Q4 — Parameters and shapes

| Parameter | Meaning                  | Why it has that shape                        | Shape    |
| --------- | ------------------------ | -------------------------------------------- | -------- |
| W₁        | Input-to-hidden weights  | 2 input features connect to 4 hidden neurons | (2, 4)   |
| b₁        | Hidden-layer bias        | Each of the 4 hidden neurons needs one bias  | (1, 4)   |
| W₂        | Hidden-to-output weights | 4 hidden neurons connect to 2 output neurons | (4, 2)   |
| b₂        | Output-layer bias        | Each of the 2 output neurons needs one bias  | (1, 2)   |

### Shape visualisation

```text
                 W1: (2, 4)                     W2: (4, 2)
      ┌────────────────────────┐      ┌────────────────────────┐
      │  Every input feature   │      │ Every hidden neuron    │
      │  connects to every     │      │ connects to every      │
      │  hidden neuron.        │      │ output neuron.         │
      └────────────────────────┘      └────────────────────────┘

Input layer                  Hidden layer                  Output layer
2 features                   4 neurons                    2 class scores

 x1 ───────────────┐          h1 ───────────────┐          class 0
 x2 ───────────────┼── W1 ──► h2 ───────────────┼── W2 ──► class 1
                   │          h3                │
                   └────────► h4 ───────────────┘

X:  (4, 2)  @  W1: (2, 4)  +  b1: (1, 4)  →  Z1: (4, 4)
A1: (4, 4)  @  W2: (4, 2)  +  b2: (1, 2)  →  Z2: (4, 2)
```

Read each weight shape as **(neurons entering the layer, neurons leaving the
layer)**. Biases have one value per neuron in the layer they are entering.

Biases have one row. NumPy broadcasts that row, adding the same bias values
to every training example.

---

## Q5 — Forward pass

The forward pass calculates:

```text
X → Z1 → A1 → Z2 → P
```

### Manual calculation for one example

Before using `@`, calculate one example by expanding every multiplication.
The following values are **illustrative only**; training will initialize its
own weights and learn better values.

```text
x = [0, 1]

W1 = [[ 0.5, -0.5,  1.0, -1.0],
      [ 1.0,  0.5, -1.0, -0.5]]

b1 = [0.1, 0.1, 0.1, 0.1]
```

Calculate the weighted sum for each of the four hidden neurons:

```text
z = (input_1 × weight_1) + (input_2 × weight_2) + bias

z: Represents the raw weighted sum (pre-activation) before any activation function like ReLU is applied.
The first 1: Refers to Layer 1 (the hidden layer).
The second 1: Refers to Neuron 1 (the first of the 4 hidden neurons).

z1_1 = (0 × 0.5)  + (1 × 1.0)  + 0.1 =  1.1
z1_2 = (0 × -0.5) + (1 × 0.5)  + 0.1 =  0.6
z1_3 = (0 × 1.0)  + (1 × -1.0) + 0.1 = -0.9
z1_4 = (0 × -1.0) + (1 × -0.5) + 0.1 = -0.4

Z1 = [1.1, 0.6, -0.9, -0.4]
```

Apply ReLU to every hidden value:

```text
A1 = ReLU(Z1)
   = [max(0, 1.1), max(0, 0.6), max(0, -0.9), max(0, -0.4)]
   = [1.1, 0.6, 0.0, 0.0]
```

Use these illustrative output-layer parameters:

```text
W2 = [[-0.5,  0.5],
      [-1.0,  1.0],
      [ 0.5, -0.5],
      [-0.5,  0.5]]

b2 = [0.1, -0.1]
```

Calculate the two output logits by hand:

```text
z2_class_0 = (1.1 × -0.5) + (0.6 × -1.0) + (0.0 × 0.5) + (0.0 × -0.5) + 0.1
           = -1.05

z2_class_1 = (1.1 × 0.5) + (0.6 × 1.0) + (0.0 × -0.5) + (0.0 × 0.5) - 0.1
           = 1.05

Z2 = [-1.05, 1.05]
```

What does exp(-1.05) actually mean?
exp(x) means Euler's number e ≈ 2.71828 raised to the power of x:
```text
exp(-1.05) = e^(-1.05) = (2.71828)^(-1.05) ≈ 0.349937
```

A negative exponent means taking the reciprocal:
```text
e^(-1.05) = 1 / (e^1.05) = 1 / 2.85765 ≈ 0.349937
```

Why do we apply exp() to raw scores?
Before softmax, the two output neurons produced raw logits:

Class 0 logit: z0 = -1.05
Class 1 logit: z1 = +1.05
Raw logits cannot be used as probabilities directly for two major reasons:

Probabilities cannot be negative: z0 is -1.05. A negative probability like -105% makes no mathematical sense. The exponential function e^x is always strictly positive for any number:

e^(-10) = 0.000045 (tiny, but positive)
e^(-1.05) = 0.3499 (positive)
e^(0) = 1.0
e^(1.05) = 2.8577 (positive) Applying exp() guarantees every score becomes a positive number.
It magnifies differences (winner takes more): A score of +1.05 is higher than -1.05. Exponentiating turns -1.05 into 0.35 and +1.05 into 2.86 — making the difference between the two classes much more pronounced.

Finally, apply softmax:

```text
P(class 0) = exp(-1.05) / (exp(-1.05) + exp(1.05)) ≈ 0.109
P(class 1) = exp(1.05)  / (exp(-1.05) + exp(1.05)) ≈ 0.891

P = [0.109, 0.891]
```

The input `[0, 1]` belongs to class `1`, so this example correctly gives the
larger probability to class `1`.

### The same calculation with `@`

The expanded calculations above are exactly what matrix multiplication does:

```text
Z1 = x @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
P  = softmax(Z2)
```

For a batch, replace the one input row `x` with the complete input matrix `X`.
The same equations calculate every example at once.

### 1. Hidden-layer weighted sum

For one input $x = [x_1, x_2]$, hidden neuron $j$ calculates:

```math
z_{1,j} = x_1 W_{1,1j} + x_2 W_{1,2j} + b_{1,j}
```

For all examples and all hidden neurons:

```math
Z_1 = XW_1 + b_1
```

This represents two distinct operations:
1. **Matrix Multiplication ($X W_1$)**: Multiplying the $(4, 2)$ input matrix by the $(2, 4)$ weight matrix to compute pre-activations for all 4 examples across all 4 neurons simultaneously.
2. **Broadcasting Addition ($+ b_1$)**: Adding the $(1, 4)$ bias vector to every row of the resulting matrix.

#### Step-by-step matrix calculation:

```math
X = \begin{bmatrix}
0 & 0 \\
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}, \qquad
W_1 = \begin{bmatrix}
0.5 & -0.5 & 1.0 & -1.0 \\
1.0 & 0.5 & -1.0 & -0.5
\end{bmatrix}, \qquad
b_1 = \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}
```

**Step A — Matrix Multiplication ($X W_1$):**

Each cell $(i, j)$ is the dot product of Row $i$ of $X$ and Column $j$ of $W_1$:

```math
X W_1 = \begin{bmatrix}
(0 \times 0.5 + 0 \times 1.0) & (0 \times -0.5 + 0 \times 0.5) & (0 \times 1.0 + 0 \times -1.0) & (0 \times -1.0 + 0 \times -0.5) \\
(0 \times 0.5 + 1 \times 1.0) & (0 \times -0.5 + 1 \times 0.5) & (0 \times 1.0 + 1 \times -1.0) & (0 \times -1.0 + 1 \times -0.5) \\
(1 \times 0.5 + 0 \times 1.0) & (1 \times -0.5 + 0 \times 0.5) & (1 \times 1.0 + 0 \times -1.0) & (1 \times -1.0 + 0 \times -0.5) \\
(1 \times 0.5 + 1 \times 1.0) & (1 \times -0.5 + 1 \times 0.5) & (1 \times 1.0 + 1 \times -1.0) & (1 \times -1.0 + 1 \times -0.5)
\end{bmatrix}
```

Evaluating each cell:

```math
X W_1 = \begin{bmatrix}
0.0 &  0.0 &  0.0 &  0.0 \\
1.0 &  0.5 & -1.0 & -0.5 \\
0.5 & -0.5 &  1.0 & -1.0 \\
1.5 &  0.0 &  0.0 & -1.5
\end{bmatrix}
```

**Step B — Broadcasting Addition ($+ b_1$):**

The bias vector $b_1 = [0.1, 0.1, 0.1, 0.1]$ has a single row. NumPy broadcasts that row down all 4 rows, adding each neuron's bias to its respective column:

```math
Z_1 = \begin{bmatrix}
0.0 &  0.0 &  0.0 &  0.0 \\
1.0 &  0.5 & -1.0 & -0.5 \\
0.5 & -0.5 &  1.0 & -1.0 \\
1.5 &  0.0 &  0.0 & -1.5
\end{bmatrix}
+
\begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.1 & 0.1
\end{bmatrix}
=
\begin{bmatrix}
0.1 &  0.1 &  0.1 &  0.1 \\
1.1 &  0.6 & -0.9 & -0.4 \\
0.6 & -0.4 &  1.1 & -0.9 \\
1.6 &  0.1 &  0.1 & -1.4
\end{bmatrix}
```

#### Row-by-row meaning of Z1:
- **Row 0** ($[0, 0]$): Both inputs are 0, so all weight products vanish, leaving only the biases $[0.1, 0.1, 0.1, 0.1]$.
- **Row 1** ($[0, 1]$): Exactly matches the single-example calculation $[1.1, 0.6, -0.9, -0.4]$.
- **Row 2** ($[1, 0]$): Evaluates $[0.6, -0.4, 1.1, -0.9]$.
- **Row 3** ($[1, 1]$): Combines both active features, evaluating $[1.6, 0.1, 0.1, -1.4]$.

```text
X:  (4, 2)
W1: (2, 4)
b1: (1, 4)

(4, 2) @ (2, 4) + (1, 4) → (4, 4)
Z1.shape = (4, 4)
```

### 2. ReLU activation

The Rectified Linear Unit (ReLU) activation function is defined element-by-element as:

```math
\mathrm{ReLU}(z) = \max(0, z)
```

For the hidden layer activation matrix:

```math
A_1 = \mathrm{ReLU}(Z_1)
```

#### Mathematical solving steps on matrix Z1:

We evaluate $\max(0, z)$ on each of the 16 elements of $Z_1$:

```math
Z_1 = \begin{bmatrix}
0.1 &  0.1 &  0.1 &  0.1 \\
1.1 &  0.6 & -0.9 & -0.4 \\
0.6 & -0.4 &  1.1 & -0.9 \\
1.6 &  0.1 &  0.1 & -1.4
\end{bmatrix}
```

Substitute every pre-activation into $\max(0, z)$:

```math
A_1 = \begin{bmatrix}
\max(0, 0.1) & \max(0, 0.1) & \max(0, 0.1) & \max(0, 0.1) \\
\max(0, 1.1) & \max(0, 0.6) & \max(0, -0.9) & \max(0, -0.4) \\
\max(0, 0.6) & \max(0, -0.4) & \max(0, 1.1) & \max(0, -0.9) \\
\max(0, 1.6) & \max(0, 0.1) & \max(0, 0.1) & \max(0, -1.4)
\end{bmatrix}
```

#### Element-by-element evaluation:
- **Row 0** ($[0, 0]$):
  - $\max(0, 0.1) = 0.1$ (neuron 1 fires)
  - $\max(0, 0.1) = 0.1$ (neuron 2 fires)
  - $\max(0, 0.1) = 0.1$ (neuron 3 fires)
  - $\max(0, 0.1) = 0.1$ (neuron 4 fires)
  - Result: $[0.1, 0.1, 0.1, 0.1]$
- **Row 1** ($[0, 1]$):
  - $\max(0, 1.1) = 1.1$ (neuron 1 fires)
  - $\max(0, 0.6) = 0.6$ (neuron 2 fires)
  - $\max(0, -0.9) = 0.0$ (neuron 3 is clamped to 0)
  - $\max(0, -0.4) = 0.0$ (neuron 4 is clamped to 0)
  - Result: $[1.1, 0.6, 0.0, 0.0]$
- **Row 2** ($[1, 0]$):
  - $\max(0, 0.6) = 0.6$
  - $\max(0, -0.4) = 0.0$ (neuron 2 clamped to 0)
  - $\max(0, 1.1) = 1.1$
  - $\max(0, -0.9) = 0.0$ (neuron 4 clamped to 0)
  - Result: $[0.6, 0.0, 1.1, 0.0]$
- **Row 3** ($[1, 1]$):
  - $\max(0, 1.6) = 1.6$
  - $\max(0, 0.1) = 0.1$
  - $\max(0, 0.1) = 0.1$
  - $\max(0, -1.4) = 0.0$ (neuron 4 clamped to 0)
  - Result: $[1.6, 0.1, 0.1, 0.0]$

Yielding the activation matrix $A_1$:

```math
A_1 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}
```

```text
A1.shape = (4, 4)
```

#### Why ReLU is mathematically required:
1. **Clamping negative values acts as a feature gate**: Only neurons that detect relevant patterns fire ($> 0$). Neurons with negative responses are silenced ($= 0$).
2. **Breaks linearity**: Without ReLU, $A_1 = Z_1$. Then $Z_2 = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2)$. Because $W_1 W_2$ is just another $(2, 2)$ matrix, two linear layers collapse into one linear layer, which mathematically cannot solve XOR.

---

### 3. Output-layer weighted sum

The output layer calculates two raw class scores, called logits:

```math
Z_2 = A_1W_2 + b_2
```

This represents two distinct operations:
1. **Matrix Multiplication ($A_1 W_2$)**: Multiplying the $(4, 4)$ hidden activation matrix by the $(4, 2)$ output weight matrix to compute raw scores for all 4 examples across both classes simultaneously.
2. **Broadcasting Addition ($+ b_2$)**: Adding the $(1, 2)$ output bias vector to every row of the resulting matrix.

#### Step-by-step matrix calculation:

```math
A_1 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}, \qquad
W_2 = \begin{bmatrix}
-0.5 &  0.5 \\
-1.0 &  1.0 \\
 0.5 & -0.5 \\
-0.5 &  0.5
\end{bmatrix}, \qquad
b_2 = \begin{bmatrix} 0.1 & -0.1 \end{bmatrix}
```

**Step A — Matrix Multiplication ($A_1 W_2$):**

Each cell $(i, c)$ is the dot product of Row $i$ of $A_1$ (hidden activations for Example $i$) and Column $c$ of $W_2$ (weights connecting to Class $c$):

```math
A_1 W_2 = \begin{bmatrix}
(0.1 \times -0.5 + 0.1 \times -1.0 + 0.1 \times 0.5 + 0.1 \times -0.5) & (0.1 \times 0.5 + 0.1 \times 1.0 + 0.1 \times -0.5 + 0.1 \times 0.5) \\
(1.1 \times -0.5 + 0.6 \times -1.0 + 0.0 \times 0.5 + 0.0 \times -0.5) & (1.1 \times 0.5 + 0.6 \times 1.0 + 0.0 \times -0.5 + 0.0 \times 0.5) \\
(0.6 \times -0.5 + 0.0 \times -1.0 + 1.1 \times 0.5 + 0.0 \times -0.5) & (0.6 \times 0.5 + 0.0 \times 1.0 + 1.1 \times -0.5 + 0.0 \times 0.5) \\
(1.6 \times -0.5 + 0.1 \times -1.0 + 0.1 \times 0.5 + 0.0 \times -0.5) & (1.6 \times 0.5 + 0.1 \times 1.0 + 0.1 \times -0.5 + 0.0 \times 0.5)
\end{bmatrix}
```

Evaluating each cell:

```math
A_1 W_2 = \begin{bmatrix}
(-0.05 - 0.10 + 0.05 - 0.05) & (0.05 + 0.10 - 0.05 + 0.05) \\
(-0.55 - 0.60 + 0.00 - 0.00) & (0.55 + 0.60 - 0.00 + 0.00) \\
(-0.30 - 0.00 + 0.55 - 0.00) & (0.30 + 0.00 - 0.55 + 0.00) \\
(-0.80 - 0.10 + 0.05 - 0.00) & (0.80 + 0.10 - 0.05 + 0.00)
\end{bmatrix}
=
\begin{bmatrix}
-0.15 &  0.15 \\
-1.15 &  1.15 \\
 0.25 & -0.25 \\
-0.85 &  0.85
\end{bmatrix}
```

**Step B — Broadcasting Addition ($+ b_2$):**

The bias vector $b_2 = [0.1, -0.1]$ has a single row. NumPy broadcasts that row down all 4 rows, adding the Class 0 bias ($+0.1$) to column 0 and Class 1 bias ($-0.1$) to column 1:

```math
Z_2 = \begin{bmatrix}
-0.15 &  0.15 \\
-1.15 &  1.15 \\
 0.25 & -0.25 \\
-0.85 &  0.85
\end{bmatrix}
+
\begin{bmatrix}
0.1 & -0.1 \\
0.1 & -0.1 \\
0.1 & -0.1 \\
0.1 & -0.1
\end{bmatrix}
=
\begin{bmatrix}
(-0.15 + 0.1) & (0.15 - 0.1) \\
(-1.15 + 0.1) & (1.15 - 0.1) \\
(0.25 + 0.1)  & (-0.25 - 0.1) \\
(-0.85 + 0.1) & (0.85 - 0.1)
\end{bmatrix}
=
\begin{bmatrix}
-0.05 &  0.05 \\
-1.05 &  1.05 \\
 0.35 & -0.35 \\
-0.75 &  0.75
\end{bmatrix}
```

#### Detailed row-by-row equations:
- **Row 0 ($[0, 0]$)**:
  - $z_{2, 0} = (0.1 \times -0.5) + (0.1 \times -1.0) + (0.1 \times 0.5) + (0.1 \times -0.5) + 0.1 = -0.15 + 0.1 = -0.05$
  - $z_{2, 1} = (0.1 \times 0.5) + (0.1 \times 1.0) + (0.1 \times -0.5) + (0.1 \times 0.5) - 0.1 = 0.15 - 0.1 = 0.05$
  - Logits: $[-0.05, 0.05]$ (nearly equal, model is undecided).

- **Row 1 ($[0, 1]$)**:
  - $z_{2, 0} = (1.1 \times -0.5) + (0.6 \times -1.0) + (0.0 \times 0.5) + (0.0 \times -0.5) + 0.1 = -1.15 + 0.1 = -1.05$
  - $z_{2, 1} = (1.1 \times 0.5) + (0.6 \times 1.0) + (0.0 \times -0.5) + (0.0 \times 0.5) - 0.1 = 1.15 - 0.1 = 1.05$
  - Logits: $[-1.05, 1.05]$ (class 1 is higher, exactly matches single example).

- **Row 2 ($[1, 0]$)**:
  - $z_{2, 0} = (0.6 \times -0.5) + (0.0 \times -1.0) + (1.1 \times 0.5) + (0.0 \times -0.5) + 0.1 = 0.25 + 0.1 = 0.35$
  - $z_{2, 1} = (0.6 \times 0.5) + (0.0 \times 1.0) + (1.1 \times -0.5) + (0.0 \times 0.5) - 0.1 = -0.25 - 0.1 = -0.35$
  - Logits: $[0.35, -0.35]$ (class 0 is higher, wrong before training).

- **Row 3 ($[1, 1]$)**:
  - $z_{2, 0} = (1.6 \times -0.5) + (0.1 \times -1.0) + (0.1 \times 0.5) + (0.0 \times -0.5) + 0.1 = -0.85 + 0.1 = -0.75$
  - $z_{2, 1} = (1.6 \times 0.5) + (0.1 \times 1.0) + (0.1 \times -0.5) + (0.0 \times 0.5) - 0.1 = 0.85 - 0.1 = 0.75$
  - Logits: $[-0.75, 0.75]$ (class 1 is higher, wrong before training).

```text
A1: (4, 4)
W2: (4, 2)
b2: (1, 2)

(4, 4) @ (4, 2) + (1, 2) → (4, 2)
Z2.shape = (4, 2)
```

These are raw scores (logits), not probabilities. Softmax converts them to probabilities next.

### 4. Softmax activation

```math
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

```math
P = \mathrm{softmax}(Z_2)
```

For two classes:

```math
P_0 = \frac{e^{z_0}}{e^{z_0} + e^{z_1}}, \qquad
P_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}
```

```text
0 < P_i < 1
P(class 0) + P(class 1) = 1
P.shape = (4, 2)
```

The NumPy implementation will use stable softmax:

```math
\mathrm{softmax}(z_i) =
\frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
```

Subtracting the same maximum from every logit does not change probabilities,
but prevents very large exponential values.

#### Mathematical solving steps on matrix Z2:

We start with the $(4, 2)$ logits matrix computed by the output layer:

```math
Z_2 = \begin{bmatrix}
-0.05 &  0.05 \\
-1.05 &  1.05 \\
 0.35 & -0.35 \\
-0.75 &  0.75
\end{bmatrix}
```

For each row $i$, the softmax operation follows three arithmetic steps:
1. **Exponentiate each logit**: Calculate $e^{z_{i,0}}$ and $e^{z_{i,1}}$.
2. **Compute row sum (normalizer)**: $S_i = e^{z_{i,0}} + e^{z_{i,1}}$.
3. **Normalize by row sum**: $P_{i,0} = \frac{e^{z_{i,0}}}{S_i}$ and $P_{i,1} = \frac{e^{z_{i,1}}}{S_i}$.

#### Detailed row-by-row solving steps:

- **Row 0 ($[0, 0]$, Logits $[-0.05, 0.05]$)**:
  - Exponentiate logits:
```math
    e^{-0.05} \approx 0.951229, \qquad e^{0.05} \approx 1.051271
```
  - Row normalizer:
```math
    S_0 = 0.951229 + 1.051271 = 2.002501
```
  - Probabilities:
```math
    P_{0,0} = \frac{0.951229}{2.002501} \approx 0.475021, \qquad P_{0,1} = \frac{1.051271}{2.002501} \approx 0.524979
```
  - Verification: `0.475021 + 0.524979 = 1.000000` (model is nearly undecided, `P ≈ 50% / 50%`).

- **Row 1 ($[0, 1]$, Logits $[-1.05, 1.05]$)**:
  - Exponentiate logits:
```math
    e^{-1.05} \approx 0.349938, \qquad e^{1.05} \approx 2.857651
```
  - Row normalizer:
```math
    S_1 = 0.349938 + 2.857651 = 3.207589
```
  - Probabilities:
```math
    P_{1,0} = \frac{0.349938}{3.207589} \approx 0.109097, \qquad P_{1,1} = \frac{2.857651}{3.207589} \approx 0.890903
```
  - Verification: `0.109097 + 0.890903 = 1.000000` (strongly predicts Class 1 with 89.1%).

- **Row 2 ($[1, 0]$, Logits $[0.35, -0.35]$)**:
  - Exponentiate logits:
```math
    e^{0.35} \approx 1.419068, \qquad e^{-0.35} \approx 0.704688
```
  - Row normalizer:
```math
    S_2 = 1.419068 + 0.704688 = 2.123756
```
  - Probabilities:
```math
    P_{2,0} = \frac{1.419068}{2.123756} \approx 0.668188, \qquad P_{2,1} = \frac{0.704688}{2.123756} \approx 0.331812
```
  - Verification: `0.668188 + 0.331812 = 1.000000` (predicts Class 0 with 66.8%, incorrect for XOR before training).

- **Row 3 ($[1, 1]$, Logits $[-0.75, 0.75]$)**:
  - Exponentiate logits:
```math
    e^{-0.75} \approx 0.472367, \qquad e^{0.75} \approx 2.117000
```
  - Row normalizer:
```math
    S_3 = 0.472367 + 2.117000 = 2.589367
```
  - Probabilities:
```math
    P_{3,0} = \frac{0.472367}{2.589367} \approx 0.182426, \qquad P_{3,1} = \frac{2.117000}{2.589367} \approx 0.817574
```
  - Verification: `0.182426 + 0.817574 = 1.000000` (predicts Class 1 with 81.8%, incorrect for XOR before training).

#### Resulting Probability Matrix P (shape (4, 2)):

```math
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
```

### Complete batch forward pass calculation

Using the illustrative weights, here is the complete batch forward pass for all 4 XOR examples:

#### 1. Input matrix X and Target matrix Y

```math
X = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \qquad
Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
```

#### 2. Hidden pre-activation: Z1 = X W1 + b1

```math
Z_1 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 0.5 & -0.5 & 1.0 & -1.0 \\ 1.0 & 0.5 & -1.0 & -0.5 \end{bmatrix}
+ \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}
= \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & -0.9 & -0.4 \\
0.6 & -0.4 & 1.1 & -0.9 \\
1.6 & 0.1 & 0.1 & -1.4
\end{bmatrix}
```

#### 3. Hidden activation: A1 = ReLU(Z1)

Clamp negative values to 0:

```math
A_1 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}
```

#### 4. Output logits: Z2 = A1 W2 + b2

```math
Z_2 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}
\begin{bmatrix}
-0.5 & 0.5 \\
-1.0 & 1.0 \\
0.5 & -0.5 \\
-0.5 & 0.5
\end{bmatrix}
+ \begin{bmatrix} 0.1 & -0.1 \end{bmatrix}
= \begin{bmatrix}
-0.05 & 0.05 \\
-1.05 & 1.05 \\
0.35 & -0.35 \\
-0.75 & 0.75
\end{bmatrix}
```

#### 5. Output probabilities: P = softmax(Z2)

Normalize each row independently:

```math
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
```

Each row sums to 1.000000.

---

## Q6 — Softmax cross-entropy loss

Cross-entropy measures the divergence between the predicted probability distribution $P$ and the ground-truth one-hot distribution $Y$.

For a single example $i$, cross-entropy over $C = 2$ classes is:

```math
L_i = -\sum_{c=1}^{2} Y_{i,c} \log(P_{i,c})
```

Where:
- $\log$ is the natural logarithm ($\ln$, base $e$).
- $Y_{i}$ is a one-hot target vector ($[1, 0]$ for Class 0, or $[0, 1]$ for Class 1).
- Because exactly one target class has $Y_{i,c} = 1$ and the other has $Y_{i,c} = 0$, the sum collapses directly to:

```math
L_i = -\log(P_{i, \mathrm{true\_class}})
```

For a batch of $m$ examples ($m = 4$), the total loss is the arithmetic mean across all examples:

```math
L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{c=1}^{2} Y_{i,c}\log(P_{i,c}) = \frac{1}{m}\sum_{i=1}^{m} L_i
```

### Step-by-step matrix calculation

We start with the one-hot target matrix $Y$ and predicted probability matrix $P$:

```math
Y = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \qquad
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
```

#### Step A — Element-wise natural logarithm (log(P)):

```math
\log(P) = \begin{bmatrix}
\log(0.475021) & \log(0.524979) \\
\log(0.109097) & \log(0.890903) \\
\log(0.668188) & \log(0.331812) \\
\log(0.182426) & \log(0.817574)
\end{bmatrix}
=
\begin{bmatrix}
-0.744396 & -0.644397 \\
-2.215518 & -0.115520 \\
-0.403186 & -1.103187 \\
-1.701411 & -0.201414
\end{bmatrix}
```

#### Step B — One-hot masking (Y ⊙ log(P)):

Multiplying element-by-element with the one-hot target matrix zeroes out all false classes, keeping only the log-probability of the true label:

```math
Y \odot \log(P) = \begin{bmatrix}
1 \times -0.744396 & 0 \times -0.644397 \\
0 \times -2.215518 & 1 \times -0.115520 \\
0 \times -0.403186 & 1 \times -1.103187 \\
1 \times -1.701411 & 0 \times -0.201414
\end{bmatrix}
=
\begin{bmatrix}
-0.744396 &  0.000000 \\
 0.000000 & -0.115520 \\
 0.000000 & -1.103187 \\
-1.701411 &  0.000000
\end{bmatrix}
```

#### Step C — Individual example loss Li:

- **Example 0 ($[0, 0]$, True Class 0)**:
```math
  L_0 = -(1 \times \log(0.475021) + 0 \times \log(0.524979)) = -(-0.744396) = 0.744396
```
  *Intuition*: The model assigns roughly equal probabilities (47.5% vs 52.5%). The loss is moderate (approx 0.744), reflecting maximum uncertainty ($\ln(2) \approx 0.693$).

- **Example 1 ($[0, 1]$, True Class 1)**:
```math
  L_1 = -(0 \times \log(0.109097) + 1 \times \log(0.890903)) = -(-0.115520) = 0.115520
```
  *Intuition*: The model predicts Class 1 with high confidence (89.1%). The loss is very low (0.116), indicating an accurate prediction.

- **Example 2 ($[1, 0]$, True Class 1)**:
```math
  L_2 = -(0 \times \log(0.668188) + 1 \times \log(0.331812)) = -(-1.103187) = 1.103187
```
  *Intuition*: The model predicts Class 0 (66.8%) instead of Class 1 (33.2%). The loss exceeds 1.0 (1.103), penalizing the wrong decision.

- **Example 3 ($[1, 1]$, True Class 0)**:
```math
  L_3 = -(1 \times \log(0.182426) + 0 \times \log(0.817574)) = -(-1.701411) = 1.701411
```
  *Intuition*: The model is confidently wrong, assigning 81.8% to Class 1 and only 18.2% to the true Class 0. The loss is heavily penalized at 1.701.

### Summary Table of Individual Losses

| Example | Input x | Target y | Predicted Probabilities P | Target Prob P_true | Individual Loss Li = -log(P_true) | Prediction State |
|:---|:---|:---|:---|:---|:---|:---|
| **0** | [0, 0] | Class 0 ([1, 0]) | [0.475021, 0.524979] | 0.475021 | -log(0.475021) ≈ 0.744396 | Undecided |
| **1** | [0, 1] | Class 1 ([0, 1]) | [0.109097, 0.890903] | 0.890903 | -log(0.890903) ≈ 0.115520 | Accurate prediction |
| **2** | [1, 0] | Class 1 ([0, 1]) | [0.668188, 0.331812] | 0.331812 | -log(0.331812) ≈ 1.103187 | Predicts wrong class |
| **3** | [1, 1] | Class 0 ([1, 0]) | [0.182426, 0.817574] | 0.182426 | -log(0.182426) ≈ 1.701411 | Confidently wrong |

### Batch Mean Loss Calculation

We average the 4 individual losses:

```math
L = \frac{1}{m} \sum_{i=1}^{4} L_i = \frac{0.744396 + 0.115520 + 1.103187 + 1.701411}{4}
```

Sum of losses:

```math
\sum_{i=1}^{4} L_i = 3.664514
```

Batch mean loss:

```math
L = \frac{3.664514}{4} = 0.91612888
```

#### Numerical Stability in Implementation

If a model predicts $P_{i, \mathrm{true}} \to 0$, then $\log(0) \to -\infty$, producing a numerical `NaN` or `inf` error.
To prevent this, `cross_entropy(Y, P)` in `src/ann.py` clamps probabilities using an $\epsilon = 10^{-15}$ threshold:

```math
P_{\mathrm{clipped}} = \mathrm{clip}(P, 10^{-15}, 1 - 10^{-15})
```

---

## Q7 — Backpropagation derivations and equations

Backpropagation calculates the derivative of the loss with respect to every weight and bias by moving backward through the network using the chain rule.

### 1. Derivation of the Output Gradient: dL / dZ2

For a single example, let the logits be $z = [z_1, z_2]$, probabilities $p = [p_1, p_2]$, and one-hot target $y = [y_1, y_2]$.

The loss is:

```math
L = -y_1 \log(p_1) - y_2 \log(p_2) = -\sum_{k} y_k \log(p_k)
```

Where softmax is:

```math
p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
```

By the multivariate chain rule:

```math
\frac{\partial L}{\partial z_i} = \sum_{k} \frac{\partial L}{\partial p_k} \frac{\partial p_k}{\partial z_i}
```

#### Step A: Derivative of Cross-Entropy with respect to pk

```math
\frac{\partial L}{\partial p_k} = -\frac{y_k}{p_k}
```

#### Step B: Derivative of Softmax pk with respect to logit zi

Using the quotient rule:

**Case 1: When $k = i$**

```math
\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \sum e^{z_j} - e^{z_i} e^{z_i}}{\left(\sum e^{z_j}\right)^2} = p_i - p_i^2 = p_i (1 - p_i)
```

**Case 2: When $k \neq i$**

```math
\frac{\partial p_k}{\partial z_i} = \frac{0 \cdot \sum e^{z_j} - e^{z_k} e^{z_i}}{\left(\sum e^{z_j}\right)^2} = -p_k p_i
```

#### Step C: Chain rule combination

Substitute Step A and Step B into the chain rule:

```math
\begin{aligned}
\frac{\partial L}{\partial z_i} &= \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_i} + \sum_{k \neq i} \frac{\partial L}{\partial p_k}\frac{\partial p_k}{\partial z_i} \\
&= -\frac{y_i}{p_i} \cdot p_i(1 - p_i) + \sum_{k \neq i} \left(-\frac{y_k}{p_k}\right) \cdot (-p_k p_i) \\
&= -y_i(1 - p_i) + \sum_{k \neq i} y_k p_i \\
&= -y_i + y_i p_i + p_i \sum_{k \neq i} y_k \\
&= -y_i + p_i \left(y_i + \sum_{k \neq i} y_k\right)
\end{aligned}
```

Because $y$ is one-hot, the sum over all classes is identically 1:

```math
\sum_k y_k = y_i + \sum_{k \neq i} y_k = 1
```

Therefore:

```math
\frac{\partial L}{\partial z_i} = p_i - y_i
```

For the entire batch of $m$ examples with mean loss $L = \frac{1}{m} \sum L_i$:

```math
dZ_2 = \frac{P - Y}{m}
```

### 2. Output-layer parameter gradients: dW2 and db2

Recall that $Z_2 = A_1 W_2 + b_2$. Applying the chain rule:

```math
dW_2 = A_1^T dZ_2
```

```math
db_2 = \sum_{i=1}^{m} dZ_{2, i}
```

**Dimension check:**
- $A_1^T$ has shape $(4, 4)$
- $dZ_2$ has shape $(4, 2)$
- `(4, 4) @ (4, 2) -> (4, 2)` (matches `W2` shape)
- $db_2$ sums over rows of $dZ_2 \to (1, 2)$ (matches $b_2$ shape)

### 3. Propagating gradients to the hidden layer: dA1 and dZ1

#### Step A: Derivation of activation gradient dA1

Hidden neuron $j$'s activation $a_{1, j}$ contributes to both output logits $z_{2, 0}$ and $z_{2, 1}$ through weights $W_{2, j, 0}$ and $W_{2, j, 1}$:

```math
z_{2, c} = \sum_{k=1}^{4} a_{1, k} W_{2, k, c} + b_{2, c}
```

Taking the partial derivative of output logit $z_{2, c}$ with respect to hidden activation $a_{1, j}$:

```math
\frac{\partial z_{2, c}}{\partial a_{1, j}} = W_{2, j, c}
```

By the multivariate chain rule, the gradient of the loss with respect to hidden activation $a_{1, j}$ is the sum of contributions from both output classes:

```math
\frac{\partial L}{\partial a_{1, j}} = \sum_{c=0}^{1} \frac{\partial L}{\partial z_{2, c}} \frac{\partial z_{2, c}}{\partial a_{1, j}} = dz_{2, 0} W_{2, j, 0} + dz_{2, 1} W_{2, j, 1}
```

In matrix form across all $m = 4$ examples and all 4 hidden neurons:

```math
dA_1 = dZ_2 W_2^T
```

**Dimension check:**
- $dZ_2$ has shape $(4, 2)$
- $W_2^T$ has shape $(2, 4)$
- `(4, 2) @ (2, 4) -> (4, 4)` (matches `A1` shape (4, 4))

#### Step B: Passing through ReLU activation derivative to get dZ1

Recall that $a_{1, j} = \mathrm{ReLU}(z_{1, j}) = \max(0, z_{1, j})$. The derivative of ReLU is:

```math
\frac{\partial a_{1, j}}{\partial z_{1, j}} = \mathrm{ReLU}'(z_{1, j}) = \begin{cases} 1 & z_{1, j} > 0 \\ 0 & z_{1, j} \leq 0 \end{cases}
```

Applying the chain rule:

```math
\frac{\partial L}{\partial z_{1, j}} = \frac{\partial L}{\partial a_{1, j}} \cdot \frac{\partial a_{1, j}}{\partial z_{1, j}} = \begin{cases} da_{1, j} & z_{1, j} > 0 \\ 0 & z_{1, j} \leq 0 \end{cases}
```

In matrix form across all examples and neurons, this is an element-wise product ($\odot$) with the boolean mask $(Z_1 > 0)$:

```math
dZ_1 = dA_1 \odot (Z_1 > 0)
```

**Dimension check:**
- $dA_1$ has shape $(4, 4)$
- $(Z_1 > 0)$ has shape $(4, 4)$
- `(4, 4) ⊙ (4, 4) -> (4, 4)` (matches $Z_1$ shape $(4, 4)$)

### 4. Hidden-layer parameter gradients: dW1 and db1

Recall that $Z_1 = X W_1 + b_1$. Applying the chain rule:

```math
dW_1 = X^T dZ_1
```

```math
db_1 = \sum_{i=1}^{m} dZ_{1, i}
```

**Dimension check:**
- $X^T$ has shape $(2, 4)$
- $dZ_1$ has shape $(4, 4)$
- `(2, 4) @ (4, 4) -> (2, 4)` (matches `W1` shape)
- `db1` sums over rows of `dZ1 -> (1, 4)` (matches `b1` shape)


---

### Manual numerical calculation of gradients

Using our illustrative weights on the 4 XOR examples ($m = 4$):

#### 1. Output error gradient: dZ2 = (P - Y) / m

Compute the difference matrix $P - Y$:

```math
P - Y = \begin{bmatrix}
0.475021 - 1 & 0.524979 - 0 \\
0.109097 - 0 & 0.890903 - 1 \\
0.668188 - 0 & 0.331812 - 1 \\
0.182426 - 1 & 0.817574 - 0
\end{bmatrix}
=
\begin{bmatrix}
-0.524979 &  0.524979 \\
 0.109097 & -0.109097 \\
 0.668188 & -0.668188 \\
-0.817574 &  0.817574
\end{bmatrix}
```

Divide by batch size $m = 4$:

```math
dZ_2 = \frac{1}{4} (P - Y) = \begin{bmatrix}
-0.524979 / 4 &  0.524979 / 4 \\
 0.109097 / 4 & -0.109097 / 4 \\
 0.668188 / 4 & -0.668188 / 4 \\
-0.817574 / 4 &  0.817574 / 4
\end{bmatrix}
=
\begin{bmatrix}
-0.131245 &  0.131245 \\
 0.027274 & -0.027274 \\
 0.167047 & -0.167047 \\
-0.204394 &  0.204394
\end{bmatrix}
```

#### 2. Output parameter gradients: dW2 = A1^T dZ2 and db2 = sum(dZ2)

Transpose $A_1$ (shape $(4, 4)$):

```math
A_1^T = \begin{bmatrix}
0.1 & 1.1 & 0.6 & 1.6 \\
0.1 & 0.6 & 0.0 & 0.1 \\
0.1 & 0.0 & 1.1 & 0.1 \\
0.1 & 0.0 & 0.0 & 0.0
\end{bmatrix}
```

Compute the matrix product $dW_2 = A_1^T dZ_2$ of shape $(4, 2)$:
- **Neuron 1 ($j = 0$, Class 0)**:
  `0.1 * (-0.131245) + 1.1 * 0.027274 + 0.6 * 0.167047 + 1.6 * (-0.204394) = -0.209924`
- **Neuron 1 ($j = 0$, Class 1)**:
  `0.1 * 0.131245 + 1.1 * (-0.027274) + 0.6 * (-0.167047) + 1.6 * 0.204394 = +0.209924`
- **Neuron 2 ($j = 1$, Class 0)**:
  `0.1 * (-0.131245) + 0.6 * 0.027274 + 0.0 * 0.167047 + 0.1 * (-0.204394) = -0.017199`
- **Neuron 2 ($j = 1$, Class 1)**:
  `0.1 * 0.131245 + 0.6 * (-0.027274) + 0.0 * (-0.167047) + 0.1 * 0.204394 = +0.017199`
- **Neuron 3 ($j = 2$, Class 0)**:
  `0.1 * (-0.131245) + 0.0 * 0.027274 + 1.1 * 0.167047 + 0.1 * (-0.204394) = +0.150188`
- **Neuron 3 ($j = 2$, Class 1)**:
  `0.1 * 0.131245 + 0.0 * (-0.027274) + 1.1 * (-0.167047) + 0.1 * 0.204394 = -0.150188`
- **Neuron 4 ($j = 3$, Class 0)**:
  `0.1 * (-0.131245) + 0.0 + 0.0 + 0.0 = -0.013124`
- **Neuron 4 ($j = 3$, Class 1)**:
  `0.1 * 0.131245 + 0.0 + 0.0 + 0.0 = +0.013124`

Yielding:

```math
dW_2 = \begin{bmatrix}
-0.209924 &  0.209924 \\
-0.017199 &  0.017199 \\
 0.150188 & -0.150188 \\
-0.013124 &  0.013124
\end{bmatrix}
```

Sum down columns of $dZ_2$ to obtain $db_2$:
- Class 0: `-0.131245 + 0.027274 + 0.167047 - 0.204394 = -0.141317`
- Class 1: `0.131245 - 0.027274 - 0.167047 + 0.204394 = +0.141317`

```math
db_2 = \begin{bmatrix} -0.141317 & 0.141317 \end{bmatrix}
```

#### 3. Hidden activation gradient: dA1 = dZ2 W2^T

Transpose $W_2$ (shape $(2, 4)$):

```math
W_2^T = \begin{bmatrix}
-0.5 & -1.0 &  0.5 & -0.5 \\
 0.5 &  1.0 & -0.5 &  0.5
\end{bmatrix}
```

Multiply $dZ_2$ of shape $(4, 2)$ by $W_2^T$ of shape $(2, 4)$:

```math
dA_1 = \begin{bmatrix}
-0.131245 &  0.131245 \\
 0.027274 & -0.027274 \\
 0.167047 & -0.167047 \\
-0.204394 &  0.204394
\end{bmatrix}
\begin{bmatrix}
-0.5 & -1.0 &  0.5 & -0.5 \\
 0.5 &  1.0 & -0.5 &  0.5
\end{bmatrix}
```

Each cell $(i, j)$ is $dZ_{2, i, 0} \times W_{2, j, 0} + dZ_{2, i, 1} \times W_{2, j, 1}$:

```math
dA_1 = \begin{bmatrix}
 0.131245 &  0.262490 & -0.131245 &  0.131245 \\
-0.027274 & -0.054548 &  0.027274 & -0.027274 \\
-0.167047 & -0.334094 &  0.167047 & -0.167047 \\
 0.204394 &  0.408787 & -0.204394 &  0.204394
\end{bmatrix}
```

#### 4. ReLU backward mask: dZ1 = dA1 ⊙ (Z1 > 0)

Evaluate the boolean activation condition $(Z_1 > 0)$:

```math
(Z_1 > 0) = \begin{bmatrix}
0.1 > 0 & 0.1 > 0 & 0.1 > 0 & 0.1 > 0 \\
1.1 > 0 & 0.6 > 0 & -0.9 > 0 & -0.4 > 0 \\
0.6 > 0 & -0.4 > 0 & 1.1 > 0 & -0.9 > 0 \\
1.6 > 0 & 0.1 > 0 & 0.1 > 0 & -1.4 > 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
1 & 1 & 1 & 0
\end{bmatrix}
```

Element-wise product $dZ_1 = dA_1 \odot (Z_1 > 0)$:

```math
dZ_1 = \begin{bmatrix}
 0.131245 \times 1 &  0.262490 \times 1 & -0.131245 \times 1 &  0.131245 \times 1 \\
-0.027274 \times 1 & -0.054548 \times 1 &  0.027274 \times 0 & -0.027274 \times 0 \\
-0.167047 \times 1 & -0.334094 \times 0 &  0.167047 \times 1 & -0.167047 \times 0 \\
 0.204394 \times 1 &  0.408787 \times 1 & -0.204394 \times 1 &  0.204394 \times 0
\end{bmatrix}
```

Yielding:

```math
dZ_1 = \begin{bmatrix}
 0.131245 &  0.262490 & -0.131245 &  0.131245 \\
-0.027274 & -0.054548 &  0.000000 &  0.000000 \\
-0.167047 &  0.000000 &  0.167047 &  0.000000 \\
 0.204394 &  0.408787 & -0.204394 &  0.000000
\end{bmatrix}
```

#### 5. Hidden parameter gradients: dW1 = X^T dZ1 and db1 = sum(dZ1)

Transpose $X$ (shape $(2, 4)$):

```math
X^T = \begin{bmatrix}
0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1
\end{bmatrix}
```

Compute $dW_1 = X^T dZ_1$ of shape $(2, 4)$:
- **Row 0 ($x_1$ weights, sum of Rows 2 and 3 of $dZ_1$)**:
  - Neuron 1: `-0.167047 + 0.204394 = 0.037347`
  - Neuron 2: `0.000000 + 0.408787 = 0.408787`
  - Neuron 3: `0.167047 - 0.204394 = -0.037347`
  - Neuron 4: `0.000000 + 0.000000 = 0.000000`
- **Row 1 ($x_2$ weights, sum of Rows 1 and 3 of $dZ_1$)**:
  - Neuron 1: `-0.027274 + 0.204394 = 0.177119`
  - Neuron 2: `-0.054548 + 0.408787 = 0.354239`
  - Neuron 3: `0.000000 - 0.204394 = -0.204394`
  - Neuron 4: `0.000000 + 0.000000 = 0.000000`

Yielding:

```math
dW_1 = \begin{bmatrix}
0.037347 & 0.408787 & -0.037347 & 0.000000 \\
0.177119 & 0.354239 & -0.204394 & 0.000000
\end{bmatrix}
```

Sum down columns of $dZ_1$ to obtain $db_1$:
- Neuron 1: `0.131245 - 0.027274 - 0.167047 + 0.204394 = 0.141317`
- Neuron 2: `0.262490 - 0.054548 + 0.000000 + 0.408787 = 0.616728`
- Neuron 3: `-0.131245 + 0.000000 + 0.167047 - 0.204394 = -0.168591`
- Neuron 4: `0.131245 + 0.000000 + 0.000000 + 0.000000 = 0.131245`

```math
db_1 = \begin{bmatrix} 0.141317 & 0.616728 & -0.168591 & 0.131245 \end{bmatrix}
```

---

## Q8 — Gradient-descent update

Gradient descent updates each parameter in the opposite direction of its gradient, scaled by the learning rate $\eta$:

```math
\theta \leftarrow \theta - \eta \, \nabla_\theta L
```

For our two-layer neural network with learning rate $\eta = 0.1$:

```math
\begin{aligned}
W_1 &\leftarrow W_1 - \eta \, dW_1 \\
b_1 &\leftarrow b_1 - \eta \, db_1 \\
W_2 &\leftarrow W_2 - \eta \, dW_2 \\
b_2 &\leftarrow b_2 - \eta \, db_2
\end{aligned}
```

### Step-by-step arithmetic calculations (eta = 0.1)

#### 1. Hidden weight matrix update (W1 ← W1 - 0.1 × dW1):

```math
W_1 = \begin{bmatrix}
 0.5 & -0.5 &  1.0 & -1.0 \\
 1.0 &  0.5 & -1.0 & -0.5
\end{bmatrix}, \qquad
dW_1 = \begin{bmatrix}
0.037347 & 0.408787 & -0.037347 & 0.000000 \\
0.177119 & 0.354239 & -0.204394 & 0.000000
\end{bmatrix}
```

Compute scaled gradient `0.1 * dW1`:

```math
0.1 \times dW_1 = \begin{bmatrix}
0.003735 & 0.040879 & -0.003735 & 0.000000 \\
0.017712 & 0.035424 & -0.020439 & 0.000000
\end{bmatrix}
```

Subtract from $W_1$:

```math
W_{1, \mathrm{new}} = \begin{bmatrix}
0.5 - 0.003735 & -0.5 - 0.040879 & 1.0 - (-0.003735) & -1.0 - 0.000000 \\
1.0 - 0.017712 &  0.5 - 0.035424 & -1.0 - (-0.020439) & -0.5 - 0.000000
\end{bmatrix}
```

```math
W_{1, \mathrm{new}} = \begin{bmatrix}
0.496265 & -0.540879 &  1.003735 & -1.000000 \\
0.982288 &  0.464576 & -0.979561 & -0.500000
\end{bmatrix}
```

#### 2. Hidden bias vector update (b1 ← b1 - 0.1 × db1):

```math
b_1 = \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}, \qquad
db_1 = \begin{bmatrix} 0.141317 & 0.616728 & -0.168591 & 0.131245 \end{bmatrix}
```

Compute scaled gradient `0.1 * db1`:

```math
0.1 \times db_1 = \begin{bmatrix} 0.014132 & 0.061673 & -0.016859 & 0.013125 \end{bmatrix}
```

Subtract from $b_1$:

```math
b_{1, \mathrm{new}} = \begin{bmatrix}
0.1 - 0.014132 & 0.1 - 0.061673 & 0.1 - (-0.016859) & 0.1 - 0.013125
\end{bmatrix}
```

```math
b_{1, \mathrm{new}} = \begin{bmatrix} 0.085868 & 0.038327 & 0.116859 & 0.086875 \end{bmatrix}
```

#### 3. Output weight matrix update (W2 ← W2 - 0.1 × dW2):

```math
W_2 = \begin{bmatrix}
-0.5 &  0.5 \\
-1.0 &  1.0 \\
 0.5 & -0.5 \\
-0.5 &  0.5
\end{bmatrix}, \qquad
dW_2 = \begin{bmatrix}
-0.209924 &  0.209924 \\
-0.017199 &  0.017199 \\
 0.150188 & -0.150188 \\
-0.013124 &  0.013124
\end{bmatrix}
```

Compute scaled gradient `0.1 * dW2`:

```math
0.1 \times dW_2 = \begin{bmatrix}
-0.020992 &  0.020992 \\
-0.001720 &  0.001720 \\
 0.015019 & -0.015019 \\
-0.001312 &  0.001312
\end{bmatrix}
```

Subtract from $W_2$:

```math
W_{2, \mathrm{new}} = \begin{bmatrix}
-0.5 - (-0.020992) &  0.5 - 0.020992 \\
-1.0 - (-0.001720) &  1.0 - 0.001720 \\
 0.5 - 0.015019    & -0.5 - (-0.015019) \\
-0.5 - (-0.001312) &  0.5 - 0.001312
\end{bmatrix}
```

```math
W_{2, \mathrm{new}} = \begin{bmatrix}
-0.479008 &  0.479008 \\
-0.998280 &  0.998280 \\
 0.484981 & -0.484981 \\
-0.498688 &  0.498688
\end{bmatrix}
```

#### 4. Output bias vector update (b2 ← b2 - 0.1 × db2):

```math
b_2 = \begin{bmatrix} 0.1 & -0.1 \end{bmatrix}, \qquad
db_2 = \begin{bmatrix} -0.141317 & 0.141317 \end{bmatrix}
```

Compute scaled gradient `0.1 * db2`:

```math
0.1 \times db_2 = \begin{bmatrix} -0.014132 & 0.014132 \end{bmatrix}
```

Subtract from $b_2$:

```math
b_{2, \mathrm{new}} = \begin{bmatrix} 0.1 - (-0.014132) & -0.1 - 0.014132 \end{bmatrix}
= \begin{bmatrix} 0.114132 & -0.114132 \end{bmatrix}
```

### Impact of 1 Gradient Descent Step on Loss

Re-evaluating the batch forward pass with updated parameters:

| Metric | Before Step (Initial) | After 1 Step (eta = 0.1) | Change |
|:---|:---|:---|:---|
| **Cross-Entropy Loss** | 0.916129 | 0.838500 | **-0.077629 (Decreased)** |
| **Example 0 Loss ([0, 0])** | 0.744396 | 0.679140 | Improved |
| **Example 1 Loss ([0, 1])** | 0.115520 | 0.098412 | Improved |
| **Example 2 Loss ([1, 0])** | 1.103187 | 1.021505 | Improved |
| **Example 3 Loss ([1, 1])** | 1.701411 | 1.554942 | Improved |

Repeating this cycle for 2000 epochs with learning rate $\eta = 1.0$ converges loss from 0.916129 to 0.000478, reaching 100% classification accuracy on XOR.

Each training epoch executes:
```text
Forward Pass -> Loss Evaluation -> Backpropagation -> Parameter Update
```

---

## Q9 — Parameter initialization mathematics

### 1. The symmetry-breaking problem

If all weights are initialized to 0 ($W_1 = \mathbf{0}$, $W_2 = \mathbf{0}$, $b_1 = \mathbf{0}$, $b_2 = \mathbf{0}$):
- For any input $x$, hidden pre-activation is zero: $z_{1, j} = 0$.
- Every hidden neuron outputs the identical activation: $a_{1, 1} = a_{1, 2} = a_{1, 3} = a_{1, 4} = \mathrm{ReLU}(0) = 0$.
- In backpropagation, every hidden neuron receives the identical error gradient:
```math
  dW_{1, 1} = dW_{1, 2} = dW_{1, 3} = dW_{1, 4}
```
- After the gradient descent update, all weights remain identical. The network acts as if it has only **1 hidden neuron**, making it mathematically impossible to learn non-linear functions like XOR.
- Random initialization breaks this symmetry, allowing each neuron to specialize in detecting different feature combinations.

### 2. He (Kaiming) initialization for ReLU Layer 1

Because ReLU sets all negative values to zero ($\max(0, z)$), it deactivates approximately half of the neurons, cutting the variance of activations in half:

```math
\mathrm{Var}(a) = \frac{1}{2}\mathrm{Var}(z)
```

To prevent activations from vanishing toward zero as signals propagate deeper, He initialization compensates by doubling the variance:

```math
\mathrm{Var}(W_1) = \frac{2}{n_{\mathrm{in}}}
```

Taking the square root gives the required standard deviation $\sigma$:

```math
\sigma_{W_1} = \sqrt{\frac{2}{n_{\mathrm{in}}}}
```

#### Calculation for XOR Layer 1:
- Input feature count: $n_{\mathrm{in}} = 2$
- Hidden neuron count: $n_{\mathrm{hidden}} = 4$

```math
\sigma_{W_1} = \sqrt{\frac{2}{2}} = \sqrt{1.0} = 1.000000
```

Weights are drawn from $\mathcal{N}(0, 1.0)$ with shape $(2, 4)$:

```math
W_1 \sim \mathcal{N}(0, 1.0)
```

Using seed 42 in `ann.py`:

```math
W_1 = \begin{bmatrix}
 0.496714 & -0.138264 & 0.647689 & 1.523030 \\
-0.234153 & -0.234137 & 1.579213 & 0.767435
\end{bmatrix}
```

### 3. Xavier / Glorot initialization for Softmax Layer 2

For linear and softmax layers, signals do not experience the half-plane cut of ReLU. Xavier initialization maintains constant activation and gradient variances by scaling by the harmonic mean of fan-in and fan-out:

```math
\mathrm{Var}(W_2) = \frac{2}{n_{\mathrm{in}} + n_{\mathrm{out}}} = \frac{2}{n_{\mathrm{hidden}} + n_{\mathrm{out}}}
```

Taking the square root:

```math
\sigma_{W_2} = \sqrt{\frac{2}{n_{\mathrm{hidden}} + n_{\mathrm{out}}}}
```

#### Calculation for XOR Layer 2:
- Hidden neuron count: $n_{\mathrm{hidden}} = 4$
- Output class count: $n_{\mathrm{out}} = 2$

```math
\sigma_{W_2} = \sqrt{\frac{2}{4 + 2}} = \sqrt{\frac{2}{6}} = \sqrt{\frac{1}{3}} \approx 0.577350
```

Weights are drawn from $\mathcal{N}(0, 0.577350)$ with shape $(4, 2)$:

```math
W_2 \sim \mathcal{N}(0, 0.577350)
```

Using seed 42 in `ann.py`:

```math
W_2 = \begin{bmatrix}
-0.271051 &  0.313247 \\
-0.267554 & -0.268889 \\
 0.139697 & -1.104633 \\
-0.995882 & -0.324637
\end{bmatrix}
```

### 4. Bias initialization

Biases do not suffer from the symmetry-breaking problem because random weights already ensure each neuron receives distinct gradients. Biases are safely initialized to zeros:

```math
b_1 = \begin{bmatrix} 0.0 & 0.0 & 0.0 & 0.0 \end{bmatrix} \in \mathbf{R}^{1 \times 4}
```

```math
b_2 = \begin{bmatrix} 0.0 & 0.0 \end{bmatrix} \in \mathbf{R}^{1 \times 2}
```

---

## Q10 — Prediction and evaluation metrics

### 1. Decision rule: y_hat = argmax(P)

Softmax produces a probability distribution $P \in \mathbf{R}^{m \times 2}$ where $P_{i, 0} + P_{i, 1} = 1.0$.
The predicted class label $\hat{y}_i \in \{0, 1\}$ is the index of the class with higher probability:

```math
\hat{y}_i = \mathrm{argmax}_{c \in \{0, 1\}}(P_{i, c}) = \begin{cases} 0 & P_{i, 0} > P_{i, 1} \\ 1 & P_{i, 1} \geq P_{i, 0} \end{cases}
```

### 2. Accuracy metric calculation

Accuracy measures the proportion of samples classified correctly:

```math
\mathrm{Accuracy} = \frac{1}{m}\sum_{i=1}^{m} \mathbf{1}(\hat{y}_i == y_i)
```

Where $\mathbf{1}(\cdot)$ is the indicator function:
- $\mathbf{1}(\mathrm{true}) = 1$ (correct prediction)
- $\mathbf{1}(\mathrm{false}) = 0$ (incorrect prediction)

### 3. Concrete numerical calculation: Before Training vs After Training

#### A. Initial Untrained State (Using illustrative parameters)

```math
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}, \qquad
Y = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}
```

Ground-truth labels: $y = [0, 1, 1, 0]$

- **Example 0 ($[0, 0]$)**: $P = [0.475, 0.525] \implies \mathrm{argmax} \to \hat{y}_0 = 1$. Ground truth: $y_0 = 0$. Match: 0 (Incorrect).
- **Example 1 ($[0, 1]$)**: $P = [0.109, 0.891] \implies \mathrm{argmax} \to \hat{y}_1 = 1$. Ground truth: $y_1 = 1$. Match: 1 (Correct).
- **Example 2 ($[1, 0]$)**: $P = [0.668, 0.332] \implies \mathrm{argmax} \to \hat{y}_2 = 0$. Ground truth: $y_2 = 1$. Match: 0 (Incorrect).
- **Example 3 ($[1, 1]$)**: $P = [0.182, 0.818] \implies \mathrm{argmax} \to \hat{y}_3 = 1$. Ground truth: $y_3 = 0$. Match: 0 (Incorrect).

Predictions: $\hat{y} = [1, 1, 0, 1]$

```math
\mathrm{Accuracy}_{\mathrm{initial}} = \frac{0 + 1 + 0 + 0}{4} = \frac{1}{4} = 0.25 = 25.0\%
```

```math
\mathrm{Loss}_{\mathrm{initial}} = 0.916129
```

#### B. Final Trained State (After 2000 epochs, eta = 1.0)

Trained probability matrix $P$:

```math
P = \begin{bmatrix}
0.998674 & 0.001326 \\
0.000265 & 0.999735 \\
0.000164 & 0.999836 \\
0.999843 & 0.000157
\end{bmatrix}
```

Evaluating $\mathrm{argmax}$:
- **Example 0 ($[0, 0]$)**: $P = [0.9987, 0.0013] \implies \hat{y}_0 = 0$. Ground truth: 0. Match: 1 (Correct, 99.87% confidence).
- **Example 1 ($[0, 1]$)**: $P = [0.0003, 0.9997] \implies \hat{y}_1 = 1$. Ground truth: 1. Match: 1 (Correct, 99.97% confidence).
- **Example 2 ($[1, 0]$)**: $P = [0.0002, 0.9998] \implies \hat{y}_2 = 1$. Ground truth: 1. Match: 1 (Correct, 99.98% confidence).
- **Example 3 ($[1, 1]$)**: $P = [0.9998, 0.0002] \implies \hat{y}_3 = 0$. Ground truth: 0. Match: 1 (Correct, 99.98% confidence).

Predictions: $\hat{y} = [0, 1, 1, 0]$

```math
\mathrm{Accuracy}_{\mathrm{trained}} = \frac{1 + 1 + 1 + 1}{4} = \frac{4}{4} = 1.00 = 100.0\%
```

```math
\mathrm{Loss}_{\mathrm{trained}} = 0.000478
```

The non-linear XOR separation is solved completely.


---

## Summary of the complete mathematical pipeline

```text
0. Parameter Initialization (Symmetry Breaking)
   W1 ~ N(0, sigma = sqrt(2 / n_in))                     shape: (2, 4)  [He Normal]
   b1 = zeros((1, 4))                                    shape: (1, 4)
   W2 ~ N(0, sigma = sqrt(2 / (n_hidden + n_out)))       shape: (4, 2)  [Xavier Normal]
   b2 = zeros((1, 2))                                    shape: (1, 2)

1. Forward Pass
   Z1 = X @ W1 + b1                                      shape: (4, 2) @ (2, 4) + (1, 4) -> (4, 4)
   A1 = relu(Z1) = max(0, Z1)                            shape: (4, 4)
   Z2 = A1 @ W2 + b2                                     shape: (4, 4) @ (4, 2) + (1, 2) -> (4, 2)
   P  = softmax(Z2)                                      shape: (4, 2)

2. Loss Evaluation
   L  = -mean(sum(Y * log(clip(P, eps, 1-eps)), axis=1)) scalar (Cross-Entropy)

3. Backpropagation (Chain Rule)
   dZ2 = (P - Y) / m                                     shape: (4, 2)
   dW2 = A1.T @ dZ2                                      shape: (4, 4) @ (4, 2) -> (4, 2)
   db2 = sum(dZ2, axis=0, keepdims=True)                 shape: (1, 2)
   dA1 = dZ2 @ W2.T                                      shape: (4, 2) @ (2, 4) -> (4, 4)
   dZ1 = dA1 * (Z1 > 0)                                  shape: (4, 4)
   dW1 = X.T @ dZ1                                       shape: (2, 4) @ (4, 4) -> (2, 4)
   db1 = sum(dZ1, axis=0, keepdims=True)                 shape: (1, 4)

4. Parameter Update (Gradient Descent)
   W1 = W1 - lr * dW1                                    shape: (2, 4)
   b1 = b1 - lr * db1                                    shape: (1, 4)
   W2 = W2 - lr * dW2                                    shape: (4, 2)
   b2 = b2 - lr * db2                                    shape: (1, 2)

5. Prediction & Evaluation
   y_true      = argmax(Y, axis=1)                       shape: (4,)
   predictions = argmax(P, axis=1)                       shape: (4,)
   accuracy    = mean(predictions == y_true)             scalar (0.0 to 1.0)
```
