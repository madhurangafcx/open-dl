# Step 1 — Mathematics for a Two-Layer Neural Network

Goal: understand the complete mathematics of a small neural network that can
learn XOR. This is the model that will later be implemented with NumPy from
scratch.

---

## Q1 — The XOR problem

XOR means _exclusive OR_: the output is `1` when exactly one input is `1`.

| $x_1$ | $x_2$ | $y$ |
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
| $W_1$     | Input-to-hidden weights  | 2 input features connect to 4 hidden neurons | $(2, 4)$ |
| $b_1$     | Hidden-layer bias        | Each of the 4 hidden neurons needs one bias  | $(1, 4)$ |
| $W_2$     | Hidden-to-output weights | 4 hidden neurons connect to 2 output neurons | $(4, 2)$ |
| $b_2$     | Output-layer bias        | Each of the 2 output neurons needs one bias  | $(1, 2)$ |

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

$$
z_{1,j} = x_1 W_{1,1j} + x_2 W_{1,2j} + b_{1,j}
$$

For all examples and all hidden neurons:

$$
Z_1 = XW_1 + b_1
$$

This represents two distinct operations:
1. **Matrix Multiplication ($X W_1$)**: Multiplying the $(4, 2)$ input matrix by the $(2, 4)$ weight matrix to compute pre-activations for all 4 examples across all 4 neurons simultaneously.
2. **Broadcasting Addition ($+ b_1$)**: Adding the $(1, 4)$ bias vector to every row of the resulting matrix.

#### Step-by-step matrix calculation:

$$
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
$$

**Step A — Matrix Multiplication ($X W_1$):**

Each cell $(i, j)$ is the dot product of Row $i$ of $X$ and Column $j$ of $W_1$:

$$
X W_1 = \begin{bmatrix}
(0 \times 0.5 + 0 \times 1.0) & (0 \times -0.5 + 0 \times 0.5) & (0 \times 1.0 + 0 \times -1.0) & (0 \times -1.0 + 0 \times -0.5) \\
(0 \times 0.5 + 1 \times 1.0) & (0 \times -0.5 + 1 \times 0.5) & (0 \times 1.0 + 1 \times -1.0) & (0 \times -1.0 + 1 \times -0.5) \\
(1 \times 0.5 + 0 \times 1.0) & (1 \times -0.5 + 0 \times 0.5) & (1 \times 1.0 + 0 \times -1.0) & (1 \times -1.0 + 0 \times -0.5) \\
(1 \times 0.5 + 1 \times 1.0) & (1 \times -0.5 + 1 \times 0.5) & (1 \times 1.0 + 1 \times -1.0) & (1 \times -1.0 + 1 \times -0.5)
\end{bmatrix}
$$

Evaluating each cell:

$$
X W_1 = \begin{bmatrix}
0.0 &  0.0 &  0.0 &  0.0 \\
1.0 &  0.5 & -1.0 & -0.5 \\
0.5 & -0.5 &  1.0 & -1.0 \\
1.5 &  0.0 &  0.0 & -1.5
\end{bmatrix}
$$

**Step B — Broadcasting Addition ($+ b_1$):**

The bias vector $b_1 = [0.1, 0.1, 0.1, 0.1]$ has a single row. NumPy broadcasts that row down all 4 rows, adding each neuron's bias to its respective column:

$$
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
$$

#### Row-by-row meaning of $Z_1$:
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

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

For the hidden layer activation matrix:

$$
A_1 = \operatorname{ReLU}(Z_1)
$$

#### Mathematical solving steps on the matrix $Z_1$:

We evaluate $\max(0, z)$ on each of the 16 elements of $Z_1$:

$$
Z_1 = \begin{bmatrix}
0.1 &  0.1 &  0.1 &  0.1 \\
1.1 &  0.6 & -0.9 & -0.4 \\
0.6 & -0.4 &  1.1 & -0.9 \\
1.6 &  0.1 &  0.1 & -1.4
\end{bmatrix}
$$

Substitute every pre-activation into $\max(0, z)$:

$$
A_1 = \begin{bmatrix}
\max(0, 0.1) & \max(0, 0.1) & \max(0, 0.1) & \max(0, 0.1) \\
\max(0, 1.1) & \max(0, 0.6) & \max(0, -0.9) & \max(0, -0.4) \\
\max(0, 0.6) & \max(0, -0.4) & \max(0, 1.1) & \max(0, -0.9) \\
\max(0, 1.6) & \max(0, 0.1) & \max(0, 0.1) & \max(0, -1.4)
\end{bmatrix}
$$

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

$$
A_1 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}
$$

```text
A1.shape = (4, 4)
```

#### Why ReLU is mathematically required:
1. **Clamping negative values acts as a feature gate**: Only neurons that detect relevant patterns fire ($> 0$). Neurons with negative responses are silenced ($= 0$).
2. **Breaks linearity**: Without ReLU, $A_1 = Z_1$. Then $Z_2 = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2)$. Because $W_1 W_2$ is just another $(2, 2)$ matrix, two linear layers collapse into one linear layer, which mathematically cannot solve XOR.

---

### 3. Output-layer weighted sum

The output layer calculates two raw class scores, called logits:

$$
Z_2 = A_1W_2 + b_2
$$

This represents two distinct operations:
1. **Matrix Multiplication ($A_1 W_2$)**: Multiplying the $(4, 4)$ hidden activation matrix by the $(4, 2)$ output weight matrix to compute raw scores for all 4 examples across both classes simultaneously.
2. **Broadcasting Addition ($+ b_2$)**: Adding the $(1, 2)$ output bias vector to every row of the resulting matrix.

#### Step-by-step matrix calculation:

$$
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
$$

**Step A — Matrix Multiplication ($A_1 W_2$):**

Each cell $(i, c)$ is the dot product of Row $i$ of $A_1$ (hidden activations for Example $i$) and Column $c$ of $W_2$ (weights connecting to Class $c$):

$$
A_1 W_2 = \begin{bmatrix}
(0.1 \times -0.5 + 0.1 \times -1.0 + 0.1 \times 0.5 + 0.1 \times -0.5) & (0.1 \times 0.5 + 0.1 \times 1.0 + 0.1 \times -0.5 + 0.1 \times 0.5) \\
(1.1 \times -0.5 + 0.6 \times -1.0 + 0.0 \times 0.5 + 0.0 \times -0.5) & (1.1 \times 0.5 + 0.6 \times 1.0 + 0.0 \times -0.5 + 0.0 \times 0.5) \\
(0.6 \times -0.5 + 0.0 \times -1.0 + 1.1 \times 0.5 + 0.0 \times -0.5) & (0.6 \times 0.5 + 0.0 \times 1.0 + 1.1 \times -0.5 + 0.0 \times 0.5) \\
(1.6 \times -0.5 + 0.1 \times -1.0 + 0.1 \times 0.5 + 0.0 \times -0.5) & (1.6 \times 0.5 + 0.1 \times 1.0 + 0.1 \times -0.5 + 0.0 \times 0.5)
\end{bmatrix}
$$

Evaluating each cell:

$$
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
$$

**Step B — Broadcasting Addition ($+ b_2$):**

The bias vector $b_2 = [0.1, -0.1]$ has a single row. NumPy broadcasts that row down all 4 rows, adding the Class 0 bias ($+0.1$) to column 0 and Class 1 bias ($-0.1$) to column 1:

$$
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
$$

#### Detailed row-by-row equations:
- **Row 0 ($[0, 0]$)**:
  - $z_{2, \text{class } 0} = (0.1 \times -0.5) + (0.1 \times -1.0) + (0.1 \times 0.5) + (0.1 \times -0.5) + 0.1 = -0.15 + 0.1 = -0.05$
  - $z_{2, \text{class } 1} = (0.1 \times 0.5) + (0.1 \times 1.0) + (0.1 \times -0.5) + (0.1 \times 0.5) - 0.1 = 0.15 - 0.1 = 0.05$
  - Logits: $[-0.05, 0.05]$ (nearly equal, model is undecided).

- **Row 1 ($[0, 1]$)**:
  - $z_{2, \text{class } 0} = (1.1 \times -0.5) + (0.6 \times -1.0) + (0.0 \times 0.5) + (0.0 \times -0.5) + 0.1 = -1.15 + 0.1 = -1.05$
  - $z_{2, \text{class } 1} = (1.1 \times 0.5) + (0.6 \times 1.0) + (0.0 \times -0.5) + (0.0 \times 0.5) - 0.1 = 1.15 - 0.1 = 1.05$
  - Logits: $[-1.05, 1.05]$ (class 1 is higher, exactly matches single example).

- **Row 2 ($[1, 0]$)**:
  - $z_{2, \text{class } 0} = (0.6 \times -0.5) + (0.0 \times -1.0) + (1.1 \times 0.5) + (0.0 \times -0.5) + 0.1 = 0.25 + 0.1 = 0.35$
  - $z_{2, \text{class } 1} = (0.6 \times 0.5) + (0.0 \times 1.0) + (1.1 \times -0.5) + (0.0 \times 0.5) - 0.1 = -0.25 - 0.1 = -0.35$
  - Logits: $[0.35, -0.35]$ (class 0 is higher, wrong before training).

- **Row 3 ($[1, 1]$)**:
  - $z_{2, \text{class } 0} = (1.6 \times -0.5) + (0.1 \times -1.0) + (0.1 \times 0.5) + (0.0 \times -0.5) + 0.1 = -0.85 + 0.1 = -0.75$
  - $z_{2, \text{class } 1} = (1.6 \times 0.5) + (0.1 \times 1.0) + (0.1 \times -0.5) + (0.0 \times 0.5) - 0.1 = 0.85 - 0.1 = 0.75$
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

$$
\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$$
P = \operatorname{softmax}(Z_2)
$$

For two classes:

$$
P_0 = \frac{e^{z_0}}{e^{z_0} + e^{z_1}}, \qquad
P_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}
$$

```text
0 < P_i < 1
P(class 0) + P(class 1) = 1
P.shape = (4, 2)
```

The NumPy implementation will use stable softmax:

$$
\operatorname{softmax}(z_i) =
\frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

Subtracting the same maximum from every logit does not change probabilities,
but prevents very large exponential values.

#### Mathematical solving steps on the matrix $Z_2$:

We start with the $(4, 2)$ logits matrix computed by the output layer:

$$
Z_2 = \begin{bmatrix}
-0.05 &  0.05 \\
-1.05 &  1.05 \\
 0.35 & -0.35 \\
-0.75 &  0.75
\end{bmatrix}
$$

For each row $i$, the softmax operation follows three arithmetic steps:
1. **Exponentiate each logit**: Calculate $e^{z_{i,0}}$ and $e^{z_{i,1}}$.
2. **Compute row sum (normalizer)**: $S_i = e^{z_{i,0}} + e^{z_{i,1}}$.
3. **Normalize by row sum**: $P_{i,0} = \frac{e^{z_{i,0}}}{S_i}$ and $P_{i,1} = \frac{e^{z_{i,1}}}{S_i}$.

#### Detailed row-by-row solving steps:

- **Row 0 ($[0, 0]$, Logits $[-0.05, 0.05]$)**:
  - Exponentiate logits:
    $$
    e^{-0.05} \approx 0.951229, \qquad e^{0.05} \approx 1.051271
    $$
  - Row normalizer:
    $$
    S_0 = 0.951229 + 1.051271 = 2.002501
    $$
  - Probabilities:
    $$
    P_{0,0} = \frac{0.951229}{2.002501} \approx 0.475021, \qquad P_{0,1} = \frac{1.051271}{2.002501} \approx 0.524979
    $$
  - Verification: $0.475021 + 0.524979 = 1.000000$ (model is nearly undecided, $P \approx 50\%/50\%$).

- **Row 1 ($[0, 1]$, Logits $[-1.05, 1.05]$)**:
  - Exponentiate logits:
    $$
    e^{-1.05} \approx 0.349938, \qquad e^{1.05} \approx 2.857651
    $$
  - Row normalizer:
    $$
    S_1 = 0.349938 + 2.857651 = 3.207589
    $$
  - Probabilities:
    $$
    P_{1,0} = \frac{0.349938}{3.207589} \approx 0.109097, \qquad P_{1,1} = \frac{2.857651}{3.207589} \approx 0.890903
    $$
  - Verification: $0.109097 + 0.890903 = 1.000000$ (strongly predicts Class 1 with $89.1\%$).

- **Row 2 ($[1, 0]$, Logits $[0.35, -0.35]$)**:
  - Exponentiate logits:
    $$
    e^{0.35} \approx 1.419068, \qquad e^{-0.35} \approx 0.704688
    $$
  - Row normalizer:
    $$
    S_2 = 1.419068 + 0.704688 = 2.123756
    $$
  - Probabilities:
    $$
    P_{2,0} = \frac{1.419068}{2.123756} \approx 0.668188, \qquad P_{2,1} = \frac{0.704688}{2.123756} \approx 0.331812
    $$
  - Verification: $0.668188 + 0.331812 = 1.000000$ (predicts Class 0 with $66.8\%$, incorrect for XOR before training).

- **Row 3 ($[1, 1]$, Logits $[-0.75, 0.75]$)**:
  - Exponentiate logits:
    $$
    e^{-0.75} \approx 0.472367, \qquad e^{0.75} \approx 2.117000
    $$
  - Row normalizer:
    $$
    S_3 = 0.472367 + 2.117000 = 2.589367
    $$
  - Probabilities:
    $$
    P_{3,0} = \frac{0.472367}{2.589367} \approx 0.182426, \qquad P_{3,1} = \frac{2.117000}{2.589367} \approx 0.817574
    $$
  - Verification: $0.182426 + 0.817574 = 1.000000$ (predicts Class 1 with $81.8\%$, incorrect for XOR before training).

#### Resulting Probability Matrix $P$ (shape $(4, 2)$):

$$
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
$$

### Complete batch forward pass calculation

Using the illustrative weights, here is the complete batch forward pass for all 4 XOR examples:

#### 1. Input matrix $X$ and Target matrix $Y$

$$
X = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \qquad
Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

#### 2. Hidden pre-activation $Z_1 = X W_1 + b_1$

$$
Z_1 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 0.5 & -0.5 & 1.0 & -1.0 \\ 1.0 & 0.5 & -1.0 & -0.5 \end{bmatrix}
+ \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}
= \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & -0.9 & -0.4 \\
0.6 & -0.4 & 1.1 & -0.9 \\
1.6 & 0.1 & 0.1 & -1.4
\end{bmatrix}
$$

#### 3. Hidden activation $A_1 = \operatorname{ReLU}(Z_1)$

Clamp negative values to 0:

$$
A_1 = \begin{bmatrix}
0.1 & 0.1 & 0.1 & 0.1 \\
1.1 & 0.6 & 0.0 & 0.0 \\
0.6 & 0.0 & 1.1 & 0.0 \\
1.6 & 0.1 & 0.1 & 0.0
\end{bmatrix}
$$

#### 4. Output logits $Z_2 = A_1 W_2 + b_2$

$$
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
$$

#### 5. Output probabilities $P = \operatorname{softmax}(Z_2)$

Normalize each row independently:

$$
P = \begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
$$

Each row sums to $1.000000$.

---

## Q6 — Softmax cross-entropy loss

Cross-entropy measures the difference between predicted probabilities $P$ and
the correct one-hot targets $Y$.

For one example:

$$
L_i = -\sum_{c=1}^{2} Y_{i,c} \log(P_{i,c}) = -\log(P_{i, \text{true\_class}})
$$

For a batch of $m$ examples ($m = 4$):

$$
L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{c=1}^{2} Y_{i,c}\log(P_{i,c})
$$

### Example-by-example loss breakdown

| Example | Input $x$ | Target $y$ | Predicted Probabilities $P$ | Target Prob $P_{\text{true}}$ | Individual Loss $L_i = -\log(P_{\text{true}})$ | Note |
|:---|:---|:---|:---|:---|:---|:---|
| **0** | $[0, 0]$ | Class 0 ($[1, 0]$) | $[0.475021, 0.524979]$ | $0.475021$ | $-\log(0.475021) \approx 0.744396$ | Undecided |
| **1** | $[0, 1]$ | Class 1 ($[0, 1]$) | $[0.109097, 0.890903]$ | $0.890903$ | $-\log(0.890903) \approx 0.115520$ | Accurate prediction |
| **2** | $[1, 0]$ | Class 1 ($[0, 1]$) | $[0.668188, 0.331812]$ | $0.331812$ | $-\log(0.331812) \approx 1.103187$ | Predicts class 0 (wrong) |
| **3** | $[1, 1]$ | Class 0 ($[1, 0]$) | $[0.182426, 0.817574]$ | $0.182426$ | $-\log(0.182426) \approx 1.701411$ | Confidently wrong |

### Batch mean loss

$$
L = \frac{0.744396 + 0.115520 + 1.103187 + 1.701411}{4} = 0.91612888
$$

---

## Q7 — Backpropagation derivations and equations

Backpropagation calculates the derivative of the loss with respect to every weight and bias by moving backward through the network using the chain rule.

### 1. Derivation of the Output Gradient: $\frac{\partial L}{\partial Z_2}$

For a single example, let the logits be $z = [z_1, z_2]$, probabilities $p = [p_1, p_2]$, and one-hot target $y = [y_1, y_2]$.

The loss is:

$$
L = -y_1 \log(p_1) - y_2 \log(p_2) = -\sum_{k} y_k \log(p_k)
$$

Where softmax is:

$$
p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
$$

By the multivariate chain rule:

$$
\frac{\partial L}{\partial z_i} = \sum_{k} \frac{\partial L}{\partial p_k} \frac{\partial p_k}{\partial z_i}
$$

#### Step A: Derivative of Cross-Entropy with respect to $p_k$

$$
\frac{\partial L}{\partial p_k} = -\frac{y_k}{p_k}
$$

#### Step B: Derivative of Softmax $p_k$ with respect to logit $z_i$

Using the quotient rule:

**Case 1: When $k = i$**

$$
\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \sum e^{z_j} - e^{z_i} e^{z_i}}{\left(\sum e^{z_j}\right)^2} = p_i - p_i^2 = p_i (1 - p_i)
$$

**Case 2: When $k \neq i$**

$$
\frac{\partial p_k}{\partial z_i} = \frac{0 \cdot \sum e^{z_j} - e^{z_k} e^{z_i}}{\left(\sum e^{z_j}\right)^2} = -p_k p_i
$$

#### Step C: Chain rule combination

Substitute Step A and Step B into the chain rule:

$$
\begin{aligned}
\frac{\partial L}{\partial z_i} &= \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_i} + \sum_{k \neq i} \frac{\partial L}{\partial p_k}\frac{\partial p_k}{\partial z_i} \\
&= -\frac{y_i}{p_i} \cdot p_i(1 - p_i) + \sum_{k \neq i} \left(-\frac{y_k}{p_k}\right) \cdot (-p_k p_i) \\
&= -y_i(1 - p_i) + \sum_{k \neq i} y_k p_i \\
&= -y_i + y_i p_i + p_i \sum_{k \neq i} y_k \\
&= -y_i + p_i \left(y_i + \sum_{k \neq i} y_k\right)
\end{aligned}
$$

Because $y$ is one-hot, the sum over all classes is identically 1:

$$
\sum_{\text{all } k} y_k = y_i + \sum_{k \neq i} y_k = 1
$$

Therefore:

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

For the entire batch of $m$ examples with mean loss $L = \frac{1}{m} \sum L_i$:

$$
dZ_2 = \frac{P - Y}{m}
$$

### 2. Output-layer parameter gradients: $dW_2$ and $db_2$

Recall that $Z_2 = A_1 W_2 + b_2$. Applying the chain rule:

$$
dW_2 = A_1^T dZ_2
$$

$$
db_2 = \sum_{i=1}^{m} dZ_{2, i}
$$

**Dimension check:**
- $A_1^T$ has shape $(4, 4)$
- $dZ_2$ has shape $(4, 2)$
- $(4, 4) @ (4, 2) \to (4, 2)$ (matches $W_2$ shape)
- $db_2$ sums over rows of $dZ_2 \to (1, 2)$ (matches $b_2$ shape)

### 3. Propagating gradients to the hidden layer: $dA_1$ and $dZ_1$

Using the chain rule to find how loss changes with hidden activations:

$$
dA_1 = dZ_2 W_2^T
$$

**Dimension check:**
- $dZ_2$: $(4, 2)$
- $W_2^T$: $(2, 4)$
- $(4, 2) @ (2, 4) \to (4, 4)$ (matches $A_1$ shape)

Now pass through the ReLU activation derivative:

$$
\operatorname{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

Element-wise multiplication with the mask $(Z_1 > 0)$:

$$
dZ_1 = dA_1 \odot (Z_1 > 0)
$$

### 4. Hidden-layer parameter gradients: $dW_1$ and $db_1$

Recall that $Z_1 = X W_1 + b_1$. Applying the chain rule:

$$
dW_1 = X^T dZ_1
$$

$$
db_1 = \sum_{i=1}^{m} dZ_{1, i}
$$

**Dimension check:**
- $X^T$ has shape $(2, 4)$
- $dZ_1$ has shape $(4, 4)$
- $(2, 4) @ (4, 4) \to (2, 4)$ (matches $W_1$ shape)
- $db_1$ sums over rows of $dZ_1 \to (1, 4)$ (matches $b_1$ shape)


---

### Manual numerical calculation of gradients

Using our illustrative weights on the 4 XOR examples ($m = 4$):

#### 1. $dZ_2 = \frac{P - Y}{4}$
$$
dZ_2 = \frac{1}{4} \left(
\begin{bmatrix}
0.475021 & 0.524979 \\
0.109097 & 0.890903 \\
0.668188 & 0.331812 \\
0.182426 & 0.817574
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}
\right)
= \begin{bmatrix}
-0.131245 &  0.131245 \\
 0.027274 & -0.027274 \\
 0.167047 & -0.167047 \\
-0.204394 &  0.204394
\end{bmatrix}
$$

#### 2. $dW_2 = A_1^T dZ_2$ and $db_2 = \sum dZ_2$
$$
dW_2 = \begin{bmatrix}
-0.209924 &  0.209924 \\
-0.017199 &  0.017199 \\
 0.150188 & -0.150188 \\
-0.013124 &  0.013124
\end{bmatrix}, \qquad
db_2 = \begin{bmatrix} -0.141317 & 0.141317 \end{bmatrix}
$$

#### 3. $dA_1 = dZ_2 W_2^T$
$$
dA_1 = \begin{bmatrix}
 0.131245 &  0.262490 & -0.131245 &  0.131245 \\
-0.027274 & -0.054548 &  0.027274 & -0.027274 \\
-0.167047 & -0.334094 &  0.167047 & -0.167047 \\
 0.204394 &  0.408787 & -0.204394 &  0.204394
\end{bmatrix}
$$

#### 4. $dZ_1 = dA_1 \odot (Z_1 > 0)$
Masked by whether $Z_1 > 0$:
$$
dZ_1 = \begin{bmatrix}
 0.131245 &  0.262490 & -0.131245 &  0.131245 \\
-0.027274 & -0.054548 &  0.000000 &  0.000000 \\
-0.167047 &  0.000000 &  0.167047 &  0.000000 \\
 0.204394 &  0.408787 & -0.204394 &  0.000000
\end{bmatrix}
$$

#### 5. $dW_1 = X^T dZ_1$ and $db_1 = \sum dZ_1$
$$
dW_1 = \begin{bmatrix}
0.037347 & 0.408787 & -0.037347 & 0.000000 \\
0.177119 & 0.354239 & -0.204394 & 0.000000
\end{bmatrix}, \qquad
db_1 = \begin{bmatrix} 0.141317 & 0.616728 & -0.168591 & 0.131245 \end{bmatrix}
$$

---

## Q8 — Gradient-descent update

With learning rate $\eta$ (e.g. $\eta = 0.1$):

$$
\begin{aligned}
W_1 &\leftarrow W_1 - \eta \, dW_1 \\
b_1 &\leftarrow b_1 - \eta \, db_1 \\
W_2 &\leftarrow W_2 - \eta \, dW_2 \\
b_2 &\leftarrow b_2 - \eta \, db_2
\end{aligned}
$$

Each epoch executes:
```text
Forward Pass -> Loss Evaluation -> Backpropagation -> Parameter Update
```

---

## Q9 — Parameter initialization mathematics

### 1. The symmetry-breaking problem
If all weights are initialized to 0:
- $Z_1 = 0 + b_1$
- Every hidden neuron computes the identical activation: $a_{1, 1} = a_{1, 2} = a_{1, 3} = a_{1, 4}$
- In backpropagation, every hidden neuron receives the identical gradient: $dW_{1, j}$ are all identical
- The hidden neurons can never specialize to detect different geometric features. Symmetry must be broken with random initialization.

### 2. He (Kaiming) initialization for ReLU layers
To prevent vanishing or exploding activations across layers, the variance of the outputs should equal the variance of the inputs:

$$
\operatorname{Var}(W_1) = \frac{2}{n_{\text{in}}}
$$

Where $n_{\text{in}} = 2$ for the XOR input layer:

$$
W_1 \sim \mathcal{N}\left(0, \, \sigma = \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

### 3. Xavier / Glorot initialization for Softmax layers
For layers without ReLU:

$$
W_2 \sim \mathcal{N}\left(0, \, \sigma = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)
$$

Biases are safely initialized to zeros: $b_1 = \mathbf{0}_{(1, 4)}$, $b_2 = \mathbf{0}_{(1, 2)}$.

---

## Q10 — Prediction and evaluation metrics

### 1. Decision rule
Softmax outputs probabilities $P \in \mathbb{R}^{m \times 2}$. The predicted class label $\hat{y}$ is the index of the highest probability:

$$
\hat{y}_i = \operatorname{argmax}(P_{i, :})
$$

### 2. Accuracy metric
Accuracy measures the proportion of correctly classified examples:

$$
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\hat{y}_i == y_i)
$$

Where $\mathbb{I}(\cdot)$ is the indicator function ($1$ if true, $0$ if false).


---

## Summary of the complete mathematical pipeline

```text
1. Forward Pass
   Z1 = X @ W1 + b1                   shape: (4, 2) @ (2, 4) + (1, 4) -> (4, 4)
   A1 = max(0, Z1)                    shape: (4, 4)
   Z2 = A1 @ W2 + b2                  shape: (4, 4) @ (4, 2) + (1, 2) -> (4, 2)
   P  = softmax_batch(Z2)             shape: (4, 2)

2. Loss
   L  = -mean(sum(Y * log(P), axis=1))

3. Backpropagation
   dZ2 = (P - Y) / m                  shape: (4, 2)
   dW2 = A1.T @ dZ2                   shape: (4, 4) @ (4, 2) -> (4, 2)
   db2 = sum(dZ2, axis=0)             shape: (1, 2)
   dA1 = dZ2 @ W2.T                   shape: (4, 2) @ (2, 4) -> (4, 4)
   dZ1 = dA1 * (Z1 > 0)               shape: (4, 4)
   dW1 = X.T @ dZ1                    shape: (2, 4) @ (4, 4) -> (2, 4)
   db1 = sum(dZ1, axis=0)             shape: (1, 4)

4. Parameter Update
   W1 = W1 - lr * dW1
   b1 = b1 - lr * db1
   W2 = W2 - lr * dW2
   b2 = b2 - lr * db2

5. Evaluation
   predictions = argmax(P, axis=1)
   accuracy    = mean(predictions == y_true)
```
