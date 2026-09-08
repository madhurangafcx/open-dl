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

```text
X:  (4, 2)
W1: (2, 4)
b1: (1, 4)

(4, 2) @ (2, 4) + (1, 4) → (4, 4)
Z1.shape = (4, 4)
```

### 2. ReLU activation

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

$$
A_1 = \operatorname{ReLU}(Z_1)
$$

```text
ReLU(-3) = 0
ReLU(0)  = 0
ReLU(2)  = 2

A1.shape = (4, 4)
```

ReLU introduces non-linearity. Without it, two dense layers would still act
like one linear layer and could not solve XOR.

### 3. Output-layer weighted sum

The output layer calculates two raw class scores, called logits:

$$
Z_2 = A_1W_2 + b_2
$$

```text
A1: (4, 4)
W2: (4, 2)
b2: (1, 2)

(4, 4) @ (4, 2) + (1, 2) → (4, 2)
Z2.shape = (4, 2)
```

Example logits for one input:

```text
Z2 = [1.2, 3.5]
```

These are scores, not probabilities.

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
