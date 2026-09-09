# Step 1 — Mathematics for Sequence Prediction & Autoregressive Language Modeling

Goal: understand the complete mathematics of **Autoregressive Language Modeling**, next-token prediction under the probabilistic chain rule, causal sequence shifting, temperature scaling, top-k/top-p nucleus sampling, perplexity evaluation, and Backpropagation Through Time (BPTT) into dense embedding tables from first principles.
This document provides the rigorous theoretical foundations, mathematical proofs, step-by-step forward traces, and matrix calculus that will be implemented with pure NumPy from scratch.

---

## Q1 — The Probabilistic Foundations of Language: Joint Distributions & The Chain Rule

A language model is a statistical machine that assigns a joint probability distribution to any arbitrary sequence of discrete textual tokens:

```math
P(W) = P(w_1, w_2, w_3, \dots, w_T)
```

A well-trained language model assigns high probability to grammatically valid, semantically coherent sequences, and near-zero probability to scrambled noise:

```text
P("the blockchain transaction was confirmed") >> P("transaction the confirmed blockchain was")
```

---

### The Probability Chain Rule

By the fundamental product rule of conditional probability, any joint probability distribution over `T` random variables can be decomposed exactly—without making any independence assumptions—into a sequential product of conditional probabilities:

```math
P(w_1, w_2, \dots, w_T) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \dots P(w_T \mid w_1, \dots, w_{T-1})
```

In compact product notation:

```math
P(w_{1:T}) = \prod_{t=1}^{T} P(w_t \mid w_{<t}) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \dots, w_{t-1})
```

Where `w_{<t} = (w_1, \dots, w_{t-1})` denotes the **historical prefix** (or context window) preceding timestep `t`.

#### The Autoregressive Principle:
This decomposition is the mathematical cornerstone of modern Generative AI (from early Recurrent Neural Networks to GPT-4 and Claude):
**To model the probability of an entire document, we only need a neural network capable of solving one fundamental sub-problem: predicting the categorical probability distribution of the NEXT single token given all preceding tokens.**

---

### Why Classical Statistical N-Gram Models Failed

Before neural sequence models, computational linguistics relied on **Markovian N-gram models**, which approximated the infinite context `w_{<t}` by truncating history to the preceding `N - 1` words:

```math
P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-N+1:t-1}) = \frac{\operatorname{Count}(w_{t-N+1:t})}{\operatorname{Count}(w_{t-N+1:t-1})}
```

Statistical N-grams suffered from three catastrophic mathematical bottlenecks:

1. **The Curse of Dimensionality**:
   Tracking a 5-gram model over a modest vocabulary of `V = 50,000` tokens requires storing a discrete frequency table of size:
   ```math
   V^N = 50,000^5 = 3.125 \times 10^{23} \text{ parameters}
   ```
   This exceeds the storage capacity of all computer hard drives on Earth combined.
2. **The Zero-Frequency Sparsity Trap**:
   Language exhibits a heavy-tailed Zipfian distribution. If the exact phrase `"the decentralized network was"` never occurred in the training text, `\operatorname{Count} = 0`. An N-gram assigns:
   ```math
   P(\text{"secure"} \mid \text{"the decentralized network was"}) = 0.0
   ```
   Assigning zero probability causes the sequence loss to diverge to positive infinity (`-\ln(0) = \infty`), crippling evaluation.
3. **Total Semantic Blindness**:
   Discrete word counts treat words as mutually orthogonal isolated symbols. Observing `"the cat sat on the mat"` provides zero statistical evidence for `"the feline sat on the rug"`.

---

### The Neural Language Modeling Breakthrough (Bengio et al., 2003)

Neural language models resolve all three limitations simultaneously:
1. **Dense Continuous Embeddings (`E \in \mathbb{R}^{V \times D}`)**: Discrete token IDs are mapped into a low-dimensional continuous geometric manifold (`D \ll V`) where semantically related tokens share neighboring vector neighborhoods.
2. **Continuous Hidden Memory (`h_t \in \mathbb{R}^H`)**: Instead of storing discrete combinations of words in a table, the prefix history is compressed into an evolving recurrent state vector `h_t`.
3. **Linear Parameter Scaling**: Parameter count scales linearly with vocabulary size `O(V \cdot D + V \cdot H)`, completely independent of context length `T`!

---

## Q2 — The Causal Sequence Shift Topology (Many-to-Many with Offset +1)

To train an autoregressive model to predict the next token, the dataset must be structured with a **causal temporal offset of +1**.

```text
Sequence Length: T = 3
Original Text:   "<BOS> crypto blocks verify"

Input Sequence X:   [ <BOS>,    crypto,    blocks ]  (Tokens at positions 0, 1, 2)
                       │          │          │
                       ▼          ▼          ▼
                    [ Cell ] ──►[ Cell ] ──►[ Cell ]
                       │          │          │
                       ▼          ▼          ▼
Target Sequence Y:  [ crypto,   blocks,    verify ]  (Tokens at positions 1, 2, 3)
```

### Mathematical Formulation of Causal Shift:

For a text sequence of `T + 1` tokens `(w_0, w_1, \dots, w_T)`:

```math
X = \begin{bmatrix} w_0, & w_1, & \dots, & w_{T-1} \end{bmatrix} \in \{0, \dots, V-1\}^{1 \times T}
```

```math
Y = \begin{bmatrix} w_1, & w_2, & \dots, & w_T \end{bmatrix} \in \{0, \dots, V-1\}^{1 \times T}
```

At each timestep `t \in \{1, \dots, T\}`:
- The input to the network is token `x_t = w_{t-1}`.
- The ground-truth supervision target is token `y_t = w_t`.
- The network computes logits `o_t \in \mathbb{R}^{1 \times V}` and softmax distribution `\hat{y}_t \in (0, 1)^{1 \times V}`.
- The training loss is evaluated immediately at step `t`:
  ```math
  L_t = -\ln\left(\hat{y}_{t, y_t}\right)
  ```

---

## Q3 — Vocabulary Tokenization & The Dense Embedding Matrix (E)

Computers cannot process raw strings; words must be converted into discrete integer IDs and then into continuous dense vectors.

### 1. The Vocabulary Lookup Table
Let `\mathcal{V}` be the vocabulary of size `V`. Every token is assigned a unique index `k \in \{0, 1, \dots, V-1\}`.

### 2. The One-Hot Representation Equivalence
Mathematically, lookup of token `w` is equivalent to multiplying a one-hot row vector `\mathbf{1}_w \in \{0, 1\}^{1 \times V}` by the embedding matrix `E \in \mathbb{R}^{V \times D}`:

```math
e_t = \mathbf{1}_{w_t} E = E[w_t, :] \in \mathbb{R}^{1 \times D}
```

Where:
- `V`: Vocabulary size.
- `D`: Embedding feature dimension (`D \ll V`, typically `64` to `4096`).
- `E[k, :]`: The `k`-th row of matrix `E`, representing the continuous coordinates of token `k`.

---

### 3. Gradient Routing Through Embedding Lookups
During backpropagation, upstream gradients arrive at the dense embedding vector:

```math
\delta_{e_t} = \frac{\partial L}{\partial e_t} \in \mathbb{R}^{1 \times D}
```

Because `e_t` was extracted directly from row `w_t` of matrix `E`, the gradient with respect to `E` is a **sparse index accumulation**:

```math
\frac{\partial L}{\partial E[k, :]} = \sum_{t=1}^{T} \delta_{e_t} \cdot \mathbb{I}(w_t = k)
```

In NumPy, this is vectorized using `np.add.at(dE, token_indices, d_et)`. If a token appears multiple times in a sequence, all its gradient updates sum together into that token's embedding row.

---

## Q4 — Recurrent Context Compression & State Dynamics

At timestep `t`, the network must combine the new token embedding `e_t` with the historical summary `h_{t-1}`.

### Recurrent State Transition:

```math
z_{h, t} = e_t W_{xh} + h_{t-1} W_{hh} + b_h \in \mathbb{R}^{1 \times H}
```

```math
h_t = \tanh(z_{h, t}) \in (-1, 1)^{1 \times H}
```

Where:
- `e_t \in \mathbb{R}^{1 \times D}` is the current token embedding.
- `h_{t-1} \in \mathbb{R}^{1 \times H}` is the previous hidden state summarizing `w_{<t}`.
- `W_{xh} \in \mathbb{R}^{D \times H}` is the input-to-hidden projection matrix.
- `W_{hh} \in \mathbb{R}^{H \times H}` is the recurrent state-to-state transition matrix.
- `b_h \in \mathbb{R}^{1 \times H}` is the hidden state bias vector.

---

### Vocabulary Output Projection (Logits):

To predict probabilities over the `V` vocabulary words, the hidden state `h_t` is projected into vocabulary space:

```math
o_t = h_t W_{hy} + b_y \in \mathbb{R}^{1 \times V}
```

Where:
- `W_{hy} \in \mathbb{R}^{H \times V}` is the hidden-to-vocabulary output weight matrix.
- `b_y \in \mathbb{R}^{1 \times V}` is the vocabulary output bias vector.
- `o_{t, k}` is the unnormalized log-probability score assigned to word `k`.

---

## Q5 — Categorical Softmax Distribution & Sequence Cross-Entropy Loss

### Softmax Normalization:
The raw logits `o_t` are converted into valid non-negative probabilities summing to 1.0 via Softmax:

```math
\hat{y}_{t, k} = P(w_{t+1} = k \mid w_{1:t}) = \frac{\exp(o_{t, k})}{\sum_{j=0}^{V-1} \exp(o_{t, j})}
```

For numerical stability, we subtract the maximum logit before exponentiating:

```math
o'_{t, k} = o_{t, k} - \max_j(o_{t, j})
```

---

### Per-Token Negative Log-Likelihood Loss:
Given the true ground-truth target token `y_t^* \in \{0, \dots, V-1\}`, the model incurs a loss equal to the negative log-probability assigned to that specific correct token:

```math
L_t = -\ln\left(\hat{y}_{t, y_t^*} + \epsilon\right)
```

Where `\epsilon = 10^{-15}` prevents `\ln(0)` in edge cases.

---

### Sequence-Level Total and Mean Cross-Entropy:

For a sequence of length `T`, the total sequence loss is the sum of per-token losses:

```math
L_{\mathrm{total}} = \sum_{t=1}^{T} L_t
```

The mean token cross-entropy loss is:

```math
\bar{L} = \frac{1}{T} \sum_{t=1}^{T} L_t
```

---

## Q6 — The Mathematics of Perplexity (PPL)

Perplexity is the standard evaluation metric for language models.

### Mathematical Definition:
Perplexity is the exponentiated mean cross-entropy loss per token:

```math
\mathrm{PPL} = \exp\left(\bar{L}\right) = \exp\left( \frac{1}{T} \sum_{t=1}^{T} L_t \right)
```

Expanding `L_t = -\ln P(w_t^* \mid w_{<t})`:

```math
\mathrm{PPL} = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \ln P(w_t^* \mid w_{<t}) \right) = \left( \prod_{t=1}^{T} \frac{1}{P(w_t^* \mid w_{<t})} \right)^{1/T}
```

Perplexity is the **geometric mean reciprocal probability** of the ground-truth sequence!

---

### Physical Intuition: The Effective Branching Factor
Perplexity represents the **effective number of equally likely choices** the model is confused between at each step:
- **PPL = 1.0**: Absolute certainty. The model assigns probability `1.0` to the correct next word at every single step (`\ln(1.0) = 0 \implies \exp(0) = 1.0`).
- **PPL = 5.0**: The model's uncertainty is identical to rolling a fair 5-sided die at each word.
- **PPL = V (Vocabulary Size)**: Total ignorance. Equivalent to uniform random guessing across all `V` tokens (`P = 1/V \implies \mathrm{PPL} = \exp(\ln(V)) = V`).
- **Lower Perplexity = Superior Predictive Model**.

---

## Q7 — Training with Teacher Forcing vs. Autonomous Generation

Understanding the boundary between **Training** and **Inference** is vital in autoregressive modeling.

```text
====================================================================================================
TRAINING: TEACHER FORCING
====================================================================================================
Step 1: Input: "<BOS>"  ──► [Model] ──► Predicts: "banana" (WRONG!)
                                          │
                                          └──► Loss evaluated against target "crypto"
Step 2: Input: "crypto" ──► [Model] ──► Predicts: "blocks" (CORRECT!)
        (Ground Truth Fed!)

* The model is ALWAYS fed ground-truth tokens, ignoring its own previous mistakes!
====================================================================================================
INFERENCE: AUTONOMOUS FREE-RUNNING GENERATION
====================================================================================================
Step 1: Input: "<BOS>"  ──► [Model] ──► Sampled: "crypto"
                                           │
Step 2: Input: "crypto" ◄──────────────────┘ (Model feeds its OWN previous output!)
          │
          ▼
        [Model] ────────► Sampled: "blocks"
                            │
Step 3: Input: "blocks" ◄───┘
```

### The Exposure Bias Dilemma:
- **During Training**: The model never sees mistakes in its prefix history.
- **During Inference**: If the model samples an unusual or incorrect word at step 2, it enters an unfamiliar hidden state region `h_2` that was never encountered during training.
- This distribution mismatch can cause cascading errors and repetitive degenerative loops.

---

## Q8 — Decoding & Sampling Strategies: Mathematical Formulations & Limits

Once the model computes output logits `o_t \in \mathbb{R}^V`, how do we select the next token `w_{t+1}`?

---

### 1. Greedy Decoding (Argmax)
Selects the single token with highest logit:

```math
w_{t+1} = \operatorname{argmax}_{k \in \{0, \dots, V-1\}} o_{t, k}
```

- **Characteristics**: Deterministic, zero variance.
- **Flaw**: Highly susceptible to infinite repetitive loops (`"the blockchain was confirmed by the blockchain was confirmed..."`).

---

### 2. Temperature-Scaled Softmax Sampling
We divide the logits by a temperature hyperparameter `T_{\mathrm{temp}} > 0` before evaluating Softmax:

```math
P_T(w_{t+1} = k) = \frac{\exp\left(o_{t, k} / T_{\mathrm{temp}}\right)}{\sum_{j=0}^{V-1} \exp\left(o_{t, j} / T_{\mathrm{temp}}\right)}
```

#### Mathematical Asymptotic Limits:

1. **Cold Temperature Limit (`T_{\mathrm{temp}} \to 0^+`)**:
   ```math
   \lim_{T_{\mathrm{temp}} \to 0} P_T(w = k) = \begin{cases} 1.0 & \text{if } k = \operatorname{argmax}_j(o_j) \\ 0.0 & \text{otherwise} \end{cases}
   ```
   The distribution collapses into a **Dirac delta distribution** identical to Greedy Argmax.
2. **Standard Temperature (`T_{\mathrm{temp}} = 1.0`)**:
   Preserves the original categorical probability distribution learned during training.
3. **Hot Temperature Limit (`T_{\mathrm{temp}} \to \infty`)**:
   ```math
   \lim_{T_{\mathrm{temp}} \to \infty} \frac{o_{t, k}}{T_{\mathrm{temp}}} = 0 \implies \lim_{T_{\mathrm{temp}} \to \infty} P_T(w = k) = \frac{\exp(0)}{\sum_{j=0}^{V-1} \exp(0)} = \frac{1}{V}
   ```
   The distribution flattens into a **uniform random distribution** over the entire vocabulary.

---

### 3. Top-K Truncation Sampling
Restricts sampling candidates strictly to the `K` highest logits, setting all other logits to `-\infty`:

```math
V_{\mathrm{top-k}} = \operatorname{argtopk}_{k \in \mathcal{V}}(o_{t, k}, K)
```

```math
o'_{t, j} = \begin{cases} o_{t, j} & \text{if } j \in V_{\mathrm{top-k}} \\ -\infty & \text{otherwise} \end{cases}
```

```math
P_{\mathrm{top-k}}(w_{t+1} = k) = \operatorname{softmax}(o'_t)
```

This guarantees that catastrophic low-probability tokens in the long tail are never sampled.

---

### 4. Top-P (Nucleus) Sampling (Holtzman et al., 2019)
Instead of fixing `K`, Top-P dynamically chooses the smallest set of tokens whose **cumulative probability mass** exceeds threshold `p \in (0, 1)`:

Sort vocabulary in descending order of probability: `\hat{y}_{(1)} \ge \hat{y}_{(2)} \ge \dots \ge \hat{y}_{(V)}`.
Find cutoff index `M`:

```math
M = \min \left\{ m \in \{1, \dots, V\} : \sum_{j=1}^{m} \hat{y}_{(j)} \ge p \right\}
```

The nucleus candidate set is:

```math
V^{(p)} = \{ (1), (2), \dots, (M) \}
```

All tokens outside `V^{(p)}` are masked to `-\infty`, and probabilities are renormalized:

```math
P_{\mathrm{nucleus}}(w = k) = \begin{cases} \frac{\hat{y}_k}{\sum_{j \in V^{(p)}} \hat{y}_j} & \text{if } k \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}
```

- When the model is highly confident (e.g. `P(\text{"Obama"} \mid \text{"Barack"}) = 0.98`), `M = 1` (behaves like greedy).
- When the model is uncertain, `M` dynamically expands to 20 or 50 tokens.

---

## Q9 — Concrete Numerical Setup (Matching verify_sequence_prediction.py)

To verify every equation down to 6 decimal places, we define a compact language modeling task:

### Dimensions:
- Vocabulary Size `V = 5`
- Embedding Dimension `D = 3`
- Hidden State Dimension `H = 4`
- Sequence Length `T = 3`
- Batch Size `B = 1`

---

### Vocabulary Mapping:
```text
0: <BOS>
1: crypto
2: blocks
3: verify
4: <EOS>
```

---

### Training Sequence Tokens:
- **Input Sequence `X` (`1 \times 3`)**: `[0, 1, 2]` (`<BOS>`, `"crypto"`, `"blocks"`)
- **Target Sequence `Y` (`1 \times 3`)**: `[1, 2, 3]` (`"crypto"`, `"blocks"`, `"verify"`)

---

### Initial Parameter Matrices:

#### 1. Vocabulary Embedding Table `E \in \mathbb{R}^{5 \times 3}`:
```math
E = \begin{bmatrix}
 0.2 & -0.1 &  0.3 \quad (\text{0: <BOS>}) \\
-0.3 &  0.4 &  0.1 \quad (\text{1: crypto}) \\
 0.1 & -0.2 &  0.4 \quad (\text{2: blocks}) \\
 0.3 &  0.1 & -0.2 \quad (\text{3: verify}) \\
-0.1 &  0.2 & -0.3 \quad (\text{4: <EOS>})
\end{bmatrix}
```

#### 2. Input-to-Hidden Weights `W_{xh} \in \mathbb{R}^{3 \times 4}`:
```math
W_{xh} = \begin{bmatrix}
 0.2 & -0.3 &  0.1 &  0.4 \\
-0.1 &  0.2 & -0.4 &  0.3 \\
 0.3 &  0.1 &  0.2 & -0.2
\end{bmatrix}
```

#### 3. Recurrent Transition Weights `W_{hh} \in \mathbb{R}^{4 \times 4}`:
```math
W_{hh} = \begin{bmatrix}
 0.1 & -0.2 &  0.3 &  0.1 \\
-0.1 &  0.4 &  0.1 & -0.3 \\
 0.2 & -0.1 &  0.2 &  0.4 \\
 0.3 &  0.1 & -0.2 &  0.1
\end{bmatrix}, \quad
b_h = \begin{bmatrix} 0.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
```

#### 4. Vocabulary Output Projection `W_{hy} \in \mathbb{R}^{4 \times 5}`:
```math
W_{hy} = \begin{bmatrix}
 0.3 & -0.2 &  0.4 & -0.1 &  0.2 \\
-0.2 &  0.5 & -0.1 &  0.3 & -0.4 \\
 0.1 & -0.3 &  0.2 &  0.4 & -0.1 \\
 0.4 &  0.1 & -0.2 & -0.3 &  0.5
\end{bmatrix}, \quad
b_y = \begin{bmatrix} 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
```

Initial hidden state:
```math
h_0 = \begin{bmatrix} 0.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
```

---

## Q10 — Forward Pass: Step-by-Step Numerical Walkthrough

We now trace the forward computation across all 3 timesteps with exact arithmetic.

---

### Timestep 1 (`t = 1`):

#### 1. Token Lookup:
Input token is `x_1 = 0` (`<BOS>`):
```math
e_1 = E[0, :] = [0.2, -0.1, 0.3]
```

#### 2. Recurrent Hidden State:
```math
z_{h, 1} = e_1 W_{xh} + h_0 W_{hh} + b_h
```
- `e_1 W_{xh} = [0.2(0.2) - 0.1(-0.1) + 0.3(0.3), 0.2(-0.3) - 0.1(0.2) + 0.3(0.1), 0.2(0.1) - 0.1(-0.4) + 0.3(0.2), 0.2(0.4) - 0.1(0.3) + 0.3(-0.2)]`
  `= [0.04 + 0.01 + 0.09, -0.06 - 0.02 + 0.03, 0.02 + 0.04 + 0.06, 0.08 - 0.03 - 0.06] = [0.14, -0.05, 0.12, -0.01]`
- `h_0 W_{hh} = [0.0, 0.0, 0.0, 0.0]`
- `z_{h, 1} = [0.14, -0.05, 0.12, -0.01]`
```math
h_1 = \tanh(z_{h, 1}) = [0.139092, -0.049958, 0.119427, -0.010000]
```

#### 3. Output Logits:
```math
o_1 = h_1 W_{hy} + b_y = [0.059662, -0.089626, 0.086518, 0.021874, 0.030859]
```

#### 4. Softmax Distribution:
```math
\hat{y}_1 = \operatorname{softmax}(o_1) = [0.207337, 0.178584, 0.212981, 0.199648, 0.201450]
```

#### 5. Target Loss:
The true target token is `y_1 = 1` (`"crypto"`):
```math
P(w_2 = \text{"crypto"}) = 0.178584
```
```math
L_1 = -\ln(0.178584) = 1.722698
```

---

### Timestep 2 (`t = 2`):

#### 1. Token Lookup:
Input token is `x_2 = 1` (`"crypto"`):
```math
e_2 = E[1, :] = [-0.3, 0.4, 0.1]
```

#### 2. Recurrent Hidden State:
```math
z_{h, 2} = e_2 W_{xh} + h_1 W_{hh} + b_h = [-0.07, 0.18, -0.17, 0.02] + [0.039791, -0.060745, 0.062617, 0.035668]
         = [-0.030209, 0.119255, -0.107383, 0.055668]
```
```math
h_2 = \tanh(z_{h, 2}) = [-0.030200, 0.118693, -0.106972, 0.055610]
```

#### 3. Output Logits and Softmax:
```math
o_2 = h_2 W_{hy} + b_y = [-0.021252, 0.103039, -0.056466, -0.020844, -0.015015]
```
```math
\hat{y}_2 = \operatorname{softmax}(o_2) = [0.195909, 0.221837, 0.189130, 0.195989, 0.197135]
```

#### 4. Target Loss:
The true target token is `y_2 = 2` (`"blocks"`):
```math
P(w_3 = \text{"blocks"}) = 0.189130
```
```math
L_2 = -\ln(0.189130) = 1.665319
```

---

### Timestep 3 (`t = 3`):

#### 1. Token Lookup:
Input token is `x_3 = 2` (`"blocks"`):
```math
e_3 = E[2, :] = [0.1, -0.2, 0.4]
```

#### 2. Recurrent Hidden State:
```math
z_{h, 3} = e_3 W_{xh} + h_2 W_{hh} + b_h = [0.16, -0.03, 0.17, -0.10] + [-0.019601, 0.069776, -0.029707, -0.075856]
         = [0.140399, 0.039776, 0.140293, -0.175856]
```
```math
h_3 = \tanh(z_{h, 3}) = [0.139484, 0.039755, 0.139380, -0.174065]
```

#### 3. Output Logits and Softmax:
```math
o_3 = h_3 W_{hy} + b_y = [-0.021794, -0.067240, 0.114507, 0.105949, -0.088976]
```
```math
\hat{y}_3 = \operatorname{softmax}(o_3) = [0.193316, 0.184727, 0.221545, 0.219657, 0.180755]
```

#### 4. Target Loss:
The true target token is `y_3 = 3` (`"verify"`):
```math
P(w_4 = \text{"verify"}) = 0.219657
```
```math
L_3 = -\ln(0.219657) = 1.515687
```

---

### Sequence Loss and Perplexity Summary:

```math
L_{\mathrm{total}} = L_1 + L_2 + L_3 = 1.722698 + 1.665319 + 1.515687 = 4.903704
```

```math
\bar{L} = \frac{L_{\mathrm{total}}}{3} = \frac{4.903704}{3} = 1.634568
```

```math
\mathrm{PPL} = \exp(1.634568) = 5.127242
```

---

### Detailed Walkthrough of Decoding Algorithms on Step 3 Logits

Given step 3 logits:
```text
o_3 = [-0.021794, -0.067240, +0.114507, +0.105949, -0.088976]
Tokens: [ 0: <BOS>,  1: crypto,  2: blocks,   3: verify,   4: <EOS> ]
```

#### 1. Greedy Search:
```math
w_4 = \operatorname{argmax}(o_3) = 2 \quad (\text{"blocks"})
```

#### 2. Temperature Scaling:
- **Cold (`T = 0.5`)**: `P = [0.185465, 0.169351, 0.243585, 0.239452, 0.162147]` (sharpens top predictions).
- **Standard (`T = 1.0`)**: `P = [0.193316, 0.184727, 0.221545, 0.219657, 0.180755]`.
- **Hot (`T = 2.0`)**: `P = [0.196812, 0.192391, 0.210693, 0.209793, 0.190311]` (approaches uniform 0.20).

#### 3. Top-K Sampling (`K = 2`):
Top 2 tokens are `2` (`"blocks"`) and `3` (`"verify"`). All others set to `-\infty`:
```math
P_{\mathrm{top-2}} = [0.0, 0.0, 0.502139, 0.497861, 0.0]
```

#### 4. Top-P (Nucleus) Sampling (`p = 0.70`):
Sorting descending by probability:
1. `"blocks"` (0.221545) $\to$ cumulative = 0.221545
2. `"verify"` (0.219657) $\to$ cumulative = 0.441202
3. `"<BOS>"` (0.193316) $\to$ cumulative = 0.634518
4. `"crypto"` (0.184727) $\to$ cumulative = 0.819245 $\ge 0.70$ (Cutoff reached!)
Nucleus candidate set: `{blocks, verify, <BOS>, crypto}`. Token `<EOS>` is dropped:
```math
P_{\mathrm{nucleus}} = [0.270426, 0.268122, 0.235968, 0.225484, 0.0]
```

---

## Q11 — Backpropagation Through Time (BPTT) with Embedding Table Updates

---

### Backward Timestep 3 (`t = 3`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 3} = \hat{y}_3 - \mathbf{1}_{y_3} = [0.193316, 0.184727, 0.221545, 0.219657 - 1.0, 0.180755]
                           = [0.193316, 0.184727, 0.221545, -0.780343, 0.180755]
```

#### 2. Hidden State Error:
Since `t = 3` is the last step, `\delta_{h_3, \mathrm{recurrent}} = \mathbf{0}`:
```math
\delta_{h_3} = \delta_{\mathrm{logit}, 3} W_{hy}^T = [0.223853, -0.274859, -0.321990, 0.375970]
```

#### 3. Pre-activation Error:
```math
\delta_{z_{h, 3}} = \delta_{h_3} \odot (1 - h_3^2) = [0.219497, -0.274425, -0.315735, 0.364579]
```

#### 4. Embedding Gradient Accumulation:
```math
\delta_{e_3} = \delta_{z_{h, 3}} W_{xh}^T = [0.240485, 0.158833, -0.097656]
```
Input token was `x_3 = 2` (`"blocks"`):
```math
\frac{\partial L}{\partial E[2, :]} += [0.240485, 0.158833, -0.097656]
```

#### 5. Recurrent Error to `h_2`:
```math
\delta_{h_2, \mathrm{recurrent}} = \delta_{z_{h, 3}} W_{hh}^T = [0.018572, -0.272667, 0.154027, 0.138012]
```

---

### Backward Timestep 2 (`t = 2`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 2} = [0.195909, 0.221837, 0.189130 - 1.0, 0.195989, 0.197135]
                           = [0.195909, 0.221837, -0.810870, 0.195989, 0.197135]
```

#### 2. Hidden State Error:
```math
\delta_{h_2} = \delta_{\mathrm{logit}, 2} W_{hy}^T + \delta_{h_2, \mathrm{recurrent}}
             = [-0.290114, 0.132766, -0.150452, 0.302492] + [0.018572, -0.272667, 0.154027, 0.138012]
             = [-0.271542, -0.139900, 0.003575, 0.440504]
```

#### 3. Pre-activation Error:
```math
\delta_{z_{h, 2}} = \delta_{h_2} \odot (1 - h_2^2) = [-0.271295, -0.137930, 0.003534, 0.439141]
```

#### 4. Embedding Gradient Accumulation:
```math
\delta_{e_2} = \delta_{z_{h, 2}} W_{xh}^T = [0.163130, 0.129872, -0.182303]
```
Input token was `x_2 = 1` (`"crypto"`):
```math
\frac{\partial L}{\partial E[1, :]} += [0.163130, 0.129872, -0.182303]
```

#### 5. Recurrent Error to `h_1`:
```math
\delta_{h_1, \mathrm{recurrent}} = \delta_{z_{h, 2}} W_{hh}^T = [0.045431, -0.159431, 0.135897, -0.051974]
```

---

### Backward Timestep 1 (`t = 1`):

#### 1. Output Error:
```math
\delta_{\mathrm{logit}, 1} = [0.207337, 0.178584 - 1.0, 0.212981, 0.199648, 0.201450]
                           = [0.207337, -0.821416, 0.212981, 0.199648, 0.201450]
```

#### 2. Hidden State Error:
```math
\delta_{h_1} = \delta_{\mathrm{logit}, 1} W_{hy}^T + \delta_{h_1, \mathrm{recurrent}}
             = [0.332002, -0.494159, 0.369469, -0.000972] + [0.045431, -0.159431, 0.135897, -0.051974]
             = [0.377433, -0.653591, 0.505366, -0.052946]
```

#### 3. Pre-activation Error:
```math
\delta_{z_{h, 1}} = \delta_{h_1} \odot (1 - h_1^2) = [0.370130, -0.651959, 0.498158, -0.052941]
```

#### 4. Embedding Gradient Accumulation:
```math
\delta_{e_1} = \delta_{z_{h, 1}} W_{xh}^T = [0.298253, -0.382551, 0.156063]
```
Input token was `x_1 = 0` (`<BOS>`):
```math
\frac{\partial L}{\partial E[0, :]} += [0.298253, -0.382551, 0.156063]
```

---

### Final Accumulated Gradients Across All Timesteps:

#### 1. Embedding Table Gradients `\frac{\partial L}{\partial E} \in \mathbb{R}^{5 \times 3}`:
```math
\frac{\partial L}{\partial E} = \begin{bmatrix}
 0.298253 & -0.382551 &  0.156063 \quad (\text{Updated from <BOS> at } t=1) \\
 0.163130 &  0.129872 & -0.182303 \quad (\text{Updated from crypto at } t=2) \\
 0.240485 &  0.158833 & -0.097656 \quad (\text{Updated from blocks at } t=3) \\
 0.000000 &  0.000000 &  0.000000 \quad (\text{Unvisited: verify}) \\
 0.000000 &  0.000000 &  0.000000 \quad (\text{Unvisited: <EOS>})
\end{bmatrix}
```

#### 2. Vocabulary Projection Parameters:
```math
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^3 h_t^T \delta_{\mathrm{logit}, t} = \begin{bmatrix}
 0.049887 & -0.095186 &  0.085014 & -0.086995 &  0.047279 \\
 0.020580 &  0.074711 & -0.098078 & -0.017734 &  0.020520 \\
 0.030749 & -0.096083 &  0.143055 & -0.105886 &  0.028164 \\
-0.024828 & -0.011604 & -0.085786 &  0.144733 & -0.022515
\end{bmatrix}
```
```math
\frac{\partial L}{\partial b_y} = \sum_{t=1}^3 \delta_{\mathrm{logit}, t} = \begin{bmatrix} 0.596562 & -0.414853 & -0.376344 & -0.384705 & 0.579340 \end{bmatrix}
```

#### 3. Recurrent Parameters:
```math
\frac{\partial L}{\partial W_{xh}} = \begin{bmatrix}
 0.177364 & -0.116455 &  0.066998 & -0.105873 \\
-0.189430 &  0.064909 &  0.014745 &  0.108035 \\
 0.171709 & -0.319151 &  0.023507 &  0.173863
\end{bmatrix}
```
```math
\frac{\partial L}{\partial W_{hh}} = \begin{bmatrix}
-0.044364 & -0.010897 &  0.010027 &  0.050071 \\
 0.039606 & -0.025682 & -0.037652 &  0.021334 \\
-0.055880 &  0.012883 &  0.034197 &  0.013446 \\
 0.014919 & -0.013882 & -0.017593 &  0.015883
\end{bmatrix}, \quad
\frac{\partial L}{\partial b_h} = \begin{bmatrix} 0.318333 & -1.064314 & 0.185957 & 0.750779 \end{bmatrix}
```

---

## Q12 — Complete Mathematical Mapping to sequence_model.py Implementation

This cross-reference connects every mathematical equation derived above directly to its implementation in [`src/sequence_model.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/11-sequence-prediction/src/sequence_model.py):

| Component / Function | Mathematical Equation | NumPy Variables & Dimensions |
| :--- | :--- | :--- |
| `Embedding.forward(tokens)` | Discrete table lookup `e_t = E[w_t]` | `tokens: (B, T)` -> `X_embed: (B, T, D)` |
| `Embedding.backward(d_embed)` | Sparse index accumulation `\frac{\partial L}{\partial E}` | `np.add.at(dE, tokens, d_embed)` |
| `recurrent_forward(X_embed, h_0)` | `z_t = e_t W_{xh} + h_{t-1} W_{hh} + b_h`, `h_t = \tanh(z_t)` | `X_embed: (B, T, D)` -> `H: (B, T, H)` |
| `dense_vocab_projection(H)` | `O = H W_{hy} + b_y` | `H: (B, T, H)` -> `logits: (B, T, V)` |
| `sequence_cross_entropy(logits, targets)` | `L_t = -\ln(\hat{y}_{t, y_t})`, `\bar{L} = \frac{1}{T}\sum L_t` | Returns scalar loss and `\mathrm{PPL} = \exp(\bar{L})` |
| `recurrent_backward(d_logits)` | BPTT across all `T` steps with causal error | Computes `dW_hy, db_y, dW_xh, dW_hh, db_h, dE` |
| `generate_step(o_t, method)` | Sampling via Argmax, Temperature, Top-K, or Top-P | `logits: (V,)` -> next integer token ID |

---

## Summary of the Complete Mathematical Pipeline

```text
========================================================================================================================
FORWARD AUTOREGRESSIVE SEQUENCE PIPELINE (One Timestep)
========================================================================================================================
1. Embedding Table Lookup:             e_t = E[w_t]                                     e_t in R^(B x D)
2. Recurrent Context Compression:      z_t = e_t @ W_xh + h_{t-1} @ W_hh + b_h          z_t in R^(B x H)
                                       h_t = tanh(z_t)                                  h_t in (-1, 1)
3. Vocabulary Projection (Logits):     o_t = h_t @ W_hy + b_y                           o_t in R^(B x V)
4. Softmax Probability Distribution:   y_hat_t = softmax(o_t)                           y_hat_t in (0, 1)
5. Causal Negative Log-Likelihood:     L_t = -ln(y_hat_{t, w_{t+1}^*})                  Scalar L_t >= 0
========================================================================================================================
BACKWARD BPTT AND EMBEDDING UPDATE PIPELINE (Reverse Time)
========================================================================================================================
1. Vocabulary Logit Error:             d_logit = y_hat_t - 1_{w_{t+1}^*}                d_logit in R^(B x V)
2. Vocabulary Weight Gradients:        dW_hy += h_t.T @ d_logit                         dW_hy in R^(H x V)
                                       db_y  += sum(d_logit)                            db_y in R^(1 x V)
3. Total Hidden State Error:           d_h = d_logit @ W_hy.T + d_h_recurrent           d_h in R^(B x H)
4. Pre-activation Error:               d_z = d_h * (1 - h_t^2)                          d_z in R^(B x H)
5. Recurrent Weight Gradients:         dW_xh += e_t.T @ d_z                             dW_xh in R^(D x H)
                                       dW_hh += h_{t-1}.T @ d_z                         dW_hh in R^(H x H)
                                       db_h  += sum(d_z)                                db_h in R^(1 x H)
6. Embedding Gradient & Accumulation:  d_et = d_z @ W_xh.T                              d_et in R^(B x D)
                                       dE[w_t] += d_et                                  Sparse row addition
7. Propagate Recurrent Error (t-1):    d_h_recurrent = d_z @ W_hh.T                     d_h_recurrent in R^(B x H)
========================================================================================================================
```

This completes the foundational mathematics for Project 11. The next step is validating
each of these formulas in executable NumPy code in [`src/sequence_model.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/11-sequence-prediction/src/sequence_model.py).
