# 11 — Sequence Prediction & Autoregressive Language Modeling

## Goal

Understand how neural sequence models predict the next element in a sequence, establishing the theoretical and computational foundation for **Autoregressive Language Modeling** and text generation from first principles with pure NumPy.

All modern Generative AI systems and Large Language Models (LLMs)—including GPT, Claude, and Gemini—share a single foundational training objective: **Next-Token Prediction** under the probabilistic chain rule:

```math
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \dots, w_{t-1})
```

While state-of-the-art production LLMs implement this objective using Transformer self-attention (`13-mini-transformer`), the core principles of autoregressive generation—**dense token embeddings, causal sequence shifting, categorical cross-entropy loss, Teacher Forcing, temperature scaling, top-k/top-p sampling, and perplexity evaluation**—were pioneered by Recurrent Language Models (RNNs, LSTMs, and GRUs).

```text
Input Tokens:     [ "the",       "blockchain",   "transaction", "was"       ]
                     │                 │               │          │
                     ▼                 ▼               ▼          ▼
Embedding Table:   [ e_1 ]           [ e_2 ]         [ e_3 ]    [ e_4 ]
                     │                 │               │          │
                     ▼                 ▼               ▼          ▼
Recurrent State:   h_0 ──► [ Cell ] ─► [ Cell ] ─────► [ Cell ] ─► [ Cell ] ──► h_4
                             │           │               │          │
                             ▼           ▼               ▼          ▼
Output Projection:         [ o_1 ]     [ o_2 ]         [ o_3 ]    [ o_4 ]
                             │           │               │          │
                             ▼           ▼               ▼          ▼
Softmax Distribution:      p(w_2|w_1)  p(w_3|w_1:2)   p(w_4|w_1:3)p(w_5|w_1:4)
                             │           │               │          │
Target Tokens:    [ "blockchain", "transaction",     "was",    "confirmed" ]
```

---

## The Core Mathematical Equations of Sequence Prediction

At timestep `t`, given the sequence of past tokens `(w_1, w_2, ..., w_t)`:

### 1. Token Embedding Lookup
Convert integer token index `w_t` into a continuous dense representation:

```math
e_t = E[w_t] \in \mathbb{R}^{1 \times D}
```

Where `E \in \mathbb{R}^{V \times D}` is the learnable vocabulary embedding matrix (`V`: vocabulary size, `D`: embedding dimension).

---

### 2. Recurrent Context Compression
Update the hidden state memory using a gated recurrent cell (such as the LSTM or GRU from `09-lstm` and `10-gru`):

```math
h_t = \mathrm{RecurrentCell}(e_t, h_{t-1}) \in \mathbb{R}^{1 \times H}
```

The vector `h_t` represents a fixed-size mathematical summary of the **entire historical prefix** `w_{1:t} = (w_1, w_2, ..., w_t)`.

---

### 3. Output Vocabulary Projection (Logits)
Project the recurrent context vector `h_t` across the entire vocabulary of `V` possible candidate words:

```math
o_t = h_t W_{hy} + b_y \in \mathbb{R}^{1 \times V}
```

Where `W_{hy} \in \mathbb{R}^{H \times V}` and `b_y \in \mathbb{R}^{1 \times V}`.

---

### 4. Categorical Probability Distribution
Normalize the unnormalized logits `o_t` into valid non-negative probabilities summing to 1.0 via Softmax:

```math
\hat{y}_{t, k} = P(w_{t+1} = k \mid w_{1:t}) = \frac{\exp(o_{t, k})}{\sum_{j=0}^{V-1} \exp(o_{t, j})}
```

---

### 5. Sequence Cross-Entropy Loss (Negative Log-Likelihood)
At every step `t`, compare the predicted distribution `\hat{y}_t` against the true one-hot target token `Y_t = w_{t+1}^*`:

```math
L_t = - \ln\left(\hat{y}_{t, w_{t+1}^*} + \epsilon\right)
```

The sequence-level loss over an input sequence of length `T` is:

```math
L_{\mathrm{seq}} = \frac{1}{T} \sum_{t=1}^{T} L_t
```

---

## Why Sequence Prediction? (The Statistical Language Modeling Problem)

A language model assigns a joint probability to any arbitrary sequence of words:

```text
P("the blockchain transaction was confirmed") >> P("transaction the confirmed blockchain was")
```

Before neural language models, classical NLP relied on **statistical N-gram models**:

```math
P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-n+1:t-1}) = \frac{\operatorname{Count}(w_{t-n+1:t})}{\operatorname{Count}(w_{t-n+1:t-1})}
```

### Why N-gram Models Failed:

1. **The Curse of Dimensionality**:
   To capture 5 words of context, a 5-gram model over a modest vocabulary of `V = 50,000` words requires tracking `50,000^5 \approx 3.1 \times 10^{23}` frequency table entries—more parameters than all computers on Earth could store.
2. **Zero Generalization on Unseen Contexts**:
   If the exact 4-word phrase `"the decentralized network was"` never appeared in the training corpus, `\operatorname{Count} = 0`. An n-gram model predicts a probability of zero, assigning infinite perplexity to perfectly valid human sentences.
3. **Semantic Blindness**:
   An n-gram model does not know that `"cat"` and `"feline"` are related. Seeing `"the cat was sleeping"` gives it zero statistical predictive capacity for `"the feline was sleeping"`.

### The Neural Breakthrough:
Neural sequence models resolve all three problems:
- Words are embedded into a continuous dense space where synonyms share nearby vectors.
- Context is tracked not by discrete lookup tables, but by an evolving continuous state vector `h_t`.
- Memory scales with the hidden dimension `H`, not exponentially with sequence length `T`.

---

## Training Mode vs. Generation Mode

A critical conceptual pillar of autoregressive sequence modeling is understanding the fundamental difference between **Training** and **Inference**.

```text
====================================================================================================
MODE 1: TRAINING WITH TEACHER FORCING (Synchronous Parallel Training)
====================================================================================================
Inputs (w_t):       "the"          "blockchain"       "transaction"      "was"
                      │                 │                  │               │
                      ▼                 ▼                  ▼               ▼
                   [Cell] ───────────►[Cell] ───────────►[Cell] ────────►[Cell]
                      │                 │                  │               │
                      ▼                 ▼                  ▼               ▼
Predictions:      p(w|the)       p(w|the,block)     p(w|the..trans)    p(w|the..was)
                      │                 │                  │               │
                      ▼                 ▼                  ▼               ▼
Loss Comparison: "blockchain"     "transaction"          "was"        "confirmed"
                 (Target 1)        (Target 2)         (Target 3)       (Target 4)

* Ground truth is fed into EVERY input step, regardless of what the network predicted!

====================================================================================================
MODE 2: AUTOREGRESSIVE GENERATION (Autonomous Free-Running Inference)
====================================================================================================
Step 1:  Prompt: "the" ──► [Cell] ──► Sample: "blockchain"
                                              │
Step 2:                    [Cell] ◄───────────┘ (Self-Generated Input)
                             │
                             └──► Sample: "transaction"
                                           │
Step 3:                                  [Cell] ◄───────────┘
                                           │
                                           └──► Sample: "was"
                                                         │
Step 4:                                                [Cell] ◄───────────┘
                                                         │
                                                         └──► Sample: "confirmed"
                                                                       │
Step 5:                                                              [Cell] ◄─────────┘
                                                                       │
                                                                       └──► Sample: <EOS> (Stop!)
```

---

## Teacher Forcing: Mechanics and Trade-offs

During training, we utilize **Teacher Forcing** (Williams & Zipser, 1989):

```math
x_{t} = w_{t}^* \quad (\text{Ground-truth previous token from training text})
```

### Why Teacher Forcing is Essential:
1. **Prevents Cascading Errors**: Without teacher forcing, if the model erroneously predicts `"banana"` instead of `"blockchain"` at step 1, feeding `"banana"` into step 2 causes the hidden state to derail completely into incoherent nonsense.
2. **Stable Gradients**: Because the model is always fed correct historical contexts, the gradient signal remains clean, consistent, and well-behaved.
3. **Efficient Training**: The entire sequence forward pass can be unrolled across all `T` timesteps in a single pass.

### The Exposure Bias Dilemma:
- During **Training**, the model is spoon-fed 100% correct inputs.
- During **Inference**, the model is exposed to its own imperfect outputs.
- This discrepancy is known as **Exposure Bias**: if the model makes an early mistake during text generation, it has never learned how to recover from an unusual state, which can lead to repetitive loops or hallucinated drift.

---

## Decoding and Sampling Strategies

Once the output projection produces logits `o_t \in \mathbb{R}^V`, how do we choose the next token `w_{t+1}`?

### 1. Greedy Search (Argmax)
Picks the single most probable token at every step:

```math
w_{t+1} = \operatorname{argmax}_{k \in \{0, ..., V-1\}} o_{t, k}
```

- **Pros**: Fast, deterministic, minimizes cross-entropy.
- **Cons**: Severe repetition! Greedy decoding frequently gets trapped in degenerate cycles:
  `"the blockchain was confirmed by the blockchain was confirmed by the blockchain..."`

---

### 2. Temperature-Scaled Softmax Sampling
To inject creative variety, we sample probabilistically from the distribution after scaling the logits by a **temperature hyperparameter** `T_{\mathrm{temp}} > 0`:

```math
P(w_{t+1} = k) = \frac{\exp(o_{t, k} / T_{\mathrm{temp}})}{\sum_{j=0}^{V-1} \exp(o_{t, j} / T_{\mathrm{temp}})}
```

```text
                Distribution Shape Across Temperatures
                
  Probability
      ▲
      │             T = 0.2 (Cold, Sharp, Conservative)
      │               /\
      │              /  \
      │             /    \      T = 1.0 (Standard)
      │            /   .  \   .
      │           /  .     \ .  .
      │          / .         .    .
      │         /.             .    .   T = 2.0 (Hot, Flat, Uniform Random)
      │       .────────────────────────────.
      └───────────────────────────────────────────────► Vocabulary Tokens
```

- **Low Temperature (`T_{\mathrm{temp}} = 0.2 - 0.5`)**: Peaks become much sharper; the model is conservative, confident, and near-deterministic.
- **Normal Temperature (`T_{\mathrm{temp}} = 1.0`)**: Standard sampling according to training statistics.
- **High Temperature (`T_{\mathrm{temp}} = 1.5 - 2.0`)**: Flattens the distribution toward uniform random noise; output is highly diverse, unexpected, and prone to grammatical breakdowns.

---

### 3. Top-K Sampling
Restricts sampling candidates strictly to the `K` most probable tokens, zeroing out the probability mass of the long tail:

```math
V_{\mathrm{top-k}} = \text{Top } K \text{ tokens with highest } o_{t, k}
```

All other logits outside the top `K` are set to `-\infty` before Softmax:

```math
o'_{t, j} = \begin{cases} o_{t, j} & \text{if } j \in V_{\mathrm{top-k}} \\ -\infty & \text{otherwise} \end{cases}
```

- Prevents the model from accidentally sampling completely nonsensical low-probability words.

---

### 4. Top-P (Nucleus) Sampling
Instead of a fixed count `K`, Top-P (Holtzman et al., 2019) dynamically selects the smallest candidate set whose **cumulative probability** exceeds threshold `p \in (0, 1)` (e.g., `p = 0.90`):

```math
\sum_{k \in V^{(p)}} P(w = k) \ge p
```

- **Dynamic Head Expansion**: When the model is uncertain, the candidate pool automatically expands to 20 words.
- **Dynamic Head Contraction**: When the model is 99% certain of the next word (e.g., `"Barack"` ──► `"Obama"`), the candidate pool contracts to just 1 word.

---

## Evaluating Language Models: Perplexity (PPL)

How do we mathematically measure the quality of a sequence prediction model?

### Mathematical Definition of Perplexity
Perplexity is the exponentiated average cross-entropy loss per token:

```math
\mathrm{Perplexity} = \exp\left(\bar{L}\right) = \exp\left( \frac{1}{T} \sum_{t=1}^{T} L_t \right)
```

```math
\mathrm{PPL} = \left( \prod_{t=1}^{T} \frac{1}{P(w_t^* \mid w_{<t})} \right)^{1/T}
```

### Physical Intuition: The Effective Branching Factor
Perplexity represents the **effective number of equally likely choices** the model is confused between at each step:
- **PPL = 1.0**: Absolute perfection (the model predicts every target token with 100% confidence).
- **PPL = 15.0**: At each step, the model is as uncertain as rolling a fair 15-sided die.
- **PPL = V (Vocab Size)**: Total ignorance (equivalent to rolling a fair `V`-sided die uniformly at random).
- **Lower Perplexity = Better Language Model**.

---

## Sequence Topology: Many-to-Many with Causal Shift

In Project 07 (`07-rnn-from-scratch`), we predicted character transitions.
In Project 08 (`08-sentiment-rnn`), we mapped an entire sequence into 1 terminal label (Many-to-One).

In Project 11, we operate in the **Many-to-Many Causal Shift Topology**:

```text
Sequence Length: T = 4
Full Sentence:   "the blockchain transaction was confirmed"

Input Tensor X:  [ "the",        "blockchain",   "transaction", "was"       ]  (Length T)
Target Tensor Y: [ "blockchain", "transaction", "was",          "confirmed" ]  (Length T)
```

The target sequence `Y` is identical to the input sequence `X`, **shifted forward by exactly 1 position**.
Every timestep `t` has:
1. Its own input token: `x_t = w_t`
2. Its own target token: `y_t = w_{t+1}`
3. Its own cross-entropy loss: `L_t = -\ln(\hat{y}_{t, y_t})`
4. Its own backpropagation error: `\delta_{o_t} = \hat{y}_t - Y_t`

---

## Concrete Educational Dataset & Setup

To verify every mathematical derivative and BPTT tensor by hand, we construct a compact, pedagogically clear domain corpus:

### The Mini Blockchain / Smart Contract Corpus

```text
====================================================================================================
TRAINING SEQUENCES
====================================================================================================
1. "the blockchain transaction was confirmed"
2. "the smart contract execution was successful"
3. "the decentralized network was secure"
4. "the consensus protocol was validated"
5. "a block was verified"
====================================================================================================
```

### Vocabulary Setup (`V = 20`, `D = 6`, `H = 8`)

```text
Vocabulary Mapping (V = 20):
  0: <PAD>       5: transaction    10: decentralized   15: secure
  1: <UNK>       6: was            11: network         16: consensus
  2: the         7: confirmed      12: protocol        17: validated
  3: a           8: smart          13: execution       18: verified
  4: blockchain  9: contract       14: successful      19: <EOS>
```

### Parameter Budget

| Parameter | Tensor Shape | Numerical Formula | Parameter Count |
| :--- | :--- | :--- | :--- |
| **Token Embedding `E`** | `(V, D) = (20, 6)` | `20 \times 6` | 120 weights |
| **GRU/LSTM Recurrent Weights** | `W_xzr: (D, 2H), W_hzr: (H, 2H)` | `3 \times (6\times 8 + 8\times 8 + 8)` | 360 weights |
| **Output Projection `W_hy`** | `(H, V) = (8, 20)` | `8 \times 20` | 160 weights |
| **Output Bias `b_y`** | `(1, V) = (1, 20)` | `1 \times 20` | 20 biases |
| **TOTAL** | — | — | **660 learnable parameters** |

---

## Parameters and Tensor Shapes Reference

Let:
- `B`: Batch size (number of parallel sequences).
- `T`: Sequence length (number of unrolled steps).
- `V`: Vocabulary size.
- `D`: Dense word embedding dimension.
- `H`: Recurrent hidden state dimension.

| Tensor | Shape | Mathematical Symbol | Description |
| :--- | :--- | :--- | :--- |
| `X_tokens` | `(B, T)` | `X \in \{0, ..., V-1\}^{B \times T}` | Input batch of integer token IDs |
| `Y_targets` | `(B, T)` | `Y \in \{0, ..., V-1\}^{B \times T}` | Target batch of next token IDs (shifted by 1) |
| `E` | `(V, D)` | `E \in \mathbb{R}^{V \times D}` | Learnable vocabulary embedding table |
| `X_embed` | `(B, T, D)` | `X_e \in \mathbb{R}^{B \times T \times D}` | Dense continuous embedded sequence batch |
| `e_t` | `(B, D)` | `e_t \in \mathbb{R}^{B \times D}` | Embedding slice at timestep `t` |
| `h_t` | `(B, H)` | `h_t \in \mathbb{R}^{B \times H}` | Recurrent hidden state summarizing context `w_{1:t}` |
| `W_{hy}` | `(H, V)` | `W_{hy} \in \mathbb{R}^{H \times V}` | Hidden-to-vocabulary projection matrix |
| `b_y` | `(1, V)` | `b_y \in \mathbb{R}^{1 \times V}` | Vocabulary classification bias offset |
| `logits` | `(B, T, V)` | `O \in \mathbb{R}^{B \times T \times V}` | Raw unnormalized vocabulary prediction logits |
| `probs` | `(B, T, V)` | `\hat{Y} \in (0, 1)^{B \times T \times V}` | Normalized next-token probability distributions |
| `loss` | Scalar | `L_{\mathrm{seq}} \in \mathbb{R}^+` | Sequence negative log-likelihood loss |
| `perplexity`| Scalar | `\mathrm{PPL} \ge 1.0` | Geometric mean prediction uncertainty metric |

---

## Project Structure

```text
11-sequence-prediction/
├── README.md                      # Language modeling foundations, decoding strategies, 101 syllabus
├── src/
│   └── sequence_model.py          # Pure NumPy Tokenizer, Embedding, Recurrent LM, & Generator
└── steps/
    └── step-01-mathematics.md      # Causal shift calculus, cross-entropy derivation, sampling proofs
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Pending | `steps/step-01-mathematics.md`: Joint probability chain rule, causal cross-entropy, sampling derivations |
| **2. NumPy Implementation** | Pending | `src/sequence_model.py`: Vectorized `TeacherForcingForward`, `BPTT`, `generate()` |
| **3. Understand Forward Pass** | Pending | Step-by-step tensor unrolling verifying output distributions `P(w_{t+1}|w_{1:t})` |
| **4. Understand Loss** | Pending | Negative log-likelihood computation, masking `<PAD>` tokens, and perplexity tracking |
| **5. Derive Gradients** | Pending | Matrix calculus for vocabulary output projection `dW_hy, db_y` and recurrent accumulation |
| **6. Implement Backpropagation** | Pending | Synchronized BPTT across all causal steps with embedding table gradient accumulation |
| **7. Train Model** | Pending | Overfitting on mini blockchain corpus until PPL drops below 1.5 and sentences generate cleanly |
| **8. Debug & Analyze** | Pending | Evaluating Greedy vs Temperature vs Top-K vs Top-P generation on seed prompts |
| **9. PyTorch Implementation** | Pending | Equivalent model using `nn.Embedding`, `nn.GRU`, `nn.Linear`, `nn.CrossEntropyLoss` |
| **10. Compare Results** | Pending | Exact parity validation of cross-entropy loss, perplexity, and generated token sequences |

---

## Complete 101-Topic Foundational Curriculum

This project systematically covers the following 101 core deep learning topics across 16 structured modules:

```text
PROBABILISTIC FOUNDATIONS OF LANGUAGE
│
├── 01. What is a language model?
├── 02. Joint probability of a word sequence
├── 03. The chain rule of probability for text
├── 04. P(w_1, ..., w_T) = prod P(w_t | w_{<t})
├── 05. Conditional language modeling
├── 06. Classical N-gram language models
├── 07. The curse of dimensionality in N-grams
├── 08. Sparsity and the zero-frequency problem
├── 09. Smoothing techniques (Laplace, Good-Turing, Kneser-Ney)
│
▼
THE NEURAL LANGUAGE MODELING PARADIGM
│
├── 10. Bengio et al. (2003) neural breakthrough
├── 11. Distributed representations of words
├── 12. Continuous hidden states vs discrete count tables
├── 13. Sharing statistical strength across contexts
├── 14. Vocabulary construction and special tokens (<PAD>, <UNK>, <BOS>, <EOS>)
├── 15. The role of <BOS> (Beginning of Sequence prompt)
├── 16. The role of <EOS> (Stopping criterion in generation)
│
▼
DATASET PIPELINE & CAUSAL SHIFTING
│
├── 17. The Causal Offset: Target = Input shifted by 1
├── 18. Creating (X, Y) training pairs from text corpora
├── 19. Context window length selection (T)
├── 20. Sliding window sequence chunking
├── 21. Sequence batching and post-padding with <PAD>
├── 22. Target masking (ignoring <PAD> in cross-entropy loss)
│
▼
EMBEDDING LAYER MECHANICS
│
├── 23. Embedding table representation E in R^(V x D)
├── 24. Embedding dimension tradeoffs (D = 16 vs 64 vs 256)
├── 25. Forward embedding lookup
├── 26. Backward pass into embedding weights (dE)
├── 27. Weight tying (tying input embeddings E and output weights W_hy.T)
│
▼
RECURRENT BACKBONE ENCODING
│
├── 28. Selecting the recurrent backbone (Simple RNN vs GRU vs LSTM)
├── 29. Why Simple RNN fails on sequence prediction (vanishing gradient)
├── 30. Why GRU / LSTM enables long-range grammatical memory
├── 31. Tracking subject-verb agreement across long clauses
├── 32. Hidden state statefulness across batches (Truncated BPTT overview)
│
▼
OUTPUT PROJECTION & SOFTMAX
│
├── 33. Output classification projection W_hy in R^(H x V)
├── 34. Output bias offset b_y in R^(1 x V)
├── 35. Vocabulary logit computation: o_t = h_t @ W_hy + b_y
├── 36. Numerically stable Softmax with max-subtraction
├── 37. Probability normalization over vocabulary V
├── 38. The computational cost of large vocabulary Softmax (O(V))
│
▼
LOSS EVALUATION & METRICS
│
├── 39. Categorical cross-entropy per timestep
├── 40. Negative log-likelihood (NLL) formula
├── 41. Sequence-averaged loss
├── 42. Mathematical definition of Perplexity (PPL)
├── 43. PPL = exp(mean_loss)
├── 44. Interpreting perplexity as branching factor
├── 45. Cross-entropy vs perplexity benchmarks
│
▼
TEACHER FORCING & TRAINING
│
├── 46. What is Teacher Forcing?
├── 47. Why Teacher Forcing stabilizes training
├── 48. Causal forward unrolling across all T timesteps
├── 49. Exposure bias: The fundamental flaw of Teacher Forcing
├── 50. Scheduled sampling (Bengio et al., 2015)
├── 51. Curriculum learning in sequence prediction
│
▼
BACKPROPAGATION THROUGH TIME (CAUSAL SEQUENCE)
│
├── 52. Computational graph of unrolled sequence prediction
├── 53. Output error at each timestep: delta_o_t = p_t - Y_t
├── 54. Accumulating output weight gradients: dW_hy = sum(h_t.T @ delta_o_t)
├── 55. Accumulating output bias gradients: db_y = sum(delta_o_t)
├── 56. Dual error injection into recurrent state h_t:
├── 57. Direct error from current prediction: delta_o_t @ W_hy.T
├── 58. Future error from next recurrent step: delta_h_{next}
├── 59. Recurrent cell backward pass (LSTM / GRU)
├── 60. Accumulating embedding gradients: dE[w_t]
│
▼
AUTOREGRESSIVE GENERATION (INFERENCE LOOP)
│
├── 61. The autoregressive inference loop algorithm
├── 62. Prompt priming: Processing the initial user prompt
├── 63. State preservation across prompt tokens
├── 64. Step-by-step sequential generation
├── 65. Stopping criteria: <EOS> detection vs max_tokens limit
├── 66. Detokenization: Converting token IDs back to human text
│
▼
DECODING STRATEGY 1: GREEDY SEARCH
│
├── 67. The greedy decision rule: w_{t+1} = argmax(o_t)
├── 68. Properties of greedy decoding (deterministic, fast)
├── 69. Degenerate repetition failure modes
├── 70. Why greedy search produces dull, generic text
│
▼
DECODING STRATEGY 2: TEMPERATURE SAMPLING
│
├── 71. The temperature scaling formula: o_t / T_temp
├── 72. Mathematical effect of T_temp -> 0 (Dirac delta / argmax)
├── 73. Mathematical effect of T_temp = 1.0 (true model distribution)
├── 74. Mathematical effect of T_temp -> infinity (uniform noise)
├── 75. Multinomial / Categorical sampling implementation in NumPy
├── 76. Temperature tuning for creative vs factual generation
│
▼
DECODING STRATEGY 3: TOP-K SAMPLING
│
├── 77. The long-tail problem in raw Softmax sampling
├── 78. Top-K filtering algorithm
├── 79. Masking non-top-K logits with -infinity
├── 80. Re-normalizing probabilities over top-K candidates
├── 81. Hyperparameter selection for K (e.g., K = 10, 40, 50)
│
▼
DECODING STRATEGY 4: TOP-P (NUCLEUS) SAMPLING
│
├── 82. The limitation of fixed-K sampling
├── 83. Nucleus sampling concept (Holtzman et al., 2019)
├── 84. Sorting logits and cumulative probability summation
├── 85. Dynamic threshold cutoff at cumulative sum >= p
├── 86. Truncating the tail and re-normalizing
├── 87. Combining Top-K and Top-P (industry standard)
│
▼
ADVANCED GENERATION TECHNIQUES
│
├── 88. Repetition penalties (Keskar et al., 2019)
├── 89. Frequency and presence penalties
├── 90. Beam search decoding fundamentals
├── 91. Beam width (B) and hypothesis tracking
├── 92. Length normalization in beam search
│
▼
EVALUATION & BENCHMARKING
│
├── 93. Evaluating generated text fluency and coherence
├── 94. Train vs validation perplexity curves
├── 95. Detecting overfitting via validation perplexity divergence
├── 96. PyTorch parity: nn.Embedding + nn.GRU + nn.CrossEntropyLoss
├── 97. Matching NumPy and PyTorch loss curves exactly
│
▼
THE BRIDGE TO MODERN LLMS
│
├── 98. From Recurrent LM to Transformer LM (GPT architecture preview)
├── 99. Why Transformers replaced recurrent models (parallel training)
├── 100. How modern LLMs still use the exact same next-token objective
└── 101. Complete production-grade Autoregressive Sequence Model from scratch
```

---

## Key Takeaways

1. **The Universal LLM Pre-training Objective**:
   Whether using a simple RNN, an LSTM, a GRU, or a 100-billion parameter Transformer, the foundational training goal is identical: predict the immediate next token `w_{t+1}` given historical context `w_{1:t}`.
2. **Teacher Forcing is Essential for Stable Learning**:
   Training feeds the true target as the next input, preventing early mispredictions from derailing training into chaotic error cascades.
3. **Autoregressive Generation Feeds on Itself**:
   At test time, the model runs free, feeding its own sampled outputs back as future inputs until it generates the `<EOS>` stopping token.
4. **Sampling Shapes Output Character**:
   - Greedy search is fast but prone to repetitive loops.
   - Temperature controls entropy (creative vs conservative).
   - Top-K and Top-P (Nucleus) filter out the improbable tail, eliminating hallucinated gibberish.
5. **Perplexity Quantifies Model Uncertainty**:
   Perplexity measures the effective branching uncertainty of the language model; lower perplexity directly reflects superior sequence predictive mastery.

---

## Next Steps

1. Create [`steps/step-01-mathematics.md`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/11-sequence-prediction/steps/step-01-mathematics.md) detailing the joint probability chain rule, causal shift matrix calculus, Softmax cross-entropy gradients, and sampling proofs.
2. Implement the complete pure NumPy language model in [`src/sequence_model.py`](file:///Users/pasan/Documents/Personal/deep-learning-foundations/11-sequence-prediction/src/sequence_model.py) featuring `Tokenizer`, `Embedding`, `RecurrentLM`, Teacher Forcing, and Temperature/Top-P generation.
