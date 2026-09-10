# Deep Learning Foundations: From Scratch to Transformers & LLMs

A rigorous, first-principles journey through modern deep learning. This repository builds every foundational neural network architecture from scratch using pure **NumPy** (`import numpy as np`), derives every matrix calculus formula by hand, validates numerical precision down to 6 decimal places, and benchmarks implementations against production **PyTorch**.

The curriculum spans the complete historical and architectural evolution of deep learning—from a single biological-inspired artificial neuron to multi-layer perceptrons, computer vision convolutions, sequential recurrent memory, and modern autoregressive Large Language Models (LLMs).

```text
========================================================================================================================
                                      THE ARCHITECTURAL EVOLUTION OF DEEP LEARNING
========================================================================================================================

  ERA 1: MULTI-LAYER PERCEPTRONS (ANN)
  [ Single Neuron ] ──► [ Dense Layer (ANN) ] ──► [ MNIST Digits (ANN) ] ──► [ Tabular Churn Classification ]
        (01)                     (02)                       (03)                             (04)
         │
         ▼
  ERA 2: SPATIAL COMPUTATION (CNN)
  [ 2D Convolution & Pooling ] ──────────► [ Traffic Sign Vision (CNN) ]
             (05)                                       (06)
         │
         ▼
  ERA 3: TEMPORAL & SEQUENTIAL MEMORY (RNN, LSTM, GRU)
  [ Simple RNN (Many-to-Many) ] ──► [ Sentiment Analysis (Many-to-One) ] ──► [ LSTM (CEC Memory Highway) ]
               (07)                                   (08)                                   (09)
                                                                                               │
  [ Autoregressive Sequence Modeling ] ◄── [ Gated Recurrent Unit (GRU) ] ◄────────────────────┘
                 (11)                                    (10)
         │
         ▼
  ERA 4: ATTENTION & TRANSFORMER REVOLUTION
  [ Attention Mechanisms (Q, K, V) ] ──► [ Mini-Transformer (Encoder-Decoder) ] ──► [ Mini-LLM (Decoder-Only GPT) ]
                 (12)                                        (13)                                    (14)
========================================================================================================================
```

---

## Core Philosophy

1. **Zero Black-Box Magic**: Every layer (`Dense`, `Conv2D`, `MaxPool`, `AvgPool`, `MinPool`, `RNN`, `LSTM`, `GRU`, `Attention`, `TransformerBlock`) is written using raw linear algebra operations. No TensorFlow, PyTorch, Keras, or Scikit-Learn during the from-scratch phase.
2. **First-Principles Mathematics**: Before writing code, every forward transformation, probability distribution, objective function, and backpropagation gradient is derived on paper using multivariate matrix calculus.
3. **Automated Assertion Testing**: Unit tests assert that initial forward activations and backward gradients match hand-calculated derivations down to 6 decimal places.
4. **Standard 10-Step Workflow**: Every project adheres to the exact same rigorous learning sequence.
5. **PyTorch Verification**: Equivalent models are built with `torch.nn` to compare parameter layouts, verify gradient convergence, and benchmark computational efficiency.

---

## The Standard 10-Step Learning Workflow

Every project in this repository follows this structured 10-phase sequence:

```text
 1. Mathematics ──► 2. NumPy Implementation ──► 3. Understand Forward Pass ──► 4. Understand Loss
                                                                                       │
 8. Debug & Analyze ◄── 7. Train Model ◄── 6. Backpropagation ◄── 5. Derive Gradients ◄┘
         │
         ▼
 9. PyTorch Parity ──► 10. Compare Results & Benchmark
```

| Phase | Core Objective | Key Deliverables & Validation Criteria |
| :--- | :--- | :--- |
| **1. Mathematics** | Theory & Proofs | Formulate scalar and matrix calculus equations, activation properties, and tensor dimensions in `steps/step-01-mathematics.md`. |
| **2. NumPy Implementation** | Vectorized Code | Implement modular forward and backward primitives in `src/` using efficient NumPy vectorization. |
| **3. Understand Forward Pass** | Tensor Tracing | Walk through intermediate activations, linear projections, and non-linearities across every layer and timestep. |
| **4. Understand Loss** | Objective Evaluation | Formulate cross-entropy, binary cross-entropy, or MSE loss with numerical stability guards (`\epsilon` clipping). |
| **5. Derive Gradients** | Matrix Calculus | Compute partial derivatives with respect to all learnable weights, biases, and intermediate activations using the chain rule. |
| **6. Implement Backpropagation** | Gradient Flow | Program the backward pass, verifying gradient accumulation, weight tying, and parameter updates. |
| **7. Train Model** | Convergence | Train the model from scratch on concrete benchmark datasets (XOR, MNIST, GTSRB, Movie Sentiment, Language Corpus). |
| **8. Debug & Analyze** | Failure Diagnostics | Analyze training failure modes (vanishing/exploding gradients, dead ReLUs, saturation, exposure bias, padding drift). |
| **9. PyTorch Implementation** | Framework Parity | Implement the exact same model using production PyTorch (`torch.nn.Module`, `torch.optim`). |
| **10. Compare Results** | Parity Verification | Copy weights between NumPy and PyTorch to verify identical forward outputs, loss curves, and benchmark execution time. |

---

## Complete Project Directory & Curriculum

```text
deep-learning-foundations/
├── 01-neuron/                      # Single Artificial Neuron (Perceptron, Sigmoid, Binary Classification)
├── 02-ann-from-scratch/            # 2-Layer Artificial Neural Network (XOR problem, ReLU, Softmax, BPTT)
├── 03-mnist-ann/                   # Real-World Digit Recognition with Deep Multi-Layer Perceptrons
├── 04-customer-churn/              # Tabular Enterprise Deep Learning & Class Imbalance
├── 05-cnn-from-scratch/            # 2D Convolution, Strides, Padding, Max/Min/Avg Pooling from scratch
├── 06-traffic-sign-cnn/            # Multi-class Computer Vision on German Traffic Signs (GTSRB)
├── 07-rnn-from-scratch/            # Simple RNN, Many-to-Many Recurrence, "hello" Next-Char Prediction
├── 08-sentiment-rnn/               # Word Embeddings, Tokenization, Sequence Padding, Many-to-One NLP
├── 09-lstm/                        # Long Short-Term Memory, Constant Error Carousel, 3 Fused Gates
├── 10-gru/                         # Gated Recurrent Unit, Reset & Update Gates, Linear State Highway
├── 11-sequence-prediction/         # Autoregressive Language Modeling, Teacher Forcing, Top-P Sampling
├── 12-attention/                   # Bahdanau & Luong Attention Mechanisms, Dynamic Alignment, Q/K/V
├── 13-mini-transformer/            # Multi-Head Self-Attention, Positional Encoding, Encoder-Decoder
├── 14-mini-llm/                    # Decoder-Only Autoregressive LLM (GPT Architecture, KV Cache)
│
├── Simple MNIST NN from scratch/   # Standalone Pure-NumPy 2-Layer MNIST Neural Network Notebook
└── Transformer Arch/               # Foundational Transformer Research Papers & Architectural Schematics
```

---

## Comprehensive Project Summaries

### Part 1: Neurons & Multi-Layer Perceptrons (Dense Architectures)

#### [01 — Single Neuron](01-neuron/README.md)
- **Problem**: Binary linear classification (AND, OR logic gates) and the limits of linear separation.
- **Key Concepts**: Weighted sum $z = X w + b$, logistic Sigmoid activation, logit space, Binary Cross-Entropy (BCE) loss, scalar and vectorized gradient descent.
- **Why It Matters**: Proves that a single linear hyperplane cannot separate non-linear data (the famous Minsky & Papert XOR limitation).

#### [02 — ANN From Scratch](02-ann-from-scratch/README.md)
- **Problem**: Solving the non-linear XOR classification problem that defeated the single neuron.
- **Key Concepts**: Hidden layer feature transformation, ReLU activation, Softmax normalization, Categorical Cross-Entropy, 2-layer backpropagation matrix calculus ($dW_2, db_2, dW_1, db_1$).
- **Why It Matters**: Demonstrates how non-linear hidden representations fold feature space to make non-linearly separable problems linearly separable.

#### [03 — MNIST ANN](03-mnist-ann/README.md)
- **Problem**: Classifying 28 × 28 grayscale handwritten digits (0–9) on the classic MNIST dataset.
- **Key Concepts**: High-dimensional vector flattening (784 inputs), mini-batch gradient descent, Xavier/He weight initialization, learning rate annealing, multi-class evaluation metrics.
- **Why It Matters**: Transitions from 2D toy coordinates to high-dimensional real-world image classification with deep multi-layer perceptrons.

#### [04 — Customer Churn Classification](04-customer-churn/README.md)
- **Problem**: Predicting enterprise customer churn on tabular business data.
- **Key Concepts**: Tabular preprocessing, one-hot categorical encoding, numerical standard scaling, addressing severe class imbalance (weighted cross-entropy, focal loss), Precision, Recall, ROC-AUC.
- **Why It Matters**: Bridges machine learning theory to industry tabular deep learning workflows.

---

### Part 2: Spatial Representation Learning (Convolutional Neural Networks)

#### [05 — CNN From Scratch](05-cnn-from-scratch/README.md)
- **Problem**: Overcoming the parameter explosion and spatial destruction of ANNs on 2D images.
- **Key Concepts**: 2D discrete convolution vs cross-correlation, receptive fields, parameter sharing, translation equivariance, multi-channel 4D tensors $(B, C, H, W)$, stride arithmetic, valid vs same zero-padding, the complete **Pooling Suite** (Max, Min, and Average Pooling), spatial backpropagation with rotated kernels (`rot180`).
- **Implementation**: Available in [`05-cnn-from-scratch/src/cnn.py`](05-cnn-from-scratch/src/cnn.py) with full derivations in [`05-cnn-from-scratch/steps/step-01-mathematics.md`](05-cnn-from-scratch/steps/step-01-mathematics.md).
- **Why It Matters**: The foundational mathematical engine powering all modern computer vision.

#### [06 — Traffic Sign Recognition CNN](06-traffic-sign-cnn/README.md)
- **Problem**: Multi-class visual classification on real-world German Traffic Sign Recognition Benchmark (GTSRB).
- **Key Concepts**: Deep convolutional hierarchies (edges -> textures -> shapes -> signs), data augmentation (rotation, jitter, scaling), dropout regularization, batch normalization.
- **Why It Matters**: Real-world safety-critical autonomous driving vision using convolutional networks.

---

### Part 3: Sequential & Temporal Representation Learning (Recurrent Architectures)

#### [07 — RNN From Scratch](07-rnn-from-scratch/README.md)
- **Problem**: Modeling sequential data where temporal order dictates meaning; resolving the **Collision Paradox** (`'l' -> 'l'` vs `'l' -> 'o'` on identical inputs).
- **Key Concepts**: Recurrent hidden state memory $h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$, parameter sharing across time, unrolling the computational graph, Backpropagation Through Time (BPTT), sequence categorical cross-entropy, character-level language modeling on `"hello"`.
- **Implementation**: Available in [`07-rnn-from-scratch/src/rnn.py`](07-rnn-from-scratch/src/rnn.py) with full derivations in [`07-rnn-from-scratch/steps/step-01-mathematics.md`](07-rnn-from-scratch/steps/step-01-mathematics.md).

#### [08 — Sentiment Analysis with RNN](08-sentiment-rnn/README.md)
- **Problem**: Natural language sentiment classification on variable-length text sentences; resolving negation flipping (`"not good"` vs `"good"`).
- **Key Concepts**: Text normalization, word tokenization, vocabulary mapping (`<PAD>=0`, `<UNK>=1`), dense word embeddings ($E$), sequence padding & truncation, dynamic length indexing, hidden state masking, **Many-to-One** sequence topology, terminal BCE loss.
- **Implementation**: Available in [`08-sentiment-rnn/src/sentiment_rnn.py`](08-sentiment-rnn/src/sentiment_rnn.py) with full derivations in [`08-sentiment-rnn/steps/step-01-mathematics.md`](08-sentiment-rnn/steps/step-01-mathematics.md).
- **Why It Matters**: The bridge from character-level toy transitions to real-world natural language processing.

#### [09 — Long Short-Term Memory (LSTM)](09-lstm/README.md)
- **Problem**: Resolving the catastrophic vanishing gradient failure of simple RNNs over extended sequences ($T > 10$).
- **Key Concepts**: Two-track dynamical memory (Long-Term Cell State $C_t$ and Working Hidden State $h_t$), mathematical proof of the **Constant Error Carousel (CEC)**, Forget Gate ($f_t$), Input Gate ($i_t$), Candidate State ($\tilde{C}_t$), Output Gate ($o_t$), fused 4-gate matrix vectorization $W_x \in \mathbb{R}^{D \times 4H}$, long-range bit memory benchmark ($T = 20$: RNN 53.3% vs. LSTM 100.0%).
- **Implementation**: Available in [`09-lstm/src/lstm.py`](09-lstm/src/lstm.py) with full derivations in [`09-lstm/steps/step-01-mathematics.md`](09-lstm/steps/step-01-mathematics.md).
- **Why It Matters**: The breakthrough architecture that unlocked speech recognition, translation, and modern sequence modeling.

#### [10 — Gated Recurrent Unit (GRU)](10-gru/README.md)
- **Problem**: Reducing the computational and parameter overhead of LSTMs while retaining vanishing gradient immunity.
- **Key Concepts**: Reset-before Cho et al. (2014) formulation, single hidden state ($h_t$), coupled gating (Reset Gate $r_t$ and Update Gate $z_t$), the linear state interpolation gradient highway $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$, 4-way BPTT error propagation, batch-mean gradient scaling, and deterministic memory benchmarking.
- **Implementation**: Available in [`10-gru/src/gru.py`](10-gru/src/gru.py) with full derivations in [`10-gru/steps/step-01-mathematics.md`](10-gru/steps/step-01-mathematics.md).
- **Why It Matters**: 25% fewer parameters, faster training, smaller memory footprint, and competitive or superior performance on small-to-medium corpora.

#### [11 — Sequence Prediction & Autoregressive Modeling](11-sequence-prediction/README.md)
- **Problem**: Learning the universal generative objective underlying all modern Large Language Models: Next-Token Prediction under the probabilistic chain rule.
- **Key Concepts**: Causal sequence shifting topology ($Y = X$ shifted by $+1$), dense vocabulary embedding matrix $E$ with sparse BPTT accumulation (`np.add.at`), Teacher Forcing vs. free-running inference, complete decoding suite (**Greedy Argmax, Temperature Scaling, Top-K, Top-P Nucleus Sampling**), Perplexity evaluation metric ($\text{PPL} = \exp(\bar{L})$).
- **Implementation**: Available in [`11-sequence-prediction/src/sequence_model.py`](11-sequence-prediction/src/sequence_model.py) with full derivations in [`11-sequence-prediction/steps/step-01-mathematics.md`](11-sequence-prediction/steps/step-01-mathematics.md).
- **Why It Matters**: The exact pre-training and generation mechanics used by GPT, Claude, and Gemini.

---

### Part 4: Attention & The Modern Transformer Revolution

#### [12 — Attention Mechanisms](12-attention/README.md)
- **Problem**: Eliminating the fixed-vector information bottleneck of recurrent encoder-decoder models.
- **Key Concepts**: Bahdanau additive attention, Luong multiplicative dot-product attention, Query-Key-Value ($Q, K, V$) formulations, dynamic alignment scores, attention weight heatmaps, context vector computation $c_i = \sum \alpha_{ij} h_j$.
- **Why It Matters**: The conceptual foundation that paved the way for Transformers by replacing recurrent memory with direct associative retrieval.

#### [13 — Mini-Transformer](13-mini-transformer/README.md)
- **Problem**: Abolishing recurrent sequential bottlenecks to allow massive $O(1)$ parallel sequence training across time.
- **Key Concepts**: Scaled Dot-Product Attention, Multi-Head Attention (MHA), Sinusoidal Positional Encoding, Layer Normalization (Pre-LN vs Post-LN), Residual Connections, Position-wise Feed-Forward Networks (FFN).
- **Why It Matters**: First-principles implementation of the definitive architecture defined in *"Attention Is All You Need"* (Vaswani et al., 2017).

#### [14 — Mini-LLM (Decoder-Only GPT)](14-mini-llm/README.md)
- **Problem**: Building a complete, standalone generative Large Language Model from scratch.
- **Key Concepts**: Decoder-only architecture, causal self-attention masking (preventing future token leakage), pre-training on natural text, prompt generation engine, Key-Value (KV) caching for fast autoregressive inference.
- **Why It Matters**: The capstone project synthesizing all 14 stages into a functioning, miniature generative language model.

---

## Special & Foundational Reference Assets

### 1. Simple MNIST NN From Scratch
**Location**: [`Simple MNIST NN from scratch/`](Simple%20MNIST%20NN%20from%20scratch/)

A self-contained, interactive Jupyter notebook (`simple-mnist-nn-from-scratch-numpy-no-tf-keras.ipynb`) providing a visual, end-to-end implementation of a 2-layer neural network trained on the MNIST handwritten digit dataset directly with pure NumPy:

```text
Input Layer A[0] (784 features) ──► Hidden Layer A[1] (10 neurons, ReLU) ──► Output Layer A[2] (10 classes, Softmax)
```

- **Mathematical Engine**:
  - Forward pass:
    ```math
    Z^{[1]} = W^{[1]} X + b^{[1]}, \quad A^{[1]} = \operatorname{ReLU}(Z^{[1]})
    ```
    ```math
    Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}, \quad A^{[2]} = \operatorname{softmax}(Z^{[2]})
    ```
  - Backward pass:
    ```math
    dZ^{[2]} = A^{[2]} - Y, \quad dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}, \quad db^{[2]} = \frac{1}{m} \sum dZ^{[2]}
    ```
    ```math
    dZ^{[1]} = W^{[2]T} dZ^{[2]} \odot \mathbf{1}(Z^{[1]} > 0), \quad dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T, \quad db^{[1]} = \frac{1}{m} \sum dZ^{[1]}
    ```
- **Results**: Achieves **~85% accuracy** on 41,000 training images in 500 gradient descent iterations using pure vectorization, complete with Matplotlib digit visualization and test set evaluation (84.4%).
- **Role**: A standalone interactive sandbox demonstrating how minimal pure NumPy code can classify real-world image datasets.

---

### 2. Transformer Architecture Reference Papers & Diagrams
**Location**: [`Transformer Arch/`](Transformer%20Arch/)

This directory preserves the foundational research papers and visual engineering guides that define the theoretical blueprint for Projects 12, 13, and 14:

1. **`1706.03762v7.pdf` — *"Attention Is All You Need"***:
   - The seminal 2017 research paper by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (Google Brain & Google Research).
   - Establishes the complete elimination of recurrence and convolutions in favor of self-attention.
   - Derives the Scaled Dot-Product Attention formula:
     ```math
     \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
     ```
   - Defines the multi-head projection mechanism, sinusoidal positional encoding frequencies, and layer-normalized residual connections.

2. **`Diagrams_V2.pdf` — Visual Architecture Guide & Tensor Schematics**:
   - High-resolution architectural flowcharts detailing:
     - Exact tensor transformations through the Encoder and Decoder stacks.
     - Multi-Head Attention splitting, scaling, and concatenation pathways.
     - Additive residual connections and LayerNorm stabilization.
     - Feed-forward projection expansions ($d_{\text{model}} \to d_{ff} \to d_{\text{model}}$).

---

## Architectural Progression Matrix

| Architecture | Project | Core Innovation | Replaces / Improves | Limitations Overcome |
| :--- | :--- | :--- | :--- | :--- |
| **Single Neuron** | `01` | Scalar weighted sum + activation | Hand-crafted rules | First step in learning parameters from data |
| **Multi-Layer ANN** | `02`, `03`, `04` | Non-linear hidden layers + backprop | Single Neuron | Solves non-linear XOR and complex boundaries |
| **CNN** | `05`, `06` | Receptive fields + parameter sharing | ANN on images | Solves parameter explosion and spatial destruction |
| **Simple RNN** | `07` | Recurrent hidden state $h_t$ | ANN on sequences | Retains temporal order; solves sequential collision |
| **Sentiment RNN** | `08` | Word embeddings + Many-to-One | Bag-of-Words | Solves negation inversion and one-hot sparsity |
| **LSTM** | `09` | Constant Error Carousel ($C_t$) + 3 gates | Simple RNN | Solves vanishing gradient over long sequences ($T > 10$) |
| **GRU** | `10` | Coupled gating + single state ($h_t$) | LSTM | 25% fewer parameters; faster training and lower memory |
| **Autoregressive LM** | `11` | Causal shifting + Teacher Forcing | N-gram models | Predicts probability of any arbitrary word sequence |
| **Attention** | `12` | Associative $Q, K, V$ alignment | Fixed-vector bottleneck | Direct $O(1)$ memory access across long sequences |
| **Transformer** | `13` | Multi-Head Self-Attention + PosEnc | Recurrent models | Unlocks $O(1)$ parallel training across time |
| **Decoder-Only LLM** | `14` | Causal masked self-attention + KV cache | Autoregressive RNNs | Scalable foundation of modern generative AI |

---

## Environment Setup & Quickstart

### Prerequisites
- Python 3.8+
- Pure NumPy (`numpy`)
- Matplotlib (`matplotlib` for visualizations)
- PyTorch (`torch` for framework parity validation)

### Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/madhurangafcx/open-dl.git deep-learning-foundations
cd deep-learning-foundations

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy matplotlib torch
```

### Running Verification Tests

Every completed project contains self-contained executable assertion tests validating forward passes, loss computations, and BPTT matrix calculus against analytical and finite-difference derivations:

```bash
# Verify 2D Convolution & Pooling Suite
python3 05-cnn-from-scratch/src/cnn.py

# Verify Simple RNN on "hello" Next-Char Prediction
python3 07-rnn-from-scratch/src/rnn.py

# Verify Sentiment RNN Many-to-One Classification
python3 08-sentiment-rnn/src/sentiment_rnn.py

# Verify LSTM from Scratch & Constant Error Carousel (CEC)
python3 09-lstm/src/lstm.py

# Verify GRU from Scratch & Linear Gradient Highway
python3 10-gru/src/gru.py

# Verify Autoregressive Sequence Model & Sampling Suite
python3 11-sequence-prediction/src/sequence_model.py
```

To run the interactive standalone MNIST notebook:

```bash
jupyter notebook "Simple MNIST NN from scratch/simple-mnist-nn-from-scratch-numpy-no-tf-keras.ipynb"
```

---

## Mathematical Notation Reference

Across all documents in this repository, mathematical notation strictly adheres to the following conventions:

| Symbol | Meaning | Example Dimensions |
| :--- | :--- | :--- |
| `B` | Batch size (number of parallel training samples) | Scalar (`B = 32`) |
| `T` | Sequence length (number of unrolled temporal timesteps) | Scalar (`T = 4` or `T = 20`) |
| `D` | Feature / Embedding dimension | Scalar (`D = 4` or `D = 64`) |
| `H` | Hidden state capacity / Number of recurrent neurons | Scalar (`H = 3` or `H = 128`) |
| `V` | Vocabulary size | Scalar (`V = 20` or `V = 10,000`) |
| `K` | Number of output classification classes | Scalar (`K = 2` or `K = 10`) |
| `\odot` | Element-wise Hadamard matrix product | $A \odot B$ |
| `@` | Matrix dot product multiplication | $X @ W$ |
| `\sigma(z)` | Logistic Sigmoid function $1 / (1 + \exp(-z))$ | Range $(0, 1)$ |
| `\tanh(z)` | Hyperbolic Tangent function $(\exp(z) - \exp(-z)) / (\exp(z) + \exp(-z))$ | Range $(-1, +1)$ |

---

## Author & Citation

Created by **Pasan Madhuranga** ([@madhurangafcx](https://github.com/madhurangafcx)) as an open educational foundation for deep learning engineers, researchers, and students.

If you find this repository helpful in your machine learning journey, please star the repository on GitHub!
