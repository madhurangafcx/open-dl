# Deep Learning Foundations: From Scratch to LLMs

A first-principles journey through modern deep learning. Every foundational architecture is built from scratch in pure **NumPy** (`import numpy as np`), mathematically verified against analytical gradients down to 6 decimal places, and benchmarked against **PyTorch**.

```text
[01-04 ANN]  Perceptron ──► Dense Layer ──► MNIST Digits ──► Tabular Churn
      │
[05-06 CNN]  2D Convolution & Pooling ──► Traffic Sign Vision
      │
[07-11 RNN]  Simple RNN ──► Sentiment RNN ──► LSTM (CEC) ──► GRU ──► Sequence Model
      │
[12-14 TRF]  Attention (Q, K, V) ──► Mini-Transformer ──► Mini-LLM (Decoder-Only GPT)
```

---

### Core Principles
- **Pure NumPy First**: Zero deep learning frameworks for from-scratch layers (`Dense`, `Conv2D`, `RNN`, `LSTM`, `GRU`, `Attention`, `Transformer`).
- **Exact Matrix Calculus**: All forward activations, loss functions, and backpropagation gradients are derived by hand before coding.
- **Rigorous Verification**: Automated unit tests assert that forward losses and backward gradients match theoretical math and finite differences ($< 10^{-7}$).
- **PyTorch Parity**: Direct gradient and weight parity verification against equivalent `torch.nn` modules.

---

### Curriculum & Modules

| # | Architecture | Focus & Mechanism | Links |
| :--- | :--- | :--- | :--- |
| **01** | [Neuron](01-neuron/README.md) | Single neuron, Sigmoid, BCE loss, XOR linear limit | [README](01-neuron/README.md) |
| **02** | [ANN Scratch](02-ann-from-scratch/README.md) | Multi-layer perceptron, ReLU, Softmax, non-linear XOR solution | [README](02-ann-from-scratch/README.md) |
| **03** | [MNIST ANN](03-mnist-ann/README.md) | 784-dim image classification, mini-batch SGD, He/Xavier initialization | [README](03-mnist-ann/README.md) |
| **04** | [Customer Churn](04-customer-churn/README.md) | Tabular preprocessing, entity embeddings, class imbalance handling | [README](04-customer-churn/README.md) |
| **05** | [CNN Scratch](05-cnn-from-scratch/README.md) | 2D conv, receptive fields, Max/Avg pooling, rotated backprop | [Math](05-cnn-from-scratch/steps/step-01-mathematics.md) · [Code](05-cnn-from-scratch/src/cnn.py) |
| **06** | [Traffic Sign CNN](06-traffic-sign-cnn/README.md) | Deep conv hierarchy, Dropout, BatchNorm, vision pipeline | [README](06-traffic-sign-cnn/README.md) |
| **07** | [RNN Scratch](07-rnn-from-scratch/README.md) | Recurrent hidden state, BPTT unrolling, collision resolution on `"hello"` | [Math](07-rnn-from-scratch/steps/step-01-mathematics.md) · [Code](07-rnn-from-scratch/src/rnn.py) |
| **08** | [Sentiment RNN](08-sentiment-rnn/README.md) | Many-to-One sequence classification, word embeddings, dynamic masking | [Math](08-sentiment-rnn/steps/step-01-mathematics.md) · [Code](08-sentiment-rnn/src/sentiment_rnn.py) |
| **09** | [LSTM Scratch](09-lstm/README.md) | Constant Error Carousel ($C_t$), $f/i/o/\tilde{C}$ gates, vanishing gradient immunity | [Math](09-lstm/steps/step-01-mathematics.md) · [Code](09-lstm/src/lstm.py) |
| **10** | [GRU Scratch](10-gru/README.md) | Reset & update gates ($r, z$), linear memory highway, 25% fewer params | [Math](10-gru/steps/step-01-mathematics.md) · [Code](10-gru/src/gru.py) |
| **11** | [Sequence Model](11-sequence-prediction/README.md) | Autoregressive causal shift $+1$, sparse BPTT, PPL, greedy/top-k/top-p sampling | [Math](11-sequence-prediction/steps/step-01-mathematics.md) · [Code](11-sequence-prediction/src/sequence_model.py) |
| **12** | [Attention](12-attention/README.md) | Bahdanau & Luong attention, associative Query-Key-Value ($Q, K, V$) | [README](12-attention/README.md) |
| **13** | [Mini-Transformer](13-mini-transformer/README.md) | Scaled Dot-Product, Multi-Head Attention, Positional Encoding, Pre-LN | [README](13-mini-transformer/README.md) |
| **14** | [Mini-LLM (GPT)](14-mini-llm/README.md) | Decoder-only autoregressive GPT, causal masking, KV cache | [README](14-mini-llm/README.md) |

---

### Quickstart & Verification

```bash
# Clone & install dependencies
git clone https://github.com/madhurangafcx/open-dl.git deep-learning-foundations
cd deep-learning-foundations
pip install numpy matplotlib torch

# Run self-contained unit verification suites
python3 05-cnn-from-scratch/src/cnn.py
python3 07-rnn-from-scratch/src/rnn.py
python3 08-sentiment-rnn/src/sentiment_rnn.py
python3 09-lstm/src/lstm.py
python3 10-gru/src/gru.py
python3 11-sequence-prediction/src/sequence_model.py
```

---

### Author
Created by **Pasan Madhuranga** ([@madhurangafcx](https://github.com/madhurangafcx)) as an open educational foundation for deep learning engineers and researchers.
