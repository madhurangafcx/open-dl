# 13 — Mini Transformer From Scratch

## Goal

Build, mathematically derive, and implement a complete **Encoder-Decoder Transformer** from first principles with pure NumPy, reproducing the architecture that revolutionized Artificial Intelligence in the landmark paper by Ashish Vaswani et al. (2017), *"Attention Is All You Need"*, and documented in Umar Jamil's architectural notes.

While Project `12-attention` isolated the Multi-Head Attention mechanism in a single layer, a full **Transformer** is a complete, modular deep learning system. It integrates **Token Embeddings**, **Sinusoidal Positional Encodings**, **Multi-Head Self-Attention**, **Causal Masked Multi-Head Attention**, **Encoder-Decoder Cross-Attention**, **Position-wise Feed-Forward Networks (FFN)**, **Layer Normalization**, and **Residual Connections (Add & Norm)** into a unified sequence-to-sequence neural network:

```text
========================================================================================================================
THE CANONICAL TRANSFORMER ARCHITECTURE (Vaswani et al., 2017)
========================================================================================================================

                  OUTPUT PROBABILITIES
                          ▲
                          │
                       Softmax
                          ▲
                          │
                     Linear Head (d_model ──► vocab_size)
                          ▲
                          │
               ┌─────────────────────┐
               │    Decoder Stack    │ ◄─── Repeat N times
               │ ┌─────────────────┐ │
               │ │   Add & Norm    │ │
               │ ├─────────────────┤ │
               │ │  Feed Forward   │ │
               │ ├─────────────────┤ │
               │ │   Add & Norm    │ │
               │ ├─────────────────┤ │
               │ │ Multi-Head Attn │─┼───────┐ Cross-Attention (Q from Decoder,
               │ ├─────────────────┤ │       │                  K, V from Encoder)
               │ │   Add & Norm    │ │       │
               │ ├─────────────────┤ │       │
               │ │ Masked MH Attn  │ │       │
               │ └─────────────────┘ │       │
               └──────────┬──────────┘       │
                          ▲                  │
                          │                  │
                    Positional               │
                     Encoding                │
                        (⊕)                  │
                          ▲                  │
                          │                  │
                   Output Embedding          │
                          ▲                  │
                          │                  │
                    Target Tokens            │
                   (Shifted Right)           │
                                             │
    ┌─────────────────────┐                  │
    │    Encoder Stack    │ ◄── Repeat N     │
    │ ┌─────────────────┐ │     times        │
    │ │   Add & Norm    │ │                  │
    │ ├─────────────────┤ │                  │
    │ │  Feed Forward   │ │                  │
    │ ├─────────────────┤ │                  │
    │ │   Add & Norm    │ │                  │
    │ ├─────────────────┤ │                  │
    │ │ Multi-Head Attn │─┼──────────────────┘ Keys (K) & Values (V)
    │ └─────────────────┘ │
    └──────────┬──────────┘
               ▲
               │
          Positional
           Encoding
              (⊕)
               ▲
               │
         Input Embedding
               ▲
               │
          Source Tokens
========================================================================================================================
```

---

## The Core Mathematical Equations of the Transformer

The Transformer processes inputs through a pipeline of seven fundamental mathematical transformations:

---

### 1. Scaled Input & Output Embeddings
Discrete token IDs are looked up in learnable embedding tables `E_{\mathrm{src}} \in \mathbb{R}^{V_{\mathrm{src}} \times d_{\mathrm{model}}}` and `E_{\mathrm{tgt}} \in \mathbb{R}^{V_{\mathrm{tgt}} \times d_{\mathrm{model}}}`.
As specified in Vaswani et al. (Section 3.4), embedding vectors are scaled by `\sqrt{d_{\mathrm{model}}}` to keep their magnitude appropriately balanced with the positional encodings:

```math
X_{\mathrm{embed}} = E[w] \cdot \sqrt{d_{\mathrm{model}}} \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
```

---

### 2. Sinusoidal Positional Encoding
Without positional information, self-attention does not inherently encode the order of tokens; permuting the input sequence correspondingly permutes the output representations. Therefore, fixed deterministic positional encodings are added directly to the embeddings:

```math
X_0 = X_{\mathrm{embed}} + PE \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
```

Where:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\mathrm{model}}}}\right)
```

```math
PE_{(pos, 2i + 1)} = \cos\left(\frac{pos}{10000^{2i / d_{\mathrm{model}}}}\right)
```

- `pos \in \{0, 1, \dots, T-1\}` is the token position index.
- `i \in \{0, 1, \dots, d_{\mathrm{model}}/2 - 1\}` is the channel index.

---

### 3. Layer Normalization (Ba, Kiros, & Hinton, 2016)
Layer Normalization normalizes activations across the channel dimension for each token independently, stabilizing deep gradient flow:

```math
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
```

Where:

```math
\mu = \frac{1}{d_{\mathrm{model}}} \sum_{j=1}^{d_{\mathrm{model}}} x_j, \quad \sigma^2 = \frac{1}{d_{\mathrm{model}}} \sum_{j=1}^{d_{\mathrm{model}}} (x_j - \mu)^2
```

- `\gamma \in \mathbb{R}^{d_{\mathrm{model}}}`: Learnable multiplicative gain parameter (initialized to `1.0`).
- `\beta \in \mathbb{R}^{d_{\mathrm{model}}}`: Learnable additive bias parameter (initialized to `0.0`).
- `\epsilon = 10^{-6}`: Numerical stability constant preventing division by zero.

---

### 4. Multi-Head Attention (MHA)
Computes scaled dot-product attention across `h` parallel heads:

```math
\mathrm{MultiHead}(Q, K, V) = \operatorname{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O
```

```math
\mathrm{head}_i = \operatorname{softmax}\left(\frac{Q W_i^Q (K W_i^K)^T}{\sqrt{d_k}} + M\right) (V W_i^V)
```

Where:
- In **Encoder Self-Attention**: `Q = K = V = X_{\mathrm{enc}}`, Mask `M = \mathbf{0}` (or padding mask).
- In **Decoder Masked Self-Attention**: `Q = K = V = X_{\mathrm{dec}}`, Mask `M = M_{\mathrm{causal}}` (`-\infty` in upper triangle).
- In **Cross-Attention**: `Q = X_{\mathrm{dec}}`, `K = V = X_{\mathrm{enc\_out}}` (Queries come from the decoder; Keys and Values come from the encoder stack!).

---

### 5. Position-wise Feed-Forward Network (FFN)
Each attention output is passed through an identical two-layer multi-layer perceptron with a ReLU activation in between:

```math
\mathrm{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 = \operatorname{ReLU}(x W_1 + b_1) W_2 + b_2
```

Where:
- `W_1 \in \mathbb{R}^{d_{\mathrm{model}} \times d_{ff}}`
- `b_1 \in \mathbb{R}^{1 \times d_{ff}}`
- `W_2 \in \mathbb{R}^{d_{ff} \times d_{\mathrm{model}}}`
- `b_2 \in \mathbb{R}^{1 \times d_{\mathrm{model}}}`
- Typically `d_{ff} = 4 \times d_{\mathrm{model}}` (e.g., `512 \to 2048 \to 512`).

---

### 6. Residual Connection (Add & Norm)
Every sub-layer is wrapped in a residual connection followed by layer normalization (Post-LN formulation of Vaswani et al.):

```math
x_{\mathrm{sub\_out}} = \mathrm{LayerNorm}\left(x + \mathrm{SubLayer}(x)\right)
```

The residual addition creates a direct identity pathway for gradients, which substantially improves gradient propagation through deep networks:

```math
\frac{\partial (x + \mathrm{SubLayer}(x))}{\partial x} = \mathbf{I} + \frac{\partial \mathrm{SubLayer}(x)}{\partial x}
```

---

### 7. Linear Output Projection & Softmax
The final decoder hidden state `H_{\mathrm{dec}} \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}` is projected into vocabulary space:

```math
O = H_{\mathrm{dec}} W_{\mathrm{vocab}} + b_{\mathrm{vocab}} \in \mathbb{R}^{B \times T \times V_{\mathrm{tgt}}}
```

```math
\hat{Y} = \operatorname{softmax}(O) \in (0, 1)^{B \times T \times V_{\mathrm{tgt}}}
```

---

## The Three Architectural Pillars of the Transformer

Beyond Multi-Head Attention, three structural innovations make deep Transformer networks stable, expressive, and trainable:

```text
========================================================================================================================
THE THREE ARCHITECTURAL PILLARS OF THE TRANSFORMER
========================================================================================================================

PILLAR 1: LAYER NORMALIZATION (Sample-Independent Stability)
   * LayerNorm is particularly well suited to Transformer architectures because it normalizes
     each token independently across its feature dimensions and does not depend on batch statistics.
   * Each word vector is zero-centered (mean=0) and scaled (variance=1) individually.
   * Completely independent of batch size B!

PILLAR 2: RESIDUAL CONNECTIONS (Unbroken Gradient Highways)
   * Without residual connections, deep stacks of attention layers (N=6, 12, 24) struggle to train.
   * The identity path x + F(x) creates a direct pathway for gradients to flow backward
     all the way to the input embeddings, substantially improving gradient propagation.

PILLAR 3: POSITION-WISE FEED-FORWARD NETWORKS (Semantic Non-linear Memory)
   * Self-attention primarily performs content-dependent information routing and feature mixing
     through learned projections and attention weights.
   * The FFN provides the main position-wise nonlinear transformation within the Transformer block,
     complementing the content-dependent mixing performed by self-attention.
   * The FFN performs a shared position-wise nonlinear transformation on each token representation:
     it expands features into a higher dimension (4 * d_model), applies a non-linear activation
     (ReLU or GELU), and projects back into representation space. It can also be interpreted as
     storing and retrieving feature associations in the learned weights.
========================================================================================================================
```

---

### Deep Dive: Layer Normalization vs. Batch Normalization

```text
       BATCH NORMALIZATION (Across Batch)             LAYER NORMALIZATION (Across Features)
       
              Channels (d_model)                             Channels (d_model)
              ┌───┬───┬───┬───┐                              ┌───┬───┬───┬───┐
     Batch 1  │   │   │   │   │                     Batch 1  │ █ │ █ │ █ │ █ │ ◄── Mean & Var
              ├───┼───┼───┼───┤                              ├───┼───┼───┼───┤     computed across
     Batch 2  │   │   │   │   │                     Batch 2  │ █ │ █ │ █ │ █ │     all channels
              ├───┼───┼───┼───┤                              ├───┼───┼───┼───┤     for THIS token!
     Batch 3  │   │   │   │   │                     Batch 3  │ █ │ █ │ █ │ █ │
              └───┴───┴───┴───┘                              └───┴───┴───┴───┘
                ▲
                └── Mean & Var computed ACROSS BATCH!
```

- **Batch Normalization**: Computes statistics vertically down the batch column. Batch Normalization relies on statistics computed from the batch, which can become noisy or poorly estimated for very small batch sizes, and requires careful handling for variable sequence lengths.
- **Layer Normalization**: Computes statistics horizontally across the feature dimensions `d_{\mathrm{model}}` for a single token. Behavior is identical during training and inference, with zero dependence on batch size.

---

## Encoder vs. Decoder: Detailed Operational Mechanics

The Transformer operates as an asymmetric sequence-to-sequence machine:

```text
========================================================================================================================
ENCODER STACK (Context Extractor)                DECODER STACK (Autoregressive Generator)
========================================================================================================================
Role: Read and understand source text            Role: Generate target text one word at a time
Attention: Bidirectional Self-Attention          Attention: 1. Causal Masked Self-Attention
                                                            2. Cross-Attention (Attends to Encoder)
Visibility: Every word sees all other words       Visibility: At decoder position t, the model can attend
                                                              only to decoder-input tokens at positions <= t,
                                                              while predicting the next target token
Input: Source sequence ("I love you very much")  Input: Shifted Target sequence ("<SOS> Ti amo molto")
Output: Rich context vectors (Keys & Values)     Output: Next-token logits over vocabulary
========================================================================================================================
```

---

### How Cross-Attention Bridges Encoder and Decoder

Cross-Attention is the mathematical bridge that allows the target language to look up words in the source language:

```math
\mathrm{CrossAttention} = \operatorname{softmax}\left(\frac{Q_{\mathrm{dec}} K_{\mathrm{enc}}^T}{\sqrt{d_k}}\right) V_{\mathrm{enc}}
```

```text
Decoder Query (Q):    "Ti" (Italian word being generated)
                               │
                               ├── Dot Product Similarity Matrix ──► Softmax Alignment Weights
                               │
Encoder Keys (K):     [ "I",   "love",   "you",   "very",   "much" ]
Encoder Values (V):   [ v("I"), v("love"), v("you"), v("very"), v("much") ]
                               │
Example weighted context representation:    0.85 · v("you") + 0.10 · v("I") + 0.05 · v("love")
                                            (The decoder dynamically retrieves a context representation from the encoded source sequence.)
```

---

## Training Mode vs. Inference Mode: Parallel Training vs. Autoregressive Generation

As highlighted in Umar Jamil's notes (Slide 28), the training of a Transformer differs profoundly from an RNN:

```text
========================================================================================================================
TRAINING MODE: PARALLEL TARGET PROCESSING (Teacher Forcing via Causal Mask)
========================================================================================================================
Source:           "<SOS> I love you very much <EOS>"
Target (Input):   "<SOS> Ti amo molto"
Target (Label):   "Ti amo molto <EOS>"

Training iteration: Parallel target processing:
- The entire shifted target sequence is fed into the decoder simultaneously in one forward pass.
- The causal mask prevents earlier positions from attending to subsequent tokens.
- The model computes predictions for all target positions in parallel.
- Total cross-entropy loss is evaluated across all positions in a single backward pass.
========================================================================================================================
INFERENCE MODE: AUTOREGRESSIVE GENERATION (Step-by-Step Free Running)
========================================================================================================================
Step 1: Input: "<SOS>"                          ──► Decoder ──► Predicts: "Ti"
Step 2: Input: "<SOS> Ti"                       ──► Decoder ──► Predicts: "amo"
Step 3: Input: "<SOS> Ti amo"                   ──► Decoder ──► Predicts: "molto"
Step 4: Input: "<SOS> Ti amo molto"             ──► Decoder ──► Predicts: "<EOS>" (STOP!)
========================================================================================================================
```

---

## Parameters and Tensor Shapes Reference

Let:
- `B`: Batch size (e.g., `1`).
- `T_{\mathrm{src}}`: Source sequence length (e.g., `5`).
- `T_{\mathrm{tgt}}`: Target sequence length (e.g., `4`).
- `d_{\mathrm{model}}`: Model representation dimension (e.g., `16`).
- `h`: Number of attention heads (e.g., `2`).
- `d_k = d_v = d_{\mathrm{model}} / h`: Head dimension (e.g., `8`). For the original Transformer configuration used here, `d_k = d_v = d_{\mathrm{model}} / h`.
- `d_{ff}`: Feed-forward hidden dimension (e.g., `64`).
- `V_{\mathrm{src}}`: Source vocabulary size (e.g., `20`).
- `V_{\mathrm{tgt}}`: Target vocabulary size (e.g., `20`).

> [!NOTE]
> The parameter count below uses separate source embeddings, target embeddings, and output projection weights for implementation clarity. The original Transformer paper used shared embedding/output weights.

| Component | Tensor / Parameter | Shape | Parameter Formula | Count (`d=16, d_{ff}=64, V=20`) |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | `E_{\mathrm{src}}` | `(V_{\mathrm{src}}, d_{\mathrm{model}})` | `V_{\mathrm{src}} \times d_{\mathrm{model}}` | `20 \times 16 = 320` |
| | `E_{\mathrm{tgt}}` | `(V_{\mathrm{tgt}}, d_{\mathrm{model}})` | `V_{\mathrm{tgt}} \times d_{\mathrm{model}}` | `20 \times 16 = 320` |
| **Encoder Self-Attn** | `W^Q, W^K, W^V, W^O` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `4 \times d_{\mathrm{model}}^2` | `4 \times 256 = 1,024` |
| **Encoder LayerNorm 1**| `\gamma_1, \beta_1` | `(1, d_{\mathrm{model}})` | `2 \times d_{\mathrm{model}}` | `2 \times 16 = 32` |
| **Encoder FFN** | `W_1, b_1, W_2, b_2` | `(d, d_{ff}), (d_{ff}, d)` | `2 \cdot d \cdot d_{ff} + d_{ff} + d` | `2(1024) + 64 + 16 = 2,128` |
| **Encoder LayerNorm 2**| `\gamma_2, \beta_2` | `(1, d_{\mathrm{model}})` | `2 \times d_{\mathrm{model}}` | `2 \times 16 = 32` |
| **Decoder Masked Attn**| `W^Q, W^K, W^V, W^O` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `4 \times d_{\mathrm{model}}^2` | `4 \times 256 = 1,024` |
| **Decoder LayerNorm 1**| `\gamma_3, \beta_3` | `(1, d_{\mathrm{model}})` | `2 \times d_{\mathrm{model}}` | `2 \times 16 = 32` |
| **Decoder Cross-Attn** | `W^Q, W^K, W^V, W^O` | `(d_{\mathrm{model}}, d_{\mathrm{model}})` | `4 \times d_{\mathrm{model}}^2` | `4 \times 256 = 1,024` |
| **Decoder LayerNorm 2**| `\gamma_4, \beta_4` | `(1, d_{\mathrm{model}})` | `2 \times d_{\mathrm{model}}` | `2 \times 16 = 32` |
| **Decoder FFN** | `W_3, b_3, W_4, b_4` | `(d, d_{ff}), (d_{ff}, d)` | `2 \cdot d \cdot d_{ff} + d_{ff} + d` | `2(1024) + 64 + 16 = 2,128` |
| **Decoder LayerNorm 3**| `\gamma_5, \beta_5` | `(1, d_{\mathrm{model}})` | `2 \times d_{\mathrm{model}}` | `2 \times 16 = 32` |
| **Output Head** | `W_{\mathrm{vocab}}, b_{\mathrm{vocab}}` | `(d_{\mathrm{model}}, V_{\mathrm{tgt}})` | `d_{\mathrm{model}} \times V_{\mathrm{tgt}} + V_{\mathrm{tgt}}` | `16 \times 20 + 20 = 340` |
| **TOTAL** | — | — | — | **8,468 learnable parameters for this Mini Transformer configuration** |

---

## Project Structure

```text
13-mini-transformer/
├── README.md                      # Architecture foundations, diagrams, and 101-topic syllabus
├── src/
│   └── transformer.py             # Pure NumPy Encoder, Decoder, LayerNorm, FFN, and Trainer
└── steps/
    └── step-01-mathematics.md      # Matrix calculus derivations, LayerNorm backprop, full forward trace
```

---

## Learning Workflow

Following the repository's established 10-step sequence:

| Step | Status | Evidence / Milestone |
| :--- | :--- | :--- |
| **1. Mathematics** | Pending | `steps/step-01-mathematics.md`: Complete mathematical derivation of LayerNorm, FFN, Cross-Attention, and backprop calculus |
| **2. NumPy Implementation** | Pending | `src/transformer.py`: Vectorized `EncoderLayer`, `DecoderLayer`, `Transformer` classes in pure NumPy |
| **3. Understand Forward Pass** | Pending | End-to-end tensor verification from input tokens to vocabulary logits |
| **4. Understand Loss & Masking** | Pending | Masked sequence cross-entropy loss, causal triangle masking, padding masking |
| **5. Derive Gradients** | Pending | Analytical backpropagation formulas for LayerNorm (`d\gamma, d\beta, dx`), FFN, and Attention |
| **6. Implement Backpropagation** | Pending | Full reverse-mode automatic differentiation verified against finite differences down to `10^{-7}` |
| **7. Train Model** | Pending | Overfitting on small English-to-Italian machine translation dataset until training loss `< 0.05` |
| **8. Debug & Analyze** | Pending | Analyzing cross-attention heatmaps, residual gradient highways, and LayerNorm stability |
| **9. PyTorch Implementation** | Pending | Equivalent model built with `torch.nn.Transformer` / `torch.nn.TransformerEncoder` |
| **10. Compare Results** | Pending | Exact parity verification of parameters, activations, cross-entropy loss, and translation outputs |

---

## Complete 101-Topic Foundational Curriculum

This project systematically covers the following 101 core deep learning topics across 16 structured modules:

```text
TRANSFORMER PHILOSOPHY & FOUNDATIONS
│
├── 01. The sequence-to-sequence problem (Machine Translation)
├── 02. Why Encoder-Decoder? (Source comprehension vs target generation)
├── 03. The death of recurrence: Discarding the sequential loop
├── 04. Vaswani et al. (2017) architectural overview
├── 05. Umar Jamil's visual decomposition of the Transformer
├── 06. Macro tensor shapes: (Batch, Sequence, d_model)
│
▼
INPUT EMBEDDINGS & SCALING
│
├── 07. Source vocabulary vs Target vocabulary (Separate or shared)
├── 08. Embedding lookup: Indexing into E in R^(V x d_model)
├── 09. Why multiply embeddings by sqrt(d_model)?
├── 10. Preserving variance parity with positional encodings
├── 11. Weight tying concept: Sharing embedding and output linear head
│
▼
SINUSOIDAL POSITIONAL ENCODINGS
│
├── 12. Permutation equivariance of pure self-attention and why positional encodings are required
├── 13. The geometric necessity of token order
├── 14. Sinusoidal wave equations for even and odd dimensions
├── 15. The 10,000^(2i/d_model) frequency wavelength denominator
├── 16. Linear relative translation property: PE_{pos+k} = PE_pos * M_k
├── 17. Pre-computing and caching positional encodings once
├── 18. Adding embeddings and positional encodings: X_0 = X_embed + PE
│
▼
LAYER NORMALIZATION DEEP DIVE
│
├── 19. Why Batch Normalization fails for sequential NLP
├── 20. Layer Normalization (Ba et al., 2016) formulation
├── 21. Calculating mean mu across d_model
├── 22. Calculating variance sigma^2 across d_model
├── 23. Standardizing activations: x_hat = (x - mu) / sqrt(sigma^2 + eps)
├── 24. Learnable affine parameters: gamma (scale) and beta (shift)
├── 25. Post-LN (Vaswani et al.) vs Pre-LN (Modern GPT/LLaMA)
├── 26. Backpropagation through Layer Normalization: dgamma, dbeta, dx
│
▼
RESIDUAL CONNECTIONS (ADD & NORM)
│
├── 27. Deep network degradation and vanishing gradients
├── 28. He et al. (2016) ResNet identity mapping insight
├── 29. Mathematical formulation: x + SubLayer(x)
├── 30. Direct identity gradient propagation: d(x + F(x))/dx = I + dF/dx
├── 31. Preserving feature reuse across N layers
│
▼
MULTI-HEAD ATTENTION REVIEW
│
├── 32. Unified Q, K, V linear projections
├── 33. Splitting into h heads: d_k = d_model / h
├── 34. Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) * V
├── 35. Multi-head concatenation and output projection W^O
├── 36. Caching intermediate attention weights for visualization
│
▼
POSITION-WISE FEED-FORWARD NETWORKS (FFN)
│
├── 37. Why Multi-Head Attention alone is not enough
├── 38. Attention as routing vs FFN as computation/memory
├── 39. First linear projection: W_1 in R^(d_model x d_ff)
├── 40. Expansion factor: d_ff = 4 * d_model
├── 41. Non-linear activation: ReLU vs GELU vs SwiGLU
├── 42. Second linear projection: W_2 in R^(d_ff x d_model)
├── 43. FFN parameter budget: 2 * d_model * d_ff + d_ff + d_model
├── 44. FFN backpropagation: Deriving dW_1, db_1, dW_2, db_2, dx
│
▼
ENCODER ARCHITECTURE & STACK
│
├── 45. Anatomy of an Encoder Layer: Self-Attn -> Add&Norm -> FFN -> Add&Norm
├── 46. Bidirectional attention: Zero future masking
├── 47. Source padding mask: Zeroing attention from <PAD> tokens
├── 48. Stacking N identical encoder layers
├── 49. Forward pass through the full Encoder stack
├── 50. Final encoder memory output: Continuous contextual representations
│
▼
DECODER ARCHITECTURE & CAUSAL MASKING
│
├── 51. Anatomy of a Decoder Layer: 3 sub-layers instead of 2
├── 52. Sub-layer 1: Masked Multi-Head Self-Attention
├── 53. The causal lower-triangular mask (preventing future lookahead)
├── 54. Setting future positions to -inf before Softmax
├── 55. Add & Norm after masked self-attention
│
▼
CROSS-ATTENTION (ENCODER-DECODER ATTENTION)
│
├── 56. Sub-layer 2: The bridge between target and source
├── 57. Queries (Q) originate from previous decoder sub-layer
├── 58. Keys (K) and Values (V) originate from the final Encoder output
├── 59. Cross-attention dot-product: Q_dec @ K_enc^T / sqrt(d_k)
├── 60. Shape of Cross-Attention matrix: (B, h, T_tgt, T_src)
├── 61. Aligning target words with source representations
├── 62. Add & Norm after cross-attention
│
▼
DECODER FFN & VOCABULARY HEAD
│
├── 63. Sub-layer 3: Decoder position-wise Feed-Forward Network
├── 64. Final decoder Add & Norm layer
├── 65. Decoder stack output tensor: (B, T_tgt, d_model)
├── 66. Linear output vocabulary projection: W_vocab in R^(d_model x V_tgt)
├── 67. Generating unnormalized logits: (B, T_tgt, V_tgt)
├── 68. Softmax conversion to next-token probabilities
│
▼
LOSS FUNCTION & TRAINING DYNAMICS
│
├── 69. Shifted right target sequence: Feeding <SOS> at position 0
├── 70. Target sequence alignment: y_t = w_{t+1}^*
├── 71. Cross-entropy loss across all sequence positions
├── 72. Masking the loss at <PAD> positions
├── 73. Teacher Forcing in parallel: All positions trained in 1 time step
├── 74. Label smoothing regularization (Vaswani et al. Section 5.4)
├── 75. Warmup learning rate schedule: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
│
▼
AUTOREGRESSIVE INFERENCE & GENERATION
│
├── 76. Why inference cannot be parallelized: The sequential generation loop
├── 77. Encoding the source prompt once: H_enc = Encoder(prompt)
├── 78. Initializing decoder input with <SOS>
├── 79. Step 1: Feed <SOS>, retrieve logits, pick token 1
├── 80. Step 2: Append token 1 to decoder input, re-run decoder
├── 81. Stopping criterion: Encountering <EOS> or reaching max_len
├── 82. Greedy decoding (Argmax)
├── 83. Beam Search decoding (Maintaining top B hypotheses)
├── 84. Key-Value (KV) Caching concept preview (Avoiding redundant recomputation: reducing cumulative autoregressive self-attention from roughly O(T^3) toward O(T^2))
│
▼
FULL-SYSTEM BACKPROPAGATION CALCULUS
│
├── 85. Reverse computation graph through the entire Transformer
├── 86. Output projection gradient: dW_vocab, db_vocab
├── 87. Backward pass through Decoder Layer N: FFN -> Cross-Attn -> Masked Self-Attn
├── 88. Cross-attention gradient split: Gradients flow to both Decoder and Encoder!
├── 89. Backward pass through Encoder Layer N: FFN -> Self-Attn
├── 90. Accumulating gradients into embedding tables E_src and E_tgt
├── 91. Finite-difference numerical gradient verification (< 1e-7 relative error)
│
▼
TRAINING EXPERIMENT & VALIDATION
│
├── 92. Mini Machine Translation corpus (English -> Italian)
├── 93. Training loop with Adam optimizer in pure NumPy
├── 94. Overfitting validation: Driving loss from 3.0 down to < 0.05
├── 95. Tracking translation accuracy on training sentences
├── 96. Visualizing Cross-Attention alignment heatmaps (English vs Italian words)
│
▼
PYTORCH BENCHMARK & SYSTEM COMPARISON
│
├── 97. PyTorch torch.nn.Transformer implementation
├── 98. Weight-by-weight mapping and parameter copying
├── 99. Exact numerical agreement validation down to 7 decimal places
├── 100. Profiling memory and latency: NumPy CPU vs PyTorch GPU
└── 101. Final synthesis: The Transformer as the foundation of modern AI
```

---

## Conclusion & Curriculum Progression

Mastering the **Mini Transformer** synthesizes every fundamental mathematical concept developed across this repository—matrix calculus, activation derivatives, linear projections, continuous embeddings, layer normalization, residual shortcuts, and multi-head attention—into the world's most powerful deep learning architecture.

The next immediate step is working through the exhaustive mathematical derivations, tensor proofs, and finite-difference validation in [steps/step-01-mathematics.md](file:///Users/pasan/Documents/Personal/deep-learning-foundations/13-mini-transformer/steps/step-01-mathematics.md), followed by the complete NumPy implementation in [src/transformer.py](file:///Users/pasan/Documents/Personal/deep-learning-foundations/13-mini-transformer/src/transformer.py).
