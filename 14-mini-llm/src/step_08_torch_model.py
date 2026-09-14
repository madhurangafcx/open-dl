"""The existing NumPy architecture, expressed with differentiable tensors.

Every parameter is copied from a NumPy MiniLLM. This lets us verify the
forward pass before training. Matrix orientations stay the same as NumPy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def parameter(array) -> nn.Parameter:
    # torch.tensor copies the array: training cannot mutate the NumPy model.
    return nn.Parameter(torch.tensor(array, dtype=torch.float32))


class TorchLayerNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gamma = parameter(source.gamma)
        self.beta = parameter(source.beta)
        self.eps = source.eps

    def forward(self, x):
        centered = x - x.mean(dim=-1, keepdim=True)
        # Population variance, matching NumPy's default ddof=0.
        variance = centered.square().mean(dim=-1, keepdim=True)
        return self.gamma * centered / torch.sqrt(variance + self.eps) + self.beta


class TorchAttentionHead(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.Wq = parameter(source.Wq)
        self.Wk = parameter(source.Wk)
        self.Wv = parameter(source.Wv)
        self.head_dim = source.head_dim

    def forward(self, x):
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        length = x.shape[1]
        future = torch.ones(
            length, length, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        weights = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
        return weights @ v


class TorchMultiHeadAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        # ModuleList registers every head's parameters with the parent model.
        self.heads = nn.ModuleList([
            TorchAttentionHead(head) for head in source.heads
        ])
        self.Wo = parameter(source.Wo)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return combined @ self.Wo


class TorchFeedForward(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.W1 = parameter(source.W1)
        self.b1 = parameter(source.b1)
        self.W2 = parameter(source.W2)
        self.b2 = parameter(source.b2)

    def forward(self, x):
        # The default exact GELU would differ from our NumPy approximation.
        hidden = F.gelu(x @ self.W1 + self.b1, approximate="tanh")
        return hidden @ self.W2 + self.b2


class TorchTransformerBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = TorchLayerNorm(source.norm1)
        self.attention = TorchMultiHeadAttention(source.attention)
        self.norm2 = TorchLayerNorm(source.norm2)
        self.ffn = TorchFeedForward(source.ffn)

    def forward(self, x):
        h = x + self.attention(self.norm1(x))
        return h + self.ffn(self.norm2(h))


class TorchMiniLLM(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.vocab_size = source.vocab_size
        self.max_seq_len = source.max_seq_len
        self.token_embedding = parameter(source.token_embedding)
        # A buffer moves with model.to(device) and is saved in state_dict,
        # but the optimizer does not update these fixed sinusoidal positions.
        self.register_buffer(
            "positions", torch.tensor(source.positions, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList([
            TorchTransformerBlock(block) for block in source.blocks
        ])
        self.final_norm = TorchLayerNorm(source.final_norm)
        self.W_vocab = parameter(source.W_vocab)
        self.b_vocab = parameter(source.b_vocab)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("Expected torch.long token IDs with shape (B, T)")
        length = token_ids.shape[1]
        if not 1 <= length <= self.max_seq_len:
            raise ValueError("Sequence length is outside the context window")
        # F.embedding also rejects negative and out-of-range token IDs.
        x = F.embedding(token_ids, self.token_embedding)
        x = x + self.positions[None, :length, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Return raw logits. Cross-entropy handles log-softmax internally.
        return x @ self.W_vocab + self.b_vocab
