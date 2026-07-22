"""
baseline/attention.py

Standard multi-head self-attention - O(n^2) reference implementation.

Longformer, Linformer and Reformer are all drop-in replacements for MultiHeadSelfAttention.
They share the same forward(x, mask) signature so the encoder block doesnt need to change between experiments.

Reference:
    [X] Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes softmax(QK^T / sqrt(d_k)) V - Equation 1 in [X] Vaswani et al. (2017).
    Kept separate so it can be unit tested on its own.
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # q, k: (batch, heads, num_tokens, head_dim)
        # v: (batch, heads, num_tokens, head_dim)
        d_k = q.size(-1)

        # (batch, heads, num_tokens, head_dim) @ (batch, heads, head_dim, num_tokens) -> (batch, heads, num_tokens, num_tokens)
        # scale by sqrt(d_k) - Vaswani et al. [X] footnote p.4 explains that large d_k causes dot products to grow large, pushing softmax into regions of very small gradients
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # set padding positions to -inf so they become 0 after softmax
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        # if a row is entirely masked softmax returns NaN - replace with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # (batch, heads, num_tokens, num_tokens) @ (batch, heads, num_tokens, head_dim) -> (batch, heads, num_tokens, head_dim)
        context_vecs = attn_weights @ v

        return context_vecs, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention from [X] Vaswani et al. (2017), Section 3.2.

    Projects input into h query/key/value spaces of dimension head_dim = d_model // h, runs attention in parallel across all heads then concatenates and projects back.

    This is the baseline module - Longformer, Linformer and Reformer all replace this class while keeping the same forward(x, mask) signature.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of attention heads. d_model must divide evenly.
        dropout (float): applied to attention weights and output projection.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # W_q, W_k, W_v, W_o kept explicit to match notation in [X] Vaswani et al.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(self.W_o.bias)

    def _split_heads(self, x):
        batch, num_tokens, _ = x.size()
        # (batch, num_tokens, d_model) -> (batch, num_tokens, num_heads, head_dim)
        x = x.view(batch, num_tokens, self.num_heads, self.head_dim)
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch, _, num_tokens, _ = x.size()
        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, num_heads, head_dim)
        x = x.transpose(1, 2).contiguous()
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_tokens, d_model)
        return x.view(batch, num_tokens, self.d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): (batch, num_tokens, d_model)
            mask (Tensor): (batch, num_tokens), True = padding, ignore this position.

        Returns:
            out (Tensor): (batch, num_tokens, d_model)
            attn_weights (Tensor): (batch, num_heads, num_tokens, num_tokens)
        """
        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # expand mask from (batch, num_tokens) to (batch, 1, 1, num_tokens) so it broadcasts over heads and query positions
        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None

        # context_vecs: (batch, num_heads, num_tokens, head_dim)
        context_vecs, attn_weights = self.attention(q, k, v, mask=attn_mask)

        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, d_model)
        out = self.dropout(self.W_o(self._merge_heads(context_vecs)))

        return out, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise FFN - two linear layers with GELU activation.
    Structure and inner dimension (4 * d_model) follow [X] Vaswani et al. Section 3.3.
    Original paper uses ReLU but GELU is used here as it has become standard in more recent transformer implementations and generally performs better.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (batch, num_tokens, d_model) -> (batch, num_tokens, d_ff) -> (batch, num_tokens, d_model)
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    Single encoder block: pre-norm self-attention + pre-norm FFN, both with residuals.

    Pre-norm is used instead of post-norm as it trains more stably at smaller batch sizes - also used in [X] Beltagy et al. (2020) Longformer, Appendix B.

    attn_cls is how the four variants share this block - passing a different attention class here swaps the mechanism without changing anything else.
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, attn_cls=None):
        super().__init__()
        attn_cls = attn_cls or MultiHeadSelfAttention

        self.attn = attn_cls(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # pre-norm self-attention with residual
        # (batch, num_tokens, d_model) -> (batch, num_tokens, d_model)
        residual = x
        x, attn_weights = self.attn(self.norm1(x), mask=mask)
        x = residual + self.dropout(x)

        # pre-norm feed-forward with residual
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from [X] Vaswani et al. (2017), Section 3.5.

    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Stored as a buffer so it moves with the model but is not updated during training.
    """

    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model) # (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)) # (d_model // 2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # unsqueeze adds batch dimension so it broadcasts over any batch size
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_seq_len, d_model)

    def forward(self, x):
        # (batch, num_tokens, d_model) + (1, num_tokens, d_model) -> (batch, num_tokens, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaselineTransformerEncoder(nn.Module):
    """
    Full encoder stack used across all LRA experiments.

    Architecture: token embedding -> positional encoding -> N x TransformerEncoderBlock -> LayerNorm -> mean pool -> classifier.

    attn_cls is passed to every block so all four variants run through the same stack under identical conditions, satisfying the unified evaluation framework required by [X] Tay et al. (2020) LRA benchmark.

    Args:
        vocab_size (int): vocabulary size.
        d_model (int): model dimension.
        num_heads (int): attention heads per block.
        num_layers (int): number of stacked encoder blocks.
        num_classes (int): number of output classes.
        max_seq_len (int): max sequence length for positional encoding.
        d_ff (int): FFN inner dimension.
        dropout (float): dropout rate.
        attn_cls (type): attention class injected into every block.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_seq_len=4096, d_ff=None, dropout=0.1, attn_cls=None):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout, attn_cls=attn_cls)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        self.embedding.weight.data[0].zero_() # zero out padding token embedding

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): (batch, num_tokens) token indices.
            attention_mask (Tensor): (batch, num_tokens), 1 = real token, 0 = padding.
            Follows HuggingFace convention used by LRA data loaders.

        Returns:
            dict:
                'logits' (Tensor): (batch, num_classes)
                'attn_weights' (list): per-layer attention weights for visualisation.
        """
        # convert HF mask (1=real, 0=pad) to boolean ignore mask (True=ignore)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None

        # (batch, num_tokens) -> (batch, num_tokens, d_model)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        all_attn_weights = []
        for block in self.blocks:
            # (batch, num_tokens, d_model) -> (batch, num_tokens, d_model)
            x, attn_w = block(x, mask=pad_mask)
            all_attn_weights.append(attn_w)

        x = self.norm(x)

        # mean pool over real tokens only - padding positions excluded
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float() # (batch, num_tokens, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1) # (batch, d_model)
        else:
            x = x.mean(dim=1) # (batch, d_model)

        # (batch, d_model) -> (batch, num_classes)
        logits = self.classifier(x)

        return {"logits": logits, "attn_weights": all_attn_weights}


""" NOTES

    nn.LayerNorm - keeps the numbers in the model from getting too big or small as they pass through each layer. Makes training more stable.

    nn.Embedding - converts word/token IDs into vectors the model can actually learn from.

    nn.ModuleList - a list of layers that PyTorch sees and trains. Normal Python list would make PyTorch ignore the layers.

    register_buffer - saves the positional encoding inside the model so it moves to GPU with everything else, not updated during training.

    masked_fill - stamps -inf onto padding positions before softmax so they end up as 0 and get ignored.

    nan_to_num - if every position in a row is padding you get 0/0 which is NaN. Replace with 0 so training doesnt crash.

    .contiguous() - PyTorch needs data laid out in memory after swapping dimensions before it can reshape.

    .clamp(min=1) - stops division by 0 in the mean pooling step if entire sequence is padding.

    unsqueeze - adds an extra dimension to a tensor so it lines up correctly with another tensor of a different shape.

    Masked mean pooling - when collapsing the sequence to one vector for classification, only average real tokens and ignore padding.

    Returning a dict from forward - easy to grab both predictions and attention patterns from a single forward pass without changing anything else.

"""