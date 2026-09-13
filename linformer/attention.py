"""
linformer/attention.py

Linformer low-rank projected attention.

Replaces the O(n^2) full attention matrix with a low-rank approximation:
    - K and V are projected from seq_len down to a much smaller fixed size (proj_dim) using learned linear projections E and F
    - Q stays full length so attention scores become (num_tokens, proj_dim) instead of (num_tokens, num_tokens)

This reduces complexity from O(n^2) to O(n*k) where k = proj_dim is fixed and independent of sequence length following Wang et al. (2020), Theorem 2.

Padding tokens are zeroed out in K and V before projection since after E and F are applied each compressed position is a mixture of all original positions and can no longer be masked individually.

E and F require a fixed seq_len at construction time since they are linear layers that map seq_len -> proj_dim.
This works for LRA tasks where sequences are padded to a constant length per task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinformerAttention(nn.Module):
    """
    Linformer attention using low-rank projections of K and V.

    Q stays at full sequence length but K and V are compressed from seq_len down to proj_dim using learned linear projections E and F respectively. 
    Attention is then computed between the full-length Q and the compressed K, V, giving attention scores of shape (num_tokens, proj_dim) instead of (num_tokens, num_tokens).

    E and F are separate learned projections, following the basic linear projection method in Wang et al. (2020) Section 4, Equation 7. 
    Parameter sharing across heads/layers is mentioned in the paper as a further optimisation but is not implemented here to keep the architecture straightforward.

    This class exposes the same forward(x, mask) signature as MultiHeadSelfAttention in baseline/attention.py so it can be passed as attn_cls into TransformerEncoderBlock without any other changes.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of attention heads.
        dropout (float): applied to attention weights.
        seq_len (int): fixed sequence length E and F are built for. Must match the padded sequence length used by the task.
        proj_dim (int): compressed dimension k that K and V are projected down to. Defaults to 128, following Wang et al. (2020) who show k=128 performs close to full attention.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, seq_len=512, proj_dim=128):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.seq_len = seq_len
        self.proj_dim = proj_dim

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        # E and F project along the sequence dimension - they reduce the number of tokens from seq_len to proj_dim, not the size of each token's vector
        self.E = nn.Linear(seq_len, proj_dim, bias=False)
        self.F = nn.Linear(seq_len, proj_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.W_q, self.W_k, self.W_v, self.W_o, self.E, self.F]:
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

    def _project_sequence(self, x, projection):
        """
        Apply a sequence-length projection (E or F) to compress the token dimension.

        nn.Linear acts on the last dimension but E and F need to act on the sequence dimension. x is transposed so seq_len is last, projected down to proj_dim, then transposed back.

        Args:
            x (Tensor): (batch, num_tokens, d_model)
            projection (nn.Linear): E or F, maps seq_len -> proj_dim

        Returns:
            Tensor: (batch, proj_dim, d_model)
        """
        x = x.transpose(1, 2) # (batch, num_tokens, d_model) -> (batch, d_model, num_tokens)
        x = projection(x) # (batch, d_model, num_tokens) -> (batch, d_model, proj_dim)
        return x.transpose(1, 2) # (batch, d_model, proj_dim) -> (batch, proj_dim, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): (batch, num_tokens, d_model)
            mask (Tensor): (batch, num_tokens), True = padding token.

        Returns:
            out (Tensor): (batch, num_tokens, d_model)
            attn_weights (Tensor): (batch, num_heads, num_tokens, proj_dim)
        """
        batch, num_tokens, _ = x.size()
        assert num_tokens == self.seq_len, f"input seq_len ({num_tokens}) does not match the seq_len this layer was built for ({self.seq_len})"

        k = self.W_k(x) # (batch, num_tokens, d_model)
        v = self.W_v(x) # (batch, num_tokens, d_model)

        # zero out padding positions before compression since after E and F are applied each compressed
        # position mixes information from every original position and can no longer be masked individually
        if mask is not None:
            keep = (~mask).unsqueeze(-1).float() # (batch, num_tokens, 1), 1=real 0=pad
            k = k * keep
            v = v * keep

        # compress K and V along the sequence dimension: num_tokens -> proj_dim
        k = self._project_sequence(k, self.E) # (batch, proj_dim, d_model)
        v = self._project_sequence(v, self.F) # (batch, proj_dim, d_model)

        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        q = self._split_heads(self.W_q(x))
        # (batch, proj_dim, d_model) -> (batch, num_heads, proj_dim, head_dim)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # (batch, num_heads, num_tokens, head_dim) @ (batch, num_heads, head_dim, proj_dim) -> (batch, num_heads, num_tokens, proj_dim)
        # K and V now have proj_dim positions instead of num_tokens so scores are (num_tokens, proj_dim) rather than (num_tokens, num_tokens)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch, num_heads, num_tokens, proj_dim) @ (batch, num_heads, proj_dim, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        context_vecs = attn_weights @ v

        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, d_model)
        out = self.dropout(self.W_o(self._merge_heads(context_vecs)))

        return out, attn_weights
    