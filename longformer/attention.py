"""
longformer/attention.py

Longformer sliding window attention with global tokens.

Replaces the O(n^2) full attention matrix with a sparse attention pattern:
    - most tokens attend to a local window of neighbours (sliding window attention)
    - position 0 is a global token that attends to and is attended to by all tokens

This reduces complexity from O(n^2) to O(n * window_size), making it practical for longer sequences.
The window size can be changed to allow experiments across different hardware.

Two sets of Q, K, V projections are used - one for local window attention and one for global attention -
following Beltagy et al. (2020) Section 3.1 which shows that separate projections are critical for
downstream performance (Table 10).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LongformerAttention(nn.Module):
    """
    Longformer attention combining sliding window and global attention.

    Each token attends to a local window of size window_size around it. Position 0 is treated as a global token - it attends to all positions and vice versa. 
    This gives the model a way to aggregate document-level context needed for classification.

    Two separate sets of Q, K, V projections are used for local and global attention respectively, following Beltagy et al. (2020) Section 3.1.

    This class exposes the same forward(x, mask) signature as MultiHeadSelfAttention in baseline/attention.py so it can be passed as attn_cls into TransformerEncoderBlock without any other changes.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of attention heads.
        dropout (float): applied to attention weights.
        window_size (int): number of tokens each token attends to on each side.
        Total local window is 2 * window_size + 1. Defaults to 64, adjust based on available hardware.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, window_size=64):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        # local projections - used for sliding window attention
        self.W_qs = nn.Linear(d_model, d_model, bias=False)
        self.W_ks = nn.Linear(d_model, d_model, bias=False)
        self.W_vs = nn.Linear(d_model, d_model, bias=False)

        # global projections - used for global token at position 0
        # initialised to match local projections as recommended in Beltagy et al. (2020).
        self.W_qg = nn.Linear(d_model, d_model, bias=False)
        self.W_kg = nn.Linear(d_model, d_model, bias=False)
        self.W_vg = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.W_qs, self.W_ks, self.W_vs, self.W_qg, self.W_kg, self.W_vg, self.W_o]:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(self.W_o.bias)

        # initialise global projections to match local ones as per Beltagy et al. (2020).
        self.W_qg.weight.data.copy_(self.W_qs.weight.data)
        self.W_kg.weight.data.copy_(self.W_ks.weight.data)
        self.W_vg.weight.data.copy_(self.W_vs.weight.data)

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

    def _sliding_window_attention(self, q, k, v, mask=None):
        """
        Compute local attention for each token within its window.

        For each token at position i, attention is computed only over positions [i - window_size, i + window_size].
        Everything outside is masked to -inf.

        A production implementation would use banded matrix multiplication for efficiency but this simpler approach is sufficient for our sequence lengths and is easier to follow and explain.

        Args:
            q, k, v (Tensor): (batch, num_heads, num_tokens, head_dim)
            mask (Tensor): padding mask (batch, num_tokens), True = ignore

        Returns:
            out (Tensor): (batch, num_heads, num_tokens, head_dim)
            attn_weights (Tensor): (batch, num_heads, num_tokens, num_tokens)
        """
        batch, _, num_tokens, head_dim = q.size()

        # (batch, num_heads, num_tokens, head_dim) @ (batch, num_heads, head_dim, num_tokens) -> (batch, num_heads, num_tokens, num_tokens)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        # build the window mask - positions outside the window get -inf
        window_mask = self._build_window_mask(num_tokens, q.device) # (num_tokens, num_tokens)
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(2) # (batch, 1, 1, num_tokens)
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # (batch, num_heads, num_tokens, num_tokens) @ (batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        out = attn_weights @ v

        return out, attn_weights

    def _build_window_mask(self, num_tokens, device):
        """
        Build a boolean mask that is True outside the sliding window.

        For each query position i, positions outside [i - window_size, i + window_size] are True meaning they get masked out before softmax.

        Args:
            num_tokens (int): sequence length.
            device: torch device to create the mask on.

        Returns:
            Tensor: boolean mask of shape (num_tokens, num_tokens).
        """
        positions = torch.arange(num_tokens, device=device) # (num_tokens,)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() # (num_tokens, num_tokens)
        # True where distance exceeds window - these positions get masked out
        return dist > self.window_size

    def _global_token_attention(self, q_local, k_local, v_local, q_global, k_global, v_global, mask=None):
        """
        Compute attention for the global token at position 0.

        The global token attends to all positions using its global projections. All other tokens also attend back to the global token, with their score for position 0 replaced by a global score.

        Args:
            q_local, k_local, v_local (Tensor): local projections, (batch, num_heads, num_tokens, head_dim)
            q_global, k_global, v_global (Tensor): global projections, same shape

        Returns:
            global_out (Tensor): updated output for position 0, (batch, num_heads, 1, head_dim)
            scores_to_global (Tensor): scores from all tokens to global token, (batch, num_heads, num_tokens, 1)
            global_val (Tensor): value at position 0, (batch, num_heads, 1, head_dim)
            attn_global (Tensor): attention weights for global token, (batch, num_heads, 1, num_tokens)
        """
        global_query = q_global[:, :, 0:1, :] # (batch, num_heads, 1, head_dim)

        # (batch, num_heads, 1, head_dim) @ (batch, num_heads, head_dim, num_tokens) -> (batch, num_heads, 1, num_tokens)
        scores_global = (global_query @ k_global.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(2) # (batch, 1, 1, num_tokens)
            scores_global = scores_global.masked_fill(pad_mask, float("-inf"))

        attn_global = F.softmax(scores_global, dim=-1) # (batch, num_heads, 1, num_tokens)
        attn_global = torch.nan_to_num(attn_global, nan=0.0)
        attn_global = self.dropout(attn_global)

        # (batch, num_heads, 1, num_tokens) @ (batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, 1, head_dim)
        global_out = attn_global @ v_global

        global_key = k_global[:, :, 0:1, :] # (batch, num_heads, 1, head_dim)
        global_val = v_global[:, :, 0:1, :] # (batch, num_heads, 1, head_dim)

        # (batch, num_heads, num_tokens, head_dim) @ (batch, num_heads, head_dim, 1) -> (batch, num_heads, num_tokens, 1)
        scores_to_global = (q_local @ global_key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        return global_out, scores_to_global, global_val, attn_global

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): (batch, num_tokens, d_model)
            mask (Tensor): (batch, num_tokens), True = padding token.

        Returns:
            out (Tensor): (batch, num_tokens, d_model)
            attn_weights (Tensor): (batch, num_heads, num_tokens, num_tokens)
        """
        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        q_local = self._split_heads(self.W_qs(x))
        k_local = self._split_heads(self.W_ks(x))
        v_local = self._split_heads(self.W_vs(x))

        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        q_global = self._split_heads(self.W_qg(x))
        k_global = self._split_heads(self.W_kg(x))
        v_global = self._split_heads(self.W_vg(x))

        # compute sliding window attention for all tokens
        out, attn_weights = self._sliding_window_attention(q_local, k_local, v_local, mask)
        out = out.clone()
        attn_weights = attn_weights.clone()

        # compute global token attention and update position 0
        global_out, scores_to_global, global_val, attn_global = self._global_token_attention(
            q_local, k_local, v_local, q_global, k_global, v_global, mask
        )

        # replace position 0 output with global attention output
        out[:, :, 0:1, :] = global_out # (batch, num_heads, 1, head_dim)

        # write global attention weights back so position 0 row reflects full sequence attention
        attn_weights[:, :, 0:1, :] = attn_global

        # update all tokens to include attention to the global token
        attn_to_global = F.softmax(scores_to_global, dim=-1) # (batch, num_heads, num_tokens, 1)
        attn_to_global = torch.nan_to_num(attn_to_global, nan=0.0)

        # (batch, num_heads, num_tokens, 1) @ (batch, num_heads, 1, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        global_contribution = attn_to_global @ global_val
        out = out + global_contribution

        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, d_model)
        out = self.dropout(self.W_o(self._merge_heads(out)))

        return out, attn_weights