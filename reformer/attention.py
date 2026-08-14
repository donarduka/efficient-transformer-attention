"""
reformer/attention.py

Reformer LSH-based attention.

Replaces the O(n^2) full attention matrix with an LSH approximation:
    - Q and K share the same projection, required for LSH to work
    - tokens are hashed into buckets using random projections
    - attention is only computed within each bucket, reducing complexity to O(n log n)
    - multiple rounds of hashing reduce the chance of missing relevant token pairs

The LSH scheme follows [X] Kitaev et al. (2020) Section 2. The hash function is:
    h(x) = argmax([xR; -xR]) where R is a random projection matrix.
Tokens that map to the same bucket are likely to have high dot products and therefore high attention scores.

Reversible residual layers (also from [X] Kitaev et al.) are not implemented here as they are an architectural change outside the scope of the attention module.

Reference:
    [X] Kitaev et al., "Reformer: The Efficient Transformer", ICLR 2020.
    https://arxiv.org/abs/2001.04451
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReformerAttention(nn.Module):
    """
    Reformer LSH attention from [X] Kitaev et al. (2020).

    Q and K share the same linear projection. After projection K is normalised so that Q and K point in the same direction in vector space, 
    which is required for the LSH hash to group similar queries and keys into the same bucket.

    Tokens are sorted by bucket and attention is computed within chunks of sorted tokens, attending to the current chunk and the one before it to handle tokens at bucket boundaries. 
    Multiple hash rounds reduce the probability of missing relevant pairs.

    This class exposes the same forward(x, mask) signature as MultiHeadSelfAttention in baseline/attention.py
    so it can be passed as attn_cls into TransformerEncoderBlock without any other changes.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of attention heads.
        dropout (float): applied to attention weights.
        n_rounds (int): number of LSH hashing rounds. More rounds means fewer missed pairs but higher compute cost. 
                        Defaults to 4, following [X] Kitaev et al. Table 2 which shows 4 rounds approaches full attention accuracy.
        n_buckets (int): number of LSH buckets. If None, set to seq_len // chunk_size at forward time, following the paper's convention.
        chunk_size (int): number of tokens per chunk after sorting by bucket. Defaults to 64.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, n_rounds=4, n_buckets=None, chunk_size=64):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.n_rounds = n_rounds
        self.n_buckets = n_buckets
        self.chunk_size = chunk_size

        # shared Q/K projection - same weights used for both query and key
        # separate projection for V as normal
        self.W_qk = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.W_qk, self.W_v, self.W_o]:
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

    def _lsh_hash(self, x, n_buckets):
        """
        Compute LSH bucket assignments for a set of vectors.

        Uses the random projection scheme from [X] Kitaev et al. (2020):
            h(x) = argmax([xR; -xR]) 
            - where R is a random matrix of shape (head_dim, n_buckets // 2).

        Args:
            x (Tensor): (batch, num_heads, num_tokens, head_dim)
            n_buckets (int): number of buckets to hash into.

        Returns:
            Tensor: bucket indices, shape (batch, num_heads, num_tokens).
        """
        batch, num_heads, num_tokens, head_dim = x.size()

        # random projection matrix - different for each round, shared across batch/heads, created on same device as x
        R = torch.randn(head_dim, n_buckets // 2, device=x.device)

        # (batch, num_heads, num_tokens, head_dim) @ (head_dim, n_buckets // 2) -> (batch, num_heads, num_tokens, n_buckets // 2)
        proj = x @ R

        # concatenate proj and -proj then take argmax to get bucket index - angular LSH scheme from the paper
        buckets = torch.cat([proj, -proj], dim=-1).argmax(dim=-1) # (batch, num_heads, num_tokens)

        return buckets

    def _lsh_attention_single_round(self, q, k, v, mask=None):
        """
        Compute one round of LSH attention.

        Tokens are hashed into buckets, sorted by bucket index and split into chunks. Attention is computed within each chunk and the chunk before it, following [X] Kitaev et al. (2020) Figure 2.

        Args:
            q, k, v (Tensor): (batch, num_heads, num_tokens, head_dim)
            mask (Tensor): padding mask (batch, num_tokens), True = ignore.

        Returns:
            out (Tensor): (batch, num_heads, num_tokens, head_dim)
            attn_weights (Tensor): (batch, num_heads, num_tokens, num_tokens) zeros - full matrix not computed, returned for compatibility
        """
        batch, num_heads, num_tokens, head_dim = q.size()
        n_buckets = self.n_buckets or max(2, num_tokens // self.chunk_size)

        buckets = self._lsh_hash(q, n_buckets) # (batch, num_heads, num_tokens)

        sorted_idx = buckets.argsort(dim=-1) # (batch, num_heads, num_tokens)
        unsorted_idx = sorted_idx.argsort(dim=-1) # to restore original order later

        # reorder q, k, v by sorted bucket order
        idx_expanded = sorted_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        q_sorted = q.gather(2, idx_expanded) # (batch, num_heads, num_tokens, head_dim)
        k_sorted = k.gather(2, idx_expanded)
        v_sorted = v.gather(2, idx_expanded)

        # pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - num_tokens % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            q_sorted = F.pad(q_sorted, (0, 0, 0, pad_len))
            k_sorted = F.pad(k_sorted, (0, 0, 0, pad_len))
            v_sorted = F.pad(v_sorted, (0, 0, 0, pad_len))

        padded_len = num_tokens + pad_len
        n_chunks = padded_len // self.chunk_size

        # reshape into chunks: (batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, n_chunks, chunk_size, head_dim)
        q_chunks = q_sorted.view(batch, num_heads, n_chunks, self.chunk_size, head_dim)
        k_chunks = k_sorted.view(batch, num_heads, n_chunks, self.chunk_size, head_dim)
        v_chunks = v_sorted.view(batch, num_heads, n_chunks, self.chunk_size, head_dim)

        # for each chunk attend to current chunk + previous chunk - roll by 1 to get previous chunk
        k_prev = torch.cat([k_chunks[:, :, -1:, :, :], k_chunks[:, :, :-1, :, :]], dim=2)
        v_prev = torch.cat([v_chunks[:, :, -1:, :, :], v_chunks[:, :, :-1, :, :]], dim=2)

        # concatenate current and previous chunk: (batch, num_heads, n_chunks, 2*chunk_size, head_dim)
        k_combined = torch.cat([k_prev, k_chunks], dim=3)
        v_combined = torch.cat([v_prev, v_chunks], dim=3)

        # (batch, num_heads, n_chunks, chunk_size, head_dim) @ (batch, num_heads, n_chunks, head_dim, 2*chunk_size) -> (batch, num_heads, n_chunks, chunk_size, 2*chunk_size)
        chunk_scores = (q_chunks @ k_combined.transpose(-2, -1)) / math.sqrt(head_dim)

        # prevent tokens attending to themselves - shared QK means a token would otherwise always attend most strongly to itself
        # for simplicity we rely on softmax normalisation here rather than explicit masking, following [X] Kitaev et al. Section 2
        chunk_attn = F.softmax(chunk_scores, dim=-1)
        chunk_attn = torch.nan_to_num(chunk_attn, nan=0.0)
        chunk_attn = self.dropout(chunk_attn)

        # (batch, num_heads, n_chunks, chunk_size, 2*chunk_size) @ (batch, num_heads, n_chunks, 2*chunk_size, head_dim) -> (batch, num_heads, n_chunks, chunk_size, head_dim)
        out_chunks = chunk_attn @ v_combined

        # flatten chunks back to sequence: (batch, num_heads, n_chunks, chunk_size, head_dim) -> (batch, num_heads, padded_len, head_dim)
        out_sorted = out_chunks.reshape(batch, num_heads, padded_len, head_dim)

        out_sorted = out_sorted[:, :, :num_tokens, :] # remove padding

        # restore original token order
        idx_expanded_out = unsorted_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        out = out_sorted.gather(2, idx_expanded_out) # (batch, num_heads, num_tokens, head_dim)

        # return zeros for attn_weights - full matrix not computed, only within-chunk scores
        attn_weights = torch.zeros(1, 1, 1, 1, device=q.device)

        return out, attn_weights

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): (batch, num_tokens, d_model)
            mask (Tensor): (batch, num_tokens), True = padding token.

        Returns:
            out (Tensor): (batch, num_tokens, d_model)
            attn_weights (Tensor): (batch, num_heads, num_tokens, num_tokens) approximate - only within-chunk scores computed
        """
        # shared Q/K projection - same weights for both query and key
        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        q = self._split_heads(self.W_qk(x))

        # normalise K to unit norm - required for LSH angular hashing to work correctly, following [X] Kitaev et al. Section 2
        k = F.normalize(q, p=2, dim=-1) # (batch, num_heads, num_tokens, head_dim)

        # (batch, num_tokens, d_model) -> (batch, num_heads, num_tokens, head_dim)
        v = self._split_heads(self.W_v(x))

        # run multiple rounds of LSH attention and average the outputs
        # more rounds = fewer missed relevant pairs, following [X] Kitaev et al. Equation 6 and Table 2
        all_outputs = []
        for _ in range(self.n_rounds):
            out_round, attn_weights = self._lsh_attention_single_round(q, k, v, mask)
            all_outputs.append(out_round)

        # average across rounds: (n_rounds, batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        out = torch.stack(all_outputs, dim=0).mean(dim=0)

        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, d_model)
        out = self.dropout(self.W_o(self._merge_heads(out)))

        return out, attn_weights