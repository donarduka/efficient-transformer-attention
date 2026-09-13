"""
tests/test_longformer.py

Sanity checks for the Longformer attention module. Checks output shapes, no NaNs and that sliding window and global token behaviour work as expected.

Run from repo root with:
    python3 tests/test_longformer.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longformer.attention import LongformerAttention

#small dimensions so tests run fast - window size small enough to actually restrict attention
BATCH = 2
SEQ_LEN = 32
D_MODEL = 64
NUM_HEADS = 4
WINDOW_SIZE = 4 # small window so most tokens cant see each other

def test_forward_no_mask():
    attn = LongformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, window_size=WINDOW_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    out, weights = attn(x)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert not torch.isnan(out).any()
    print("test forward no mask passed")


def test_forward_with_mask():
    attn = LongformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, window_size=WINDOW_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # last 6 positions are padding
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, -6:] = True

    out, weights = attn(x, mask=mask)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test forward with mask passed")


def test_window_mask_restricts_attention():
    # tokens more than window_size apart should have zero attention weights
    attn = LongformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, window_size=WINDOW_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    _, weights = attn(x)

    # token 0 is global so skip it - check token at position 10
    # position 10 with window_size=4 should only attend to positions 6-14
    # position 0 (global) and anyhting outside [6-14] should be 0 or very small
    # check position 20 has zero weight from position 10 (distance > window)
    # weights shape: (batch, seq_len, seq_len)
    weight_10_to_20 = weights[0, :, 10, 20] # all heads, query10, key=20
    assert (weight_10_to_20 < 1e-6).all(), \
        f"token 10 should not attend to token 20 with window_size={WINDOW_SIZE}"
    print("test window mask restricts attention passed")


def test_global_token_sees_all():
    # position 0 is the global token - its attention weights should be non-zero across the whole sequence, not just within a local window
    attn = LongformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, window_size=WINDOW_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    _, weights = attn(x)

    # global token (pos 0) attends to all positions - check a distant position
    # weight from position 0 to position 30 should be non-zero
    weight_global_to_far = weights[0, :, 0, 30] # query=0 (global), key=30
    assert (weight_global_to_far > 0).any(), \
        "global token at position 0 should attend to distant positions"
    print("test global token sees all passed")


if __name__ == "__main__":
    print("running longformer tests...\n")
    test_forward_no_mask()
    test_forward_with_mask()
    test_window_mask_restricts_attention()
    test_global_token_sees_all()
    print("\nall tests passed")