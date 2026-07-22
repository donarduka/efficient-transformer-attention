"""
tests/test_linformer.py

Sanity checks for the Linformer attention module. Checks output shapes, no NaNs, that the projection
dimension is correct and that the seq_len assertion words.
Run repo from root with:
    python3 tests/test_linformer.py

"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linformer.attention import LinformerAttention

BATCH = 2
SEQ_LEN = 32
D_MODEL = 64
NUM_HEADS = 4
PROJ_DIM = 16 # small projection dimension for testing


def test_forward_no_mask():
    attn = LinformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, seq_len=SEQ_LEN, proj_dim=PROJ_DIM)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    out, weights = attn(x)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test forward no mask passed")

def test_forward_with_mask():
    attn = LinformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, seq_len=SEQ_LEN, proj_dim=PROJ_DIM)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # last 6 positons are padding
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, -6:] = True

    out, weights = attn(x, mask=mask)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test forward with mask passed")

def test_attention_scores_are_compressed():
    # key difference from baseline - attention scores should be (seq_len, proj_dim) not (seq_len, seq_len)
    attn = LinformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, seq_len=SEQ_LEN, proj_dim=PROJ_DIM)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    _, weights = attn(x)

    assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, PROJ_DIM), \
        f" expected compressed attention shape (batch, heads, seq_len, proj_dim) but got {weights.shape}"
    print("test attention scores are compressed passed")

def test_wrong_seq_len_raises():
    # linformer requires a fixed seq_len at construction - should fail if input doesn't match
    attn = LinformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, seq_len=SEQ_LEN, proj_dim=PROJ_DIM)

    # feed a different sequence length
    x = torch.randn(BATCH, SEQ_LEN + 10, D_MODEL)

    try:
        out, _ = attn(x)
        assert False, "should have raised AssertionError for wrong seq_len"
    except AssertionError:
        print("test wrong seq_len raises passed")


if __name__ == "__main__":
    print("running linformer tests...\n")
    test_forward_no_mask()
    test_forward_with_mask()
    test_attention_scores_are_compressed()
    test_wrong_seq_len_raises()
    print("\nall tests passed")