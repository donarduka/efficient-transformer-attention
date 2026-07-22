"""
tests/test_reformer.py

Sanity checks for the Reformer LSH attention module. Checks output shapes, no NaNs and that multi-round averaging 
produces a different result to a single round (confirming rounds are actually doing different things).

Run from repo root with:
    python3 tests/test_reformer.py

"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reformer.attention import ReformerAttention

BATCH = 2
SEQ_LEN = 128 # longer than other tests since Reformer is designed for longer sequences.
D_MODEL = 64
NUM_HEADS = 4
CHUNK_SIZE = 16

def test_forward_no_mask():
    attn = ReformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, n_rounds=2, chunk_size=CHUNK_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    out, weights = attn(x)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert not torch.isnan(out).any()
    print("test forward no mask passed")

def test_forward_with_mask():
    attn = ReformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, n_rounds=2, chunk_size=CHUNK_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # last 16 positions are padding
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, -16:] = True

    out, weights = attn(x, mask=mask)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test forward with mask passed")

def test_multi_round_differs_from_single_round():
    # each round uses a different random projection matrix so the outputs should differ - avergaing multiple rounds should give a different result to just one round
    torch.manual_seed(0)
    attn_single = ReformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, n_rounds=1, chunk_size=CHUNK_SIZE)
    attn_multi = ReformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, n_rounds=4, chunk_size=CHUNK_SIZE)

    # copy weights so the only difference is number of rounds
    attn_multi.load_state_dict(attn_single.state_dict())

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        out_single, _ = attn_single(x)
        out_multi, _ = attn_multi(x)

    # outputs should differ since more rounds = different random projections each time
    assert not torch.allclose(out_single, out_multi), \
        "multi-round output should differ from single round"
    print("test multi round differs from single round passed")

def test_different_sequence_lengths():
    # n_buckets is set dynamically so different seq lengths should work
    attn = ReformerAttention(D_MODEL, NUM_HEADS, dropout=0.0, n_rounds=2, chunk_size=CHUNK_SIZE)

    for seq_len in [64, 128, 256]:
        x = torch.randn(BATCH, seq_len, D_MODEL)
        out, _ = attn(x)
        assert out.shape == (BATCH, seq_len, D_MODEL)
        assert not torch.isnan(out).any()

    print("test differet sequence lengths passed")

if __name__ == "__main__":
    print("running reformer tests...\n")
    test_forward_no_mask()
    test_forward_with_mask()
    test_multi_round_differs_from_single_round()
    test_different_sequence_lengths()
    print("\nall tests passed")