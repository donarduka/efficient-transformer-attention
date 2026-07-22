"""
tests/test_baseline.py

Sanity checks for baseline attention module. Checks output shapes and that no NaNs appear in the forward
pass. Run from repo root with:
    python3 tests/test_baseline.py

"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.attention import (
    ScaledDotProductAttention, 
    MultiHeadSelfAttention, 
    BaselineTransformerEncoder
)

# small dimensions so tests run fast
BATCH = 2
SEQ_LEN = 32
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
NUM_CLASSES = 10
VOCAB_SIZE = 100


def test_scaled_dot_product():
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS)
    k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS)
    v = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS)

    out, weights = attn(q, k, v)

    assert out.shape == (BATCH, NUM_HEADS, SEQ_LEN, D_MODEL // NUM_HEADS)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert not torch.isnan(out).any()
    print("test scaled dot product passed")

def test_multi_head_attention():
    mhsa = MultiHeadSelfAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out, weights = mhsa(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test multi head attention passed")

def test_multi_head_attention_with_mask():
    mhsa = MultiHeadSelfAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # last 4 positions are padding
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, -4:] = True

    out, weights = mhsa(x, mask=mask)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert not torch.isnan(out).any()
    print("test multi head attention with mask passed")

def test_full_encoder_no_mask():
    model = BaselineTransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        max_seq_len=SEQ_LEN,
    )
    model.eval()

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    with torch.no_grad():
        out = model(input_ids)

    assert out["logits"].shape == (BATCH, NUM_CLASSES)
    assert not torch.isnan(out["logits"]).any()
    assert len(out["attn_weights"]) == NUM_LAYERS
    print("test full encoder no mask passed")

def test_full_encoder_with_mask():
    model = BaselineTransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        max_seq_len=SEQ_LEN,
    )
    model.eval()

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    attention_mask[1, 20:] = 0 # second sequence is shorter

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)

    assert out["logits"].shape == (BATCH, NUM_CLASSES)
    assert not torch.isnan(out["logits"]).any()
    print("test full encoder with mask passed")


if __name__ == "__main__":
    print("running baseline tests...\n")
    test_scaled_dot_product()
    test_multi_head_attention()
    test_multi_head_attention_with_mask()
    test_full_encoder_no_mask()
    test_full_encoder_with_mask()
    print("\nall tests passed")