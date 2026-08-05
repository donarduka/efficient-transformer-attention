"""
scripts/benchmark_seq_len.py

Benchmarks inference time and peak memory for all four attention variants across multiple sequence lengths using random dummy input.

No trained model or real data needed - just measures how long a forward pass takes and how much memory
it uses at each sequence length. This directly tests the complexity claims from the papers:
- baseline -> O(n^2)
- longformer -> O(n * window_size)
- linformer -> O(n * proj_dim)
- reformer -> O(n long n)

The Reformer paper [X] Kitaev et al. (2020) Figure 5 shows LSH only beats full attention beyond ~2048 tokens
    - this benchmark replicates that experiment.

Results are saved to results/benchmark_seq_len.json for plotting with visualise.py

Usage: python3 scripts/benchmark_seq_len.py
"""

import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import build_model, get_device

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

VARIANTS = ["baseline", "longformer", "linformer", "reformer"]

# sequence lengths to benchmark - covers short to full LRA lengths
SEQ_LENGTHS = [128, 256, 512, 1024, 2048]

# small model config - same across all variants so compairson is fair
# not using full d_model=512 here as baseline at seq_len=2048 as OOM's Macbook Pro M3
BENCH_CONFIG = {
    "d_model" : 128,
    "num_heads" : 4,
    "num_layers" : 2,
    "d_ff" : 512,
    "dropout" : 0.0, # no dropout during benchmarking
    "vocab_size" : 100,
    "num_classes" : 10,
    "batch_size" : 1,
    "longformer_window_size" : 64,
    "linformer_proj_dim" : 64,
    "reformer_n_rounds" : 4,
    "reformer_chunk_size" : 32,
    "reformer_n_buckets" : 8,
}

N_RUNS = 20 # number of forward passes to avg over

def benchmark_variant(variant, seq_len, device, n_runs=N_RUNS):
    """
    Measure average inference time and peak memory for noe variant at on sequence length.

    Runs a warmup pass first to avoid measuring JIT/compilation overhead then averages over n_runs forward passes.

    Args:
        variant (str): attention variant name
        seq_len (int): sequence length to benchmark
        device: torch device.
        n_runs (int): number of forward passes to average

    Returns:
        avg_time_ms (float): average forward pass time in milliseconds
        peak_memory_mb (float): peak memory in MB (CUDA only, 0 on CPU/MPS)
    """

    # linformer needs seq_len at construction time
    config = dict(BENCH_CONFIG)
    config["seq_len"] = seq_len

    model = build_model(config, variant)
    model = model.to(device)
    model.eval()

    # random dummpy input - we only care about time and memory, not accuracy
    input_ids = torch.randint(0, config["vocab_size"], (1, seq_len)).to(device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    # warmup pass
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_ids, attention_mask=attention_mask)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(input_ids, attention_mask=attention_mask)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

    avg_time_ms = sum(times) / len(times)

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        # MPS and CPU dont expose peak memory stats same way
        peak_memory_mb = 0.0

    return avg_time_ms, peak_memory_mb


def main():
    device = get_device()
    print(f"using device: {device}")
    print(f"benchmarking {len(VARIANTS)} variants across {len(SEQ_LENGTHS)} sequence lengths... \n")

    results = {}

    for variant in VARIANTS:
        results[variant] = {"seq_lengths": [], "avg_time_ms": [], "peak_memory_mb": []}

        for seq_len in SEQ_LENGTHS:
            print(f"{variant} - seq_len={seq_len}...", end=" ", flush=True)

            try:
                avg_time, peak_mem = benchmark_variant(variant, seq_len, device)
                results[variant]["seq_lengths"].append(seq_len)
                results[variant]["avg_time_ms"].append(avg_time)
                results[variant]["peak_memory_mb"].append(peak_mem)
                print(f"{avg_time:.1f}ms")

            except RuntimeError as e:
                # OOM or other runtime error - skip this seq_len for this variant
                print(f"skipped ({e})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark_seq_len.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nbenchmark complete. results saved to {out_path}")
    print("run visualise.py to generate plots from these results")

if __name__ == "__main__":
    main()