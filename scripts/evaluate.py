"""
scipts/evaluate.py

Evaluation and benchmarking script for all four attention variants.

Loads a trained model from results/, runs it on the test set and records:
    - accuracy
    - inference time per forward pass
    - peak memory usage

Results are saved to results/ as a JSON file for use in visualisation.

Usage:
    python3 scipts/evaluate.py --variant baseline --task listops
    python3 scipts/evaluate.py --variant longformer --task text_classification
    python3 scipts/evaluate.py --variant linformer --task lisops
    python3 scipts/evaluate.py --variant reformer --task text_classificaiton
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import load_config, build_model, load_data, get_device

# timing follows PyTorch benchmarking best practices
# https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
def measure_inference(model, input_ids, attention_mask, device, n_runs=10):
    """
    Measure average inference time and peak memory over n_runs forward passes.

    Runs a warmup pass first to avoid measuring JIT compilation overhead then averages over n_runs to get a stable estimate.

    Args:
        model: the trained model in eval mode.
        input_ids (Tensor): (batch, num_tokens)
        attention_mask (Tensor): (batch, num_tokens)
        device: torch device
        n_runs (int): number of forward passes to average over

    Returns:
        avg_time_ms (float): average forward pass time in milliseconds
        peak_memory_mb (float): peak memory allocated in MB (GPU only, 0 on CPU)
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # warmup pass
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    # reset memory stats before measuring
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()

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
                times.append((t1 - t0) * 1000) # convert to ms

    avg_time_ms = sum(times) / len(times)

    # peak memory - only available on CUDA
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        # mps and cpu dont expose peak memory in the same way
        peak_memory_mb = 0.0

    return avg_time_ms, peak_memory_mb

def evaluate(model, loader, device):
    """
    Run the model on the full test set and compute accuracy.

    Args:
        model: tained model in eval mode
        loader: test Dataloader
        device: torch device

    Returns:
        accuracy (float): fraction of correct predictions
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            out = model(input_ids, attention_mask=attention_mask)
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["baseline", "longformer", "linformer", "reformer"])
    parser.add_argument("--task", required=True, choices=["listops", "text_classification", "integration_test"])
    parser.add_argument("--n_runs", type=int, default=10, help="number of forward passes to average for timing")
    args = parser.parse_args()

    config = load_config(args.task)
    torch.manual_seed(config["seed"])
    device = get_device()
    print(f"using device: {device}")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    model_path = os.path.join(results_dir, f"{args.variant}_{args.task}_best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"no saved model found at {model_path}. run train.py first")
    
    print(f"loading {args.variant} model from {model_path}...")
    model = build_model(config, args.variant)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"loading test data for {args.task}...")
    _, test_loader = load_data(config)

    print("evaluating accuracy..")
    accuracy = evaluate(model, test_loader, device)
    print(f"accuracy: {accuracy:.4f}")

    # use the first batch for timing - representative of a single forward pass
    first_batch = next(iter(test_loader))
    input_ids, _, attention_mask = first_batch

    print(f"benchamrking inference time over {args.n_runs} runs...")
    avg_time_ms, peak_memory_mb = measure_inference(model, input_ids, attention_mask, device, n_runs=args.n_runs)
    print(f"avg inference time: {avg_time_ms:.2f}ms")
    print(f"peak memory: {peak_memory_mb:.2f}MB")

    results = {
        "variant": args.variant,
        "task": args.task,
        "accuracy": accuracy,
        "avg_inference_time_ms": avg_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "device": str(device),
        "batch_size": config["batch_size"],
        "seq_len": config["seq_len"],
        "n_runs": args.n_runs,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{args.variant}_{args.task}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()