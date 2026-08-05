"""
scripts/visualise.py

Generates plots from saved evaluation results and training logs.

Reads JSON files from results/ and produces:
    - accuracy comparison bar chart across all four variants per task.
    - inference time bar chart across all four variants per task.
    - training loss curves per variant per task.

Saves plots to results/plots/ as .png and .pdf

Usage:
    python3 scripts/visualise.py --task listops
    python3 scripts/visualise.py --task text_classification
    python3 scripts/visualise.py --task integration_test
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # non-interactive backend so it works without display

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

VARIANTS = ["baseline", "longformer", "linformer", "reformer"]
COLOURS = {"baseline": "blue", "longformer": "red", "linformer": "green", "reformer": "orange"}

def load_eval_results(task):
    """ Load evaluation JSON files for all variants for the given task"""
    results = {}
    for variant in VARIANTS:
        path = os.path.join(RESULTS_DIR, f"{variant}_{task}_eval.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                results[variant] = json.load(f)
        else:
            print(f"warning: no eval results found for {variant} on {task} - skipping")
    return results

def load_training_logs(task):
    """ Load training log JSON files for all variants for the given task"""
    logs = {}
    for variant in VARIANTS:
        path = os.path.join(RESULTS_DIR, f"{variant}_{task}_log.json")
        print(f"looking for: {path}, exists: {os.path.exists(path)}") # debug
        if os.path.exists(path):
            with open(path, "r") as f:
                logs[variant] = json.load(f)
        else:
            print(f"warning: no training log found for {variant} on {task} - skipping")
    return logs

def save_plot(fig, name):
    """ save as both png (for dissertation doc) and pdf (for quality)"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), bbox_inches="tight")
    print(f"saved {name}.png and {name}.pdf")

def plot_accuracy(results, task):
    """ Bar chart comparing test accuracy across all 4 variants"""
    variants = list(results.keys())
    accuracies = [results[v]["accuracy"] * 100 for v in variants]

    fig, ax = plt.subplots()
    ax.bar(variants, accuracies)
    ax.set_xlabel("variant")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"accuracy comparison - {task}")

    save_plot(fig, f"accuracy_{task}")
    plt.close(fig)

def plot_inference_time(results, task):
    """Bar chart comparing avg inference time per forward pass"""
    variants = list(results.keys())
    times = [results[v]["avg_inference_time_ms"] for v in variants]

    fig, ax = plt.subplots()
    ax.bar(variants, times)
    ax.set_xlabel("variant")
    ax.set_ylabel("avg inference time (ms)")
    ax.set_title(f"inference time comparison - {task}")

    save_plot(fig, f"inference_time_{task}")
    plt.close(fig)

def plot_memory(results, task):
    """Bar chart comparing peak memory usage. Only available on CUDA - skippped on CPU/MPS"""
    variants = list(results, task)
    memory = [results[v]["peak_memory_mb"] for v in variants]

    # peak memory stats are only available on CUDA - MPS and CPU return 0
    if all(m == 0 for m in memory):
        print(f"skipping memory plot for {task} - peak memory only available on CUDA")
        return
    
    fig, ax = plt.subplots()
    ax.bar(variants, memory)
    ax.set_xlabel("variant")
    ax.set_ylabel("peak memory (MB)")
    ax.set_title(f"peak memory comparison - {task}")

    save_plot(fig, f"memory_{task}")
    plt.close(fig)

def plot_training_curves(logs, task):
    """
    Training loss and accuracy curves for all variants on the same axes.
        Solid lines = train
        Dashed lines = test
    Shows overfitting at a glance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for variant, log in logs.items():
        epochs = [e["epoch"] for e in log]
        train_loss = [e["train_loss"] for e in log]
        test_loss = [e["test_loss"] for e in log]
        train_acc = [e["train_acc"] * 100 for e in log]
        test_acc = [e["test_acc"] * 100 for e in log]

        # same colour per variant
        # solid for train / dashed for test - same for both subplots
        ax1.plot(epochs, train_loss, color=COLOURS[variant], label=f"{variant} train")
        ax1.plot(epochs, test_loss, color=COLOURS[variant], linestyle="--", label=f"{variant} test")

        ax2.plot(epochs, train_acc, color=COLOURS[variant], label=f"{variant} train")
        ax2.plot(epochs, test_acc, color=COLOURS[variant], linestyle="--", label=f"{variant} test")
    
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"training loss - {task}")
    ax1.legend(fontsize=7)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title(f"training accuracy - {task}")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    save_plot(fig, f"training_curves_{task}")
    plt.close(fig)



def main():
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["listops", "text_classification", "integration_test"])
    args = parser.parse_args()

    print(f"generating plots for {args.task}...\n")

    # training curves only need training logs
    logs = load_training_logs(args.task)
    print(f"loaded logs for: {list(logs.keys())}")
    if logs:
        plot_training_curves(logs, args.task)

    #accuracy, timing and memory need evaluate.py to have been run first
    results = load_eval_results(args.task)
    if results:
        plot_accuracy(results, args.task)
        plot_inference_time(results, args.task)
        plot_memory(results, args.task)
    else:
        print("no eval results found - run scripts/evaluate.py first for accuracy and timing plots")

    print(f"\nall plots saved to results/plots/")


if __name__ == "__main__":
    main()