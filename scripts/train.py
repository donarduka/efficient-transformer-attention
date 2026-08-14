"""
scripts/train.py

Shared training loop for all four attention variants

Loads the config for the specified task, builds the model with the specified attention variant, trains on the LRA dataset and saves the trained model and training log to results/.

Usage:
    python3 scipt/train.py --variant baseline --task listops
    python3 scipt/train.py --variant longformer --task text_classification
    python3 scipt/train.py --variant linformer --task listops
    python3 scipt/train.py --variant reformer --task text_classification

All four variants run through the same BaselineTransformerEncoder stack - only the attention class changes, satisfying the unified evaluation framework.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.attention import (BaselineTransformerEncoder, MultiHeadSelfAttention)
from longformer.attention import LongformerAttention
from linformer.attention import LinformerAttention
from reformer.attention import ReformerAttention

def load_config(task, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", f"{task}.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def build_model(config, variant):
    """
    Build the encoder with the specified attention variant.

    All four variants use the same BaselineTransformerEncoder - only attn_cls changes.
    Variant specified kwargs (keyword arguments) are passed via a closure so the encoder block's attn_cls(d_model, num_heads, dropout) call still works for all varaints.
    """
    if variant == "baseline":
        attn_cls = MultiHeadSelfAttention

    elif variant == "longformer":
        window_size = config["longformer_window_size"]
        def attn_cls(d_model, num_heads, dropout):
            return LongformerAttention(d_model, num_heads, dropout=dropout, window_size=window_size)
        
    elif variant == "linformer":
        proj_dim = config["linformer_proj_dim"]
        seq_len = config["seq_len"]
        def attn_cls(d_model, num_heads, dropout):
            return LinformerAttention(d_model, num_heads, dropout=dropout, seq_len=seq_len, proj_dim=proj_dim)
        
    elif variant == "reformer":
        n_rounds = config["reformer_n_rounds"]
        chunk_size = config["reformer_chunk_size"]
        n_buckets = config["reformer_n_buckets"]
        def attn_cls(d_model, num_heads, dropout):
            return ReformerAttention(d_model, num_heads, dropout=dropout, n_rounds=n_rounds, chunk_size=chunk_size, n_buckets=n_buckets)
        
    else:
        raise ValueError(f"unknown variant: {variant}")
    
    model = BaselineTransformerEncoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        max_seq_len=config["seq_len"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        attn_cls=attn_cls,
    )

    return model

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # apple silicon
    else:
        return torch.device("cpu")
    
def load_data(config):
    """
    Load the LRA dataset for the specified task using tensorflow_datasets.
    Returns train and test DataLoaders. The LRA datasets are loaded as described in [X] Tay et al. (2020).
    """
    from torch.utils.data import DataLoader, TensorDataset

    task = config["task"]
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]

    if task == "listops":
        # listops is not directly in tfds - load from local data directory
        # runs scripts/download_data.py first to generate the dataset
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "listops")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"ListOp data not founds at {data_dir}. Run scripts/download_data.py first")
        train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
        test_data = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)

    elif task == "text_classification":
        import tensorflow_datasets as tfds
        from torch.utils.data import Dataset

        class IMDbDataset(Dataset):
            """
            Lazy-loading dataset for IMDb byte-level text classification.
            Loads one example at a time instead of all 25000 into memory at once.
            This avoids OOM errors when seq_len=4096"""
            def __init__(self, ds, seq_len):
                self.examples = []
                for text, label in tfds.as_numpy(ds):
                    tokens = list(text[:seq_len])
                    pad_len = seq_len - len(tokens)
                    mask = [1] * len(tokens) + [0] * pad_len
                    tokens = tokens + [0] * pad_len
                    self.examples.append((tokens, int(label), mask))

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                tokens, label, mask = self.examples[idx]
                return (
                    torch.tensor(tokens, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.long),
                )

        # imdb is available directly via tfds
        ds_train = tfds.load("imdb_reviews/bytes", split="train", as_supervised=True)
        ds_test = tfds.load("imdb_reviews/bytes", split="test", as_supervised=True)
        train_data = IMDbDataset(ds_train, seq_len)
        test_data = IMDbDataset(ds_test, seq_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids, labels, attention_mask = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask=attention_mask)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            out = model(input_ids, attention_mask=attention_mask)
            loss = criterion(out["logits"], labels)

            total_loss += loss.item()
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["baseline", "longformer", "linformer", "reformer"])
    parser.add_argument("--task", required=True, choices=["listops", "text_classification", "integration_test"])
    parser.add_argument("--config", default=None, help="optional path to config file, overrides default")
    args = parser.parse_args()

    config = load_config(args.task, config_path=args.config)
    torch.manual_seed(config["seed"]) # fix seed for reproducibility

    device = get_device()
    print(f"using device: {device}")

    print(f"loading data for {args.task}..")
    train_loader, test_loader = load_data(config)

    print(f"building {args.variant} model..")
    model = build_model(config, args.variant)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # linear warmup scheduler
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log = []
    best_acc = 0.0

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()

        print(f"epoch {epoch+1}/{config['epochs']} - train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

        log.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})

        # save best model
        if test_acc > best_acc:
            best_acc = test_acc
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(results_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(results_dir, f"{args.variant}_{args.task}_best.pt"))

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{args.variant}_{args.task}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\ntraining complete. best test accuracy: {best_acc:.4f}")
    print(f"log saved to {log_path}")


if __name__ == "__main__":
    main()