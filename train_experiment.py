"""Reproducible train/test run for the class's modular-addition model.

Run: python train_experiment.py --epochs 200
Outputs: runs/metrics.csv, runs/loss.png, runs/accuracy.png
"""

import argparse
import csv
from pathlib import Path

import torch

import config
from data import generate_data_tensor
from loss import loss_fn
from model import model


@torch.no_grad()
def evaluate(network, inputs, targets):
    logits = network(inputs)
    loss = loss_fn(targets, logits).item()
    predicted = logits[:, -1, :].argmax(dim=-1)
    correct = targets[:, -1, :].argmax(dim=-1)
    accuracy = (predicted == correct).float().mean().item()
    return loss, accuracy


def save_plot(rows, output, key_train, key_test, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in rows]
    ax.plot(epochs, [row[key_train] for row in rows], label="train")
    ax.plot(epochs, [row[key_test] for row in rows], label="test")
    ax.set(xlabel="Training step", ylabel=ylabel)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(epochs, train_size, seed, output):
    total = config.modulus**2
    if total < 2:
        raise ValueError("config.modulus must be at least 2 for a train/test split")
    if train_size is None:
        train_size = max(1, min(total - 1, round(0.8 * total)))
    if not 0 < train_size < total:
        raise ValueError(f"train_size must be between 1 and {total - 1}")
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")

    torch.manual_seed(seed)
    # generate_data_tensor(total) shuffles all unique (a, b) pairs once.
    inputs, targets = generate_data_tensor(total)
    train_x, test_x = inputs[:train_size], inputs[train_size:]
    train_y, test_y = targets[:train_size], targets[train_size:]
    network = model()
    rows = []

    for epoch in range(epochs + 1):
        if epoch > 0:
            network.train(train_x, train_y)  # This project's method takes one optimizer step.
        train_loss, train_accuracy = evaluate(network, train_x, train_y)
        test_loss, test_accuracy = evaluate(network, test_x, test_y)
        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "test_loss": test_loss, "train_accuracy": train_accuracy,
                     "test_accuracy": test_accuracy})

    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, output / "loss.png", "train_loss", "test_loss", "Answer-token loss")
    save_plot(rows, output / "accuracy.png", "train_accuracy", "test_accuracy", "Answer accuracy")
    print(f"seed={seed}, training examples={train_size}, held-out examples={total - train_size}")
    print(f"step {epochs}: train loss={train_loss:.3f}, test loss={test_loss:.3f}; "
          f"train accuracy={train_accuracy:.1%}, test accuracy={test_accuracy:.1%}")
    print(f"Saved metrics and plots in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=None,
                        help="number of training pairs (default: about 80%% of N squared)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run(args.epochs, args.train_size, args.seed, args.output)
