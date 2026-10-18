"""Train the one-layer transformer on addition mod N with a triangle-wave MLP activation.

The MLP's ReLU is replaced by  2 * | (x mod P) - P/2 |,  a continuous, periodic,
piecewise-linear function with period P. Everything else (data, loss, optimizer,
attention, embeddings) is the repository's code unchanged.

Run from the repository root, e.g.

    python train_triangle.py --period 2 --steps 3000
    python train_triangle.py --period 43 --wp-zero          # also hard-codes W_p = 0

Outputs, in --output (default experiment_results/):
    triangle_P<period>_seed<seed>.pt          weights (CPU tensors), loadable with load_triangle.py
    triangle_P<period>_seed<seed>_metrics.csv train/test loss and accuracy every --log-every steps
"""

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import torch

from attention import attention
from config import default_config
from data import generate_data_tensor
from embedding import embedding
from loss import loss_fn
from model import model
from unembedding import unembedding


def triangle_wave(period):
    "Continuous periodic activation: 2 * |(x mod P) - P/2|, ranging over [0, P]."
    return lambda x: 2 * torch.abs(torch.remainder(x, period) - period / 2)


class triangle_model(model):
    "The repository's transformer with the MLP's ReLU replaced by a triangle wave."

    def __init__(self, conf, period):
        super().__init__(conf)
        self.period = period
        self.activation = triangle_wave(period)

    def forward(self, x):
        x1 = attention(self, embedding(self, x))
        hidden = self.activation(x1 @ self.W_i + self.b_i)
        return unembedding(self, hidden @ self.W_o + self.b_o)


def make_split(conf, train_frac, split_seed, device):
    "Shuffle all N**2 equations with a fixed seed and split them; same split for every run."
    torch.manual_seed(split_seed)
    X, Y = generate_data_tensor(conf)
    n_train = round(train_frac * len(X))
    return tuple(t.to(device) for t in (X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]))


@torch.no_grad()
def evaluate(net, X, Y):
    "Answer-token loss and accuracy."
    logits = net(X)
    loss = loss_fn(Y, logits).item()
    accuracy = (logits[:, -1].argmax(-1) == Y[:, -1].argmax(-1)).float().mean().item()
    return loss, accuracy


def freeze_positional_embedding(net):
    "Hard-code W_p = 0. With no gradient, AdamW skips it entirely (weight decay included)."
    with torch.no_grad():
        net.W_p.zero_()
    net.W_p.requires_grad_(False)


def train(net, X_train, Y_train, X_test, Y_test, steps, log_every):
    "Full-batch training. Returns one metrics row per logged step."
    rows = []
    start = time.time()
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            train_loss, train_acc = evaluate(net, X_train, Y_train)
            test_loss, test_acc = evaluate(net, X_test, Y_test)
            rows.append({"step": step, "train_loss": train_loss, "test_loss": test_loss,
                         "train_acc": train_acc, "test_acc": test_acc})
            print(f"step {step:5d}  train loss {train_loss:.4f}  acc {train_acc:.3f}   "
                  f"test loss {test_loss:.4f}  acc {test_acc:.3f}   ({time.time() - start:.0f}s)")
        if step < steps:
            net.train(X_train, Y_train)   # the repository's name for one AdamW step
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", type=float, default=2.0, help="period P of the triangle wave")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1, help="seed for the weight initialization")
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the train/test shuffle")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--wp-zero", action="store_true", help="hard-code the positional embedding W_p to 0")
    parser.add_argument("--output", type=Path, default=Path("experiment_results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tag = f"triangle_P{args.period:g}_seed{args.seed}" + ("_wp0" if args.wp_zero else "")
    conf = dataclasses.replace(default_config, logdir=f"runs/{tag}")
    X_train, Y_train, X_test, Y_test = make_split(conf, args.train_frac, args.split_seed, args.device)
    print(f"modulus {conf.modulus}, {conf.num_heads} head(s): {len(X_train)} training pairs, "
          f"{len(X_test)} held out, device {args.device}")

    torch.manual_seed(args.seed)
    net = triangle_model(conf, args.period).to(args.device)
    if args.wp_zero:
        freeze_positional_embedding(net)

    rows = train(net, X_train, Y_train, X_test, Y_test, args.steps, args.log_every)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({name: tensor.cpu() for name, tensor in net.state_dict().items()}, args.output / f"{tag}.pt")
    with (args.output / f"{tag}_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output / tag}.pt and {args.output / tag}_metrics.csv")


if __name__ == "__main__":
    main()
