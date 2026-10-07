# One-layer transformer class project

Build and test a small transformer for modular addition. Start with the
[class project instructions](site/index.html) and the
[PyTorch quick reference](site/torch-reference.html).

The student functions in `embedding.py`, `attention.py`, and `mlp.py` are
deliberately unfinished. `forward.py` calls those functions in sequence,
then calls the provided `unembedding.py`. The tokenizer runs before this
forward pass, and `loss.py` computes the loss afterward.
