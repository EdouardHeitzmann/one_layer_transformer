# one_layer_transformer
implements a one-layer transformer and trains on modular arithmetic

The class handout calls the modulus **N**; set its value with `modulus` in `config.py`.

Class repository: [jonathanwise/one_layer_transformer](https://github.com/jonathanwise/one_layer_transformer), using the `project` branch.
Class instructions: [site/index.html](site/index.html). PyTorch reference: [site/torch-reference.html](site/torch-reference.html).

Run the reference tests with `python -m pytest -q` after installing `requirements.txt`.
Run `python train_experiment.py --epochs 200` to produce loss and accuracy plots in `runs/`.
