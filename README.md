# Ordinal Information Processing Demo

## Overview

This repository is a simple reproduction of the ordinal information processing
experiments described in [`paper.pdf`](./paper.pdf). The code trains a simple
RNN on a synthetic two-item sequence task, visualizes the learned tree-like
hidden representation, analyzes nearby fixed points, and compares transfer
learning against training from scratch.

The practical target of the current version is to recover the main qualitative
behavior:

- Sec. 2 synthetic sequences produce ordered output ramps and tree-structured
  hidden trajectories.
- Stable fixed points can be found near selected nodes of the learned hidden
  representation.
- Sec. 3 transfer learning can reuse the pretrained recurrent matrix `W_rec`
  and train only the input/readout side more quickly than training all weights
  from scratch.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Model And Dynamics](#model-and-dynamics)
- [Main Code Entry Points](#main-code-entry-points)
- [Training Flow](#training-flow)
- [Fixed-Point And Stability Analysis](#fixed-point-and-stability-analysis)
- [Transfer Learning Experiment](#transfer-learning-experiment)
- [Outputs And Packaged Resources](#outputs-and-packaged-resources)
- [Recommended Workflow](#recommended-workflow)

## Quick Start

Install the default runtime dependencies:

```bash
pip install -r requirements.txt
```

Train the Sec. 2 synthetic model:

```bash
python src/train_sec2_synthetic.py
```

Plot a representative Sec. 2 model output:

```bash
python src/plot_sec2_model_output.py
```

Plot the Sec. 2 tree trajectory and stable fixed points:

```bash
python src/plot_sec2_tree_structure.py
```

Run the Sec. 3 transfer comparison:

```bash
python src/train_sec3_transfer.py
```

Plot the Sec. 3 transfer-learning curve:

```bash
python src/plot_sec3.py
```

Most scripts keep configuration as Python constants near the top of the file
rather than command-line arguments.  For plotting scripts, set
`ENABLE_PLOT_SHOW = True` if you want `plt.show()` after saving.

## Repository Layout

```bash
.
├── README.md
├── requirements.txt
├── paper.pdf
├── src/
│   ├── models.py                    # Continuous-time RNN model
│   ├── datasets.py                  # Synthetic and TIMIT-derived datasets
│   ├── train_sec2_synthetic.py      # Sec. 2 synthetic training
│   ├── plot_sec2_model_output.py    # Sec. 2 output-vs-target plot
│   ├── plot_sec2_tree_structure.py  # Sec. 2 PCA tree and fixed-point plot
│   ├── load_sec3_data.py            # Loads packaged Sec. 3 feature bundle
│   ├── build_sec3_timit_features.py # Optional full-TIMIT preprocessing
│   ├── train_sec3_transfer.py       # Sec. 3 transfer comparison
│   ├── plot_sec3.py                 # Figure 3B-style transfer curve
│   └── train_utils.py               # Legacy/shared training helper
├── resources/
│   ├── timit_phoneme_mfcc_means_mfcc16.npz
│   └── figure3b_transfer_history_seed1107.csv
├── model/                           # Checkpoints generated or supplied locally
└── figure/                          # Rendered figures
```

## Configuration

The scripts use top-level constants instead of a separate config framework.  The
most important defaults are:

| Constant | Default | Role |
| --- | ---: | --- |
| `T_MAX` | `30` | Length of the initial/final silent interval and fixed test gap |
| `PHONEME_LEN` | `3` | Duration of each item/phoneme segment |
| `HIDDEN_DIM` | `64` | Number of recurrent units |
| `TAU` | `2` | RNN time constant |
| `SEED` | `1107` | Random seed used for reproducibility |
| `SAMPLE_PER_CLASS` | script-specific | Number of generated training samples per class |
| `TEST_SAMPLE_PER_CLASS` | script-specific | Number of fixed-gap test samples per class |
| `GRAD_CLIP_NORM` | `0.1` in Sec. 3 | Gradient clipping threshold for transfer training |
| `DEVICE` | auto | Uses CUDA when `torch.cuda.is_available()` is true |

Sec. 2 plotting scripts have their own constants so they do not import training
configuration from `train_sec2_synthetic.py`.

## Model And Dynamics

The main model is `SimpleRNN` in `src/models.py`.  It is a continuous-time RNN
implemented with Euler integration:

$$
x_{t+1} = x_t + \frac{1}{\tau}(-x_t + W_{rec} \tanh(x_t) + W_{in} u_t + I_b) \\
y_t = W_{out} \tanh(x_t)
$$

Main variables:

| Symbol | Code | Meaning |
| --- | --- | --- |
| `x` | hidden state | State variable used for fixed-point analysis |
| `r = tanh(x)` | rate state | Nonlinear firing-rate representation |
| `u` | input sequence | Synthetic vector or TIMIT-derived phoneme vector |
| `y` | output sequence | Four-dimensional class/ramp output |
| `W_in` | `model.W_in` | Input projection |
| `W_rec` | `model.W_rec` | Recurrent matrix |
| `I_b` | `model.I_b` | Hidden bias current |
| `W_out` | `model.W_out` | Output readout |

All weights are trained in the Sec. 2 synthetic model.  In the Sec. 3 transfer
condition, only `W_rec` is copied from the Sec. 2 checkpoint and then frozen;
the input and output sides are trained for the TIMIT-derived task.

## Main Code Entry Points

| File | Function | Role |
| --- | --- | --- |
| `src/models.py` | `SimpleRNN` | Shared continuous-time RNN model |
| `src/datasets.py` | `SyntheticSequenceDataset` | Sec. 2 synthetic sequence dataset |
| `src/datasets.py` | `TimitSequenceDataset` | Sec. 3 sequence dataset built from packaged phoneme features |
| `src/datasets.py` | `collate_fn()` | Pads variable-length sequences and returns masks |
| `src/train_sec2_synthetic.py` | `train()` | Trains and saves a Sec. 2 checkpoint |
| `src/plot_sec2_model_output.py` | `main()` | Loads a Sec. 2 checkpoint and plots output behavior |
| `src/plot_sec2_tree_structure.py` | `main()` | Plots PCA trajectories and stable fixed points |
| `src/load_sec3_data.py` | `build_sec3_timit_sequences()` | Loads packaged Sec. 3 phoneme vectors |
| `src/build_sec3_timit_features.py` | `save_sec3_timit_features()` | Optional preprocessing from full TIMIT |
| `src/train_sec3_transfer.py` | `main()` | Runs scratch vs transfer training |
| `src/plot_sec3.py` | `plot_figure3b()` | Plots transfer accuracy versus epoch |

## Training Flow

The Sec. 2 synthetic run in `train_sec2_synthetic.py` follows this sequence:

1. Set the random seed.
2. Build four two-item synthetic classes: `ab`, `ac`, `ba`, and `bc`.
3. Generate variable-gap training sequences with `SyntheticSequenceDataset`.
4. Generate fixed-gap test sequences for evaluation.
5. Train `SimpleRNN` with masked MSE loss.
6. Clip gradients with `clip_grad_norm_(..., 1.0)`.
7. Evaluate accuracy from the final readout window.
8. Save a checkpoint under `model/`.

The dataset target is a ramping output code.  During the first item interval,
two outputs indicate the upper branch of the sequence tree.  During the final
readout interval, only the target class output ramps high.

## Fixed-Point And Stability Analysis

The tree/fixed-point plot in `plot_sec2_tree_structure.py` first estimates tree
nodes from hidden states collected around task-relevant time windows.  For each
candidate node, it searches for a nearby fixed point in the zero-input dynamics:

$$
F(x) = -x + W_{rec} \tanh(x) + W_{in}(0) + I_b
$$

A fixed point satisfies:

$$
F(x^*) = 0
$$

The script minimizes:

$$
q(x) = \frac{1}{2} ||F(x)||^2
$$

The optimization uses Adam followed by LBFGS refinement.  Candidate fixed points
are accepted only when the residual is small and the optimized point remains
near the estimated node.

Stability is checked from the local Jacobian:

$$
\frac{dF}{dx} = -I + W_{rec} \cdot \text{diag}(1 - \tanh(x)^2)
$$

The script reports both:

- continuous-time stability from the maximum real eigenvalue of `dF/dx`;
- discrete-step stability from the spectral radius of the Euler update Jacobian.

Stable accepted points are drawn as black `X` markers on the PCA tree plot.

## Transfer Learning Experiment

The Sec. 3 experiment in `train_sec3_transfer.py` compares two modes:

| Mode | Meaning |
| --- | --- |
| `scratch_all` | Initialize a new model and train all parameters |
| `sec2_pretrained_freeze_wrec` | Load only `W_rec` from the Sec. 2 checkpoint, freeze it, and train the rest |

The Sec. 3 inputs are not the full TIMIT corpus.  They are four two-phoneme
classes built from packaged 16-dimensional MFCC mean vectors:

```text
resources/timit_phoneme_mfcc_means_mfcc16.npz
```

The transfer script writes a CSV history under `csv/` and saves both trained
models under `model/`.  `plot_sec3.py` then plots test accuracy versus epoch. If
no local CSV exists, `plot_sec3.py` falls back to the packaged example history
in `resources/figure3b_transfer_history_seed1107.csv`.

## Outputs And Packaged Resources

| Path | Meaning |
| --- | --- |
| `model/rnn_mfcc3_hid64_spc*_sec2_seed*.pt` | Sec. 2 synthetic checkpoints |
| `model/sec3_transfer_compare_*_seed*.pt` | Sec. 3 transfer comparison checkpoints |
| `csv/sec3_transfer_compare_seed*.csv` | Generated transfer training history |
| `figure/sec2_model_output_class*.png` | Sec. 2 model output plot |
| `figure/figure2E_tree_with_stable_fixed_points.png` | Sec. 2 PCA tree/fixed-point plot |
| `figure/figure3B_transfer_test_accuracy.png` | Sec. 3 transfer curve |
| `resources/timit_phoneme_mfcc_means_mfcc16.npz` | Packaged Sec. 3 phoneme features |
| `resources/figure3b_transfer_history_seed1107.csv` | Packaged example transfer history |

## Recommended Workflow

For teaching demonstrations:

1. Use the packaged `resources/` files.
2. Run `python src/plot_sec3.py` first to verify plotting works without
   retraining.
3. Run `python src/plot_sec2_model_output.py` and
   `python src/plot_sec2_tree_structure.py` with supplied or freshly trained
   Sec. 2 checkpoints.

For training experiments:

1. Edit constants near the top of `src/train_sec2_synthetic.py` or
   `src/train_sec3_transfer.py`.
2. Run the training script.
3. Inspect generated checkpoints, CSV logs, and figures.
4. Keep the same `SEED`, device, and hyperparameters for reproducible classroom
   comparisons.

For figure styling:

1. Edit the corresponding plotting script.
2. Keep `ENABLE_PLOT_SHOW = False` for headless runs.
3. Re-run only the plotting script whenever possible.
