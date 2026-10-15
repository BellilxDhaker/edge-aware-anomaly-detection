# Edge-Aware Multi-Scale Autoencoder for Unsupervised Industrial Anomaly Detection

Research project on unsupervised industrial anomaly detection with the
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) dataset.
The final system will compare a from-scratch CNN autoencoder baseline against
multi-scale anomaly scoring, an edge-aware reconstruction loss, and their
combination. No pretrained vision backbone will be used.

**Status: Phase 2 - baseline convolutional autoencoder implemented and
trainable end to end (`notebooks/03_baseline_autoencoder.ipynb`). Research
improvements (multi-scale scoring, edge-aware loss, evaluation/ablation)
are still to come.**

## Development workflow

- **Editor:** Visual Studio Code with the official **Google Colab** extension
  (`Select Kernel` → `Colab` → `Auto Connect`).
- **Compute:** Google Colab runtimes (GPU for training). The kernel runs on
  Google's servers and **cannot see your local disk**.
- **Getting project code onto the runtime:** in the VS Code Explorer,
  right-click the repository folder → **Upload to Colab**, then run
  `notebooks/01_environment_setup.ipynb`. Re-upload after editing files inside
  `src/` (the environment notebook prints a checksum so stale uploads are
  detectable). Notebooks themselves always execute from your local files.
- **Persistence:** code and notebooks live on your disk (and later on GitHub);
  large artifacts (dataset, checkpoints) go to a mounted Google Drive, never
  only to the runtime's `/content` filesystem.
- **Version control:** this repository. Datasets, checkpoints and generated
  outputs are excluded by `.gitignore`.

## Repository layout

```text
edge-aware-anomaly-detection/
├── notebooks/
│   ├── 01_environment_setup.ipynb   # verify runtime, install missing deps, import src, seeds
│   ├── 02_dataset_inspection.ipynb  # download/verify MVTec AD, then dataset validation
│   └── 03_baseline_autoencoder.ipynb# train the from-scratch AE baseline on bottle
├── src/
│   ├── __init__.py
│   ├── config.py                    # SEED, set_seed(), find_repo_root(), Paths, get_device(),
│   │                                # ExperimentConfig + load/save_experiment_config()
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py               # find_dataset_root(), scan_category(), summarize_layout(),
│   │                                # NormalImageDataset, create_dataloaders()
│   ├── models/
│   │   ├── __init__.py              # Autoencoder re-exports, tensor_shape_report()
│   │   ├── encoder.py               # Encoder + ConvBlock (random init, no skips)
│   │   ├── decoder.py               # Decoder (nearest-upsample + sigmoid head)
│   │   └── autoencoder.py           # Autoencoder = encoder + decoder, param counting
│   ├── losses/
│   │   ├── __init__.py
│   │   └── reconstruction.py        # MSE criterion, per_image_mse, absolute_error_map
│   └── training/
│       ├── __init__.py
│       └── trainer.py               # Trainer (fit/validate/checkpoint/resume), build_optimizer
├── configs/
│   └── default.yaml                 # baseline experiment configuration (single source of truth)
├── requirements.txt                 # dependencies (installed only if missing)
├── .env.example                     # optional notebook overrides (defaults live in the notebooks)
├── .env                             # your local overrides (gitignored)
├── tests/                           # planned
└── outputs/                         # git-ignored run artifacts (checkpoints, figures, history)
```

Planned additions: `notebooks/04_...` through `06_evaluation_and_ablation.ipynb`,
plus `src/anomaly/` and `src/evaluation/`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_environment_setup.ipynb` | Verify Python/PyTorch/GPU, install missing packages only, make `src` importable, set reproducibility seeds |
| `notebooks/02_dataset_inspection.ipynb` | Download and verify the MVTec AD archive, then validate the extracted dataset |
| `notebooks/03_baseline_autoencoder.ipynb` | Train the from-scratch convolutional autoencoder baseline on normal `bottle` images; reconstruction analysis on normal data only |

## The baseline (Phase 2)

A compact convolutional autoencoder trained **from random initialisation** on
`bottle/train/good` only:

- 256x256 RGB inputs, `ToTensor` to float32 `[0, 1]`; decoder ends in a
  sigmoid so both sides share that range.
- Encoder: 5 stages of `Conv3x3 -> BatchNorm -> ReLU` (x2) + `MaxPool2d(2)`,
  then a 1x1 bottleneck to a `256x8x8` latent (12x fewer elements than the
  input). No skip connections - the decoder only sees the compressed code.
- Decoder: nearest-neighbour upsampling + mirrored conv stages + sigmoid head.
- Loss: mean squared error (`src/losses/reconstruction.py`); optimizer Adam,
  defaults in `configs/default.yaml` (batch 16, lr 1e-3, 30 epochs).
- `Trainer` records per-epoch train/validation losses, detects non-finite
  losses, saves `last.pt`/`best.pt`, `history.json`, `config.yaml` and
  `metadata.json` under `outputs/runs/<experiment_name>/` (git-ignored), and
  supports resuming from a checkpoint.

Anomalous test images, ground-truth masks and pretrained weights are never
used for training, tuning or checkpoint selection.

## Local environment file (`.env`)

Optional overrides for notebook settings — the defaults live in the notebooks,
so `.env` is **not** required to run anything:

1. Copy `.env.example` to `.env`.
2. Change a value only if you need to (e.g. a different `KAGGLE_DATASET` handle).
3. Re-upload the repo folder to Colab (right-click → **Upload to Colab**) so the
   runtime can read it.

`.env` is gitignored; only `.env.example` (defaults) is tracked.

## Data

The MVTec AD dataset is **not** included in this repository (license and size).
`notebooks/02_dataset_inspection.ipynb` downloads it from the public Kaggle
mirror `thtuan/mvtecad-mvtec-anomaly-detection` (via `kagglehub`, no credentials)
into Colab disk (`/content/datasets/mvtec/`, wiped on disconnect — re-run the
cell per session). The original CC BY-NC-SA 4.0 terms still apply: attribute
MVTec Software GmbH, non-commercial use. The dataset root can be redirected
with the `MVTEC_ROOT` environment variable.

## Reproducibility

`src/config.py` exposes `SEED` and `set_seed()`, which seed Python's `random`,
NumPy (if installed) and PyTorch/CUDA (if installed), and force cuDNN toward
deterministic kernels. Every training entry point calls it. Experiment
hyperparameters live in `configs/default.yaml`, loaded through
`load_experiment_config()` (validated, YAML + notebook overrides), and the
fully resolved config is saved next to each run's checkpoints.
