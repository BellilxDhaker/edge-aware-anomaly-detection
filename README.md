# Edge-Aware Multi-Scale Autoencoder for Unsupervised Industrial Anomaly Detection

Research project on unsupervised industrial anomaly detection with the
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) dataset.
The final system will compare a from-scratch CNN autoencoder baseline against
multi-scale anomaly scoring, an edge-aware reconstruction loss, and their
combination. No pretrained vision backbone will be used.

**Status: Phase 1 — environment setup and dataset inspection. No model code yet.**

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
│   └── 01_environment_setup.ipynb   # verify runtime, install missing deps, import src, seeds
├── src/
│   ├── __init__.py
│   └── config.py                    # SEED, set_seed(), find_repo_root(), Paths, get_device()
├── requirements.txt                 # Phase-1 dependencies (installed only if missing)
├── tests/                           # planned (Phase 2+)
├── configs/                         # planned (YAML experiment configs)
└── outputs/                         # git-ignored experiment artifacts
```

Planned additions: `notebooks/02_dataset_inspection.ipynb` … `06_evaluation_and_ablation.ipynb`,
plus `src/data/`, `src/models/`, `src/losses/`, `src/anomaly/`, `src/evaluation/`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_environment_setup.ipynb` | Verify Python/PyTorch/GPU, install missing packages only, make `src` importable, set reproducibility seeds |

## Data

The MVTec AD dataset is **not** included in this repository (license and size).
See the dataset notebook for the official download procedure. The dataset root
can be redirected with the `MVTEC_ROOT` environment variable, e.g. to a mounted
Google Drive path on a Colab runtime.

## Reproducibility

`src/config.py` exposes `SEED` and `set_seed()`, which seed Python's `random`,
NumPy (if installed) and PyTorch/CUDA (if installed), and force cuDNN toward
deterministic kernels. Every training entry point will call it.
