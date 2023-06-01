# meddlr
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ad12/meddlr/CI)
![GitHub](https://img.shields.io/github/license/ad12/meddlr)
[![Documentation Status](https://readthedocs.org/projects/meddlr/badge/?version=latest)](https://meddlr.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/ad12/meddlr/branch/main/graph/badge.svg?token=U6H83UCGFU)](https://codecov.io/gh/ad12/meddlr)

[Getting Started](GETTING_STARTED.md)

Meddlr is a config-driven ML framework built to simplify medical image reconstruction and analysis problems.

## ⚡ QuickStart
```bash
# Install Meddlr with basic dependencies
pip install meddlr

# Install Meddlr with all dependencies (e.g. pretrained models, benchmarking)
pip install 'meddlr[all]'
```

> _Installing locally_: For local development, fork and clone the repo and run `pip install -e ".[alldev]"`

> _Installing from main_: For most up-to-date code without a local install, run `pip install "meddlr @ git+https://github.com/ad12/meddlr@main"`

Configure your paths and get going!
```python
import meddlr as mr
import os

# (Optional) Configure and save machine/cluster preferences.
# This only has to be done once and will persist across sessions.
cluster = mr.Cluster()
cluster.set(results_dir="/path/to/save/results", data_dir="/path/to/datasets")
cluster.save()
# OR set these as environment variables.
os.environ["MEDDLR_RESULTS_DIR"] = "/path/to/save/results"
os.environ["MEDDLR_DATASETS_DIR"] = "/path/to/datasets"
```

Detailed instructions are available in [Getting Started](GETTING_STARTED.md).

## Visualizations
Use [MeddlrViz](https://github.com/ad12/meddlr-viz) to visualize your medical imaging datasets, ML models, and more!

```bash
pip install meddlr-viz
```

<div align="center">

<image src="https://github.com/ad12/meddlr-viz/blob/main/static/brats.gif" height=400 alt="A gallery of images from the BRATS dataset" />
</div>

## 🐘 Model Zoo
Easily serve and download pretrained models from the model zoo. A (evolving) list of pre-trained models can be found [here](MODEL_ZOO.md), on [HuggingFace 🤗](https://huggingface.co/arjundd), and in [project folders](projects).

To use them, pass the URLs for the config and weights (model) files to `mr.get_model_from_zoo`:

```python
import meddlr as mr

model = mr.get_model_from_zoo(
  cfg_or_file="https://huggingface.co/arjundd/vortex-release/resolve/main/mridata_knee_3dfse/Supervised/config.yaml",
  weights_path="https://huggingface.co/arjundd/vortex-release/resolve/main/mridata_knee_3dfse/Supervised/model.ckpt",
)
```

## 📓 Projects
Check out some [projects](projects) built with meddlr!

## ✏️ Contributing
Want to  add new features, fix a bug, or add your project? We'd love to include them! See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Acknowledgements
Meddlr's design for rapid experimentation and benchmarking is inspired by [detectron2](https://github.com/facebookresearch/detectron2).

## About
If you use Meddlr for your work, please consider citing the following work:

```
@article{desai2021noise2recon,
  title={Noise2Recon: A Semi-Supervised Framework for Joint MRI Reconstruction and Denoising},
  author={Desai, Arjun D and Ozturkler, Batu M and Sandino, Christopher M and Vasanawala, Shreyas and Hargreaves, Brian A and Re, Christopher M and Pauly, John M and Chaudhari, Akshay S},
  journal={arXiv preprint arXiv:2110.00075},
  year={2021}
}
```
