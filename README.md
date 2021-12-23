# meddlr
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ad12/meddlr/CI)
![GitHub](https://img.shields.io/github/license/ad12/meddlr)
[![Documentation Status](https://readthedocs.org/projects/meddlr/badge/?version=latest)](https://meddlr.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<!-- [![codecov](https://codecov.io/gh/ad12/meddlr/branch/main/graph/badge.svg?token=U6H83UCGFU)](https://codecov.io/gh/ad12/meddlr) -->

[Getting Started](GETTING_STARTED.md)

Meddlr is a config-driven ML framework built to simplify medical image reconstruction and analysis problems.



## Installation
To avoid cuda-related issues, downloading `torch`, `torchvision`, and `cupy` (optional)
must be done prior to downloading other requirements.

```bash
# Create and activate the environment.
conda create -n meddlr_env python=3.7
conda activate meddlr_env

# Install cuda-dependant libraries. Change cuda version as needed.
# Below we show examples for cuda-10.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install cupy-cuda102

# Go to https://github.com/ad12/meddlr and fork the repository.

# Install as package in virtual environment (recommended):
git clone https://github.com/<your-github-username>/meddlr.git
cd meddlr && python -m pip install -e '.[dev]'

# For all contributors, install development packages.
make dev
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

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
