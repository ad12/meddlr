# meddlr
[Getting Started](GETTING_STARTED.md)

Meddlr is a config-driven ML framework built to simplify medical image reconstruction and analysis problems.



#### Installation
To avoid cuda-related issues, downloading `torch`, `torchvision`, and `cupy` (optional)
must be done prior to downloading other requirements.

```bash
# Create and activate the environment.
conda create -n meddlr_env python=3.7
conda activate meddlr_env

# Install cuda-dependant libraries. Change cuda version as needed.
# Below we show examples for cuda-10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install cupy-cuda101

# Install as package in virtual environment (recommended):
git clone https://github.com/ad12/meddlr.git
cd dl-ss-recon && python -m pip install -e '.[dev]'

# For all contributors, install development packages.
make dev
```

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

### About
If you use Meddlr for your work, please consider citing the following work:

```
@article{desai2021noise2recon,
  title={Noise2Recon: A Semi-Supervised Framework for Joint MRI Reconstruction and Denoising},
  author={Desai, Arjun D and Ozturkler, Batu M and Sandino, Christopher M and Vasanawala, Shreyas and Hargreaves, Brian A and Re, Christopher M and Pauly, John M and Chaudhari, Akshay S},
  journal={arXiv preprint arXiv:2110.00075},
  year={2021}
}
```
