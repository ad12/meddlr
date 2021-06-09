# dl-ss-recon
Deep-learning based semi-supervised MRI reconstruction


### Setup

#### Environment
To avoid cuda-related issues, downloading `torch`, `torchvision`, and `cupy`
must be done prior to downloading other requirements.

```bash
# Create and activate the environment.
conda create -n dl_ss_env python=3.7
conda activate dl_ss_env

# Install cuda-dependant libraries. Change cuda version as needed.
# Below we show examples for cuda-10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install cupy-cuda101

# Install as package in virtual environment (recommended):
git clone https://github.com/ad12/dl-ss-recon.git
cd dl-ss-recon && python -m pip install -e .

# For all contributors, install formatting libs
make dev
```

#### Registering New Users
To register users to existing machines, add your username and machines to support
with that username to the `_USER_PATHS` dictionary in
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py).

#### Registering New Machines/Clusters
To register new machines, you will have to find the regex pattern(s) that can be used to
identify the machine or set of machines you want to add functionality for. See
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py) for more details.

#### Weights and Biases
Weights and Biases (W&B) is a convenient online experiment visualizer (like Tensorboard) that is currently free for academics. It's useful for sharing training runs, creating reports, and making other data-driven decisions.

Use `pip install wandb` to install W&B library

W&B account setup:
1. Create an account at [wandb.com](wandb.com)
2. Use chat to upgrade to the academic license
3. Send email to `arjundd at stanford dot edu` to get access

Adding API Key:
1. Get your user API key ([instructions](https://docs.wandb.com/library/api))
2. Add `export WANDB_API_KEY=<API KEY HERE>` to `.bashrc` (linux) or `.bash_profile` (OsX)

### Usage
To train a basic configuration from the repository folder in the command line, run
```bash
python tools/train_net.py --config-file configs/tests/basic.yaml

# Run in debug mode.
python tools/train_net.py --config-file configs/tests/basic.yaml --debug

# Run in reproducibility mode.
# This tries to make the run as reproducible as possible
# (e.g. setting seeds, deterministism, etc.).
python tools/train_net.py --config-file configs/tests/basic.yaml --reproducible
# or SSRECON_REPRO=True python tools/train_net.py --config-file configs/tests/basic.yaml
```

To evaluate the results, use `tools/eval_net.py`.
```bash
# Will automatically find best weights based on loss
python tools/eval_net.py --config-file configs/tests/basic.yaml

# Automatically find best weights based on psnr.
# options include psnr, l1, l2, ssim
python tools/eval_net.py --config-file configs/tests/basic.yaml --metric psnr

# Choose specific weights to run evaluation.
python tools/eval_net.py --config-file configs/tests/basic.yaml MODEL.WEIGHTS path/to/weights
```

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
