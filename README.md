# dl-ss-recon
Deep-learning based semi-supervised MRI reconstruction


### Setup

##### Environment
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
git clone https://github.com/ad12/MedSegPy.git
cd dl-ss-recon && python -m pip install -e .
```

##### Registering New Users
To register users to existing machines, add your username and machines to support
with that username to the `_USER_PATHS` dictionary in
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py).

##### Registering New Machines/Clusters
To register new machines, you will have to find the regex pattern(s) that can be used to
identify the machine or set of machines you want to add functionality for. See
[ss_recon/utils/cluster.py](ss_recon/utils/cluster.py) for more details.

### Usage
To train a basic configuration from the repository folder in the command line, run
```bash
python tools/train_net.py --config-file configs/tests/basic.yaml
```
