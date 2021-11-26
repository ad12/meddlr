
## Getting Started

### Installation
Note: To avoid cuda-related issues, downloading `torch`, `torchvision`, and `cupy` (optional)
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


### Configuring Paths (optional)
There are three primary kinds of paths that Meddlr uses: 

1. dataset paths: the path to the directory holding the datasets
2. result paths: the path to the directory where results should be stored
3. cache paths: the path where cachable data is stored
 
***Builtin Dataset Paths***

You can set the location for builtin datasets by
export MEDDLR_DATASETS=/path/to/datasets. If left unset, the default
is ./datasets relative to your current working directory.

***Result Paths***

Similarly, you can set the location for the results directory by
export MEDDLR_RESULTS=/path/to/results. If left unset, the default
is ./results relative to your current working directory.

As a shortcut, we designate the prefix `"results://"`
in any filepath to point to a result directory of your choosing.
For example, `"results://exp1"` will resolve to the path
`"<MEDDLR_RESULTS>/exp1"`.

An example of how to do this in python (i.e. without export statements) is shown below:

```python
import os
os.environ["MEDDLR_DATASETS"] = "/path/to/datasets"
os.environ["MEDDLR_RESULTS"] = "/path/to/results"

import medsegpy.utils  # import implicitly registers prefixes
from fvcore.common.file_io import PathManager
PathManager.get_local_path("results://exp1")  # returns "/path/to/results/exp1"
```

You can also define your own prefixes to resolve by adding your own path handler.
This is useful if you want to use the same script to run multiple projects. See fvcore's
[fileio](https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/file_io.py)
for more information.

### Usage
To train a basic configuration from the repository folder in the command line, run
```bash
python tools/train_net.py --config-file github://configs/tests/basic.yaml

# Run in debug mode.
python tools/train_net.py --config-file github://configs/tests/basic.yaml --debug

# Run in reproducibility mode.
# This tries to make the run as reproducible as possible
# (e.g. setting seeds, deterministism, etc.).
python tools/train_net.py --config-file github://configs/tests/basic.yaml --reproducible
# or SSRECON_REPRO=True python tools/train_net.py --config-file configs/tests/basic.yaml

# (ALPHA) Enable profiling RAM.
# This tracks the RAM usage using the guppy library.
# Install guppy with `pip install guppy3`
SSRECON_MPROFILE=True python tools/train_net.py --config-file github://configs/tests/basic.yaml
```

To evaluate the results, use `tools/eval_net.py`.
```bash
# Will automatically find best weights based on loss
python tools/eval_net.py --config-file github://configs/tests/basic.yaml

# Automatically find best weights based on psnr.
# options include psnr, l1, l2, ssim
python tools/eval_net.py --config-file github://configs/tests/basic.yaml --metric psnr

# Choose specific weights to run evaluation.
python tools/eval_net.py --config-file github://configs/tests/basic.yaml MODEL.WEIGHTS path/to/weights
```