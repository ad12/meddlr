# dl-ss-recon
Deep-learning based semi-supervised MRI reconstruction


### Setup

##### Registering New Users
To register users to existing machines, add your username and machines to support
with that username to the `_USER_PATHS` dictionary in 
`ss_recon/utils/cluster.py`[ss_recon/utils/cluster.py].

##### Registering New Machines/Clusters
To register new machines, you will have to find the regex pattern(s) that can be used to
identify the machine or set of machines you want to add functionality for. See 
`ss_recon/utils/cluster.py`[ss_recon/utils/cluster.py] for more details.


### Contributing
Please run `./dev/linter.sh` from the base repository directory before committing any code.

You may need to install the following libraries:
```bash
pip install black==19.3b0 isort flake8 flake8-comprehensions
```

##### Handling file paths
There are many file path manager libraries. For this project we use 
[fvcore](https://github.com/facebookresearch/fvcore).

For any opening files, writing to files, etc., do not use the `os` library as this
can cause some internal breakings. Instead use `fvcore.common.file_io.PathManager`.

```python
from fvcore.common.file_io import PathManager
path = "/my/path"

# get absolute path
PathManager.get_local_path(path)

# open file
with PathManager.open(path, "r") as f:
    ...

```