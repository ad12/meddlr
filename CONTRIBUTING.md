### Contributing
Please run `./dev/linter.sh` and `pytest tests/` from the base repository directory before committing any code.

You may need to install the following libraries:
```bash
pip install black==19.3b0 isort flake8 flake8-comprehensions
```

##### Writing Unit Tests
Please write unit tests for any new functionality that is implemented and verify that
it does not interfere with existing functionality. 
Even if you're sure that it works, write a test!

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

##### Managing file paths in configs
Please do not push configs that have absolute paths.
All config files in this repo should be usable
by all users, regardless of the machine.

Instead, use prefixes like `"data://"` for data paths, "`results://`" for results paths, etc.

