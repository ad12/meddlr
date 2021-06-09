# Contributing
Please run `make autoformat lint test` from the base repository directory before committing any code.

You may need to run `make dev` to install the appropriate dependencies.

## General Guidelines
1. Do not commit broken code to **any** branch
2. Only commit stable config files. Config files that are in development should not be committed
3. Use pull requests (PR) to the merge in code. All PRs should be to the `dev` branch

## Writing Unit Tests
Please write unit tests for any new functionality that is implemented and verify that
it does not interfere with existing functionality. 
Even if you're sure that it works, write a test!

## Handling file paths
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

## Managing file paths in configs
Please do not push configs that have absolute paths.
All config files in this repo should be usable
by all users, regardless of the machine.

Instead, use prefixes like `"data://"` for data paths, "`results://`" for results paths, etc.

