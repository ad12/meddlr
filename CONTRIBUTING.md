# Contributing
Thank you for considering to contribute to Meddlr! We welcome all contributions: code, documentation, feedback, and support.

If you use Meddlr in your work (research, company, etc.) and find it useful, spread the word!

This guide is inspired by [Huggingface transformers](https://github.com/huggingface/transformers).

## General Guidelines
1. Please do not commit broken code to **any** branch.
2. Only commit stable config files. Config files that are in development should not be committed.
3. Use pull requests (PR) to the merge in code. Do not develop on the `main` branch.

## How To Contribute

### Getting Started
1. For the [`repository`](https://github.com/ad12/meddlr) by clicking on the `Fork` button on the repository's page.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/meddlr.git
   $ cd meddlr
   $ git remote add upstream https://github.com/ad12/meddlr.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following commands in a virtual environment:

    ```bash
    pip install -e ".[dev]"
    make dev
    ```

5. Once you finish adding your code, run all major formatting and checks using the following:

```bash
make autoformat lint test
```

6. Commit your changes and submit a pull request. Follow the checklist below when creating a pull request.
### Checklist

1. Make the title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, mention the issue number in
  the pull request description
3. If your PR is a work in progress, start the title with `[WIP]`
4. Make sure existing tests pass;
5. Add high-coverage tests. Additions without tests will not be merged
6. All public methods must have informative docstrings in the google style.

### Tests

Please write unit tests for any new functionality that is implemented and verify that it does not interfere with existing functionality. 
Even if you're sure that it works, write a test.

Library tests can be found in the  [tests folder](tests/).

From the root of the repository, here's how to run tests with `pytest` for the library:

```bash
$ make test
```

### Style guide
`meddlr` follows the [google style](https://google.github.io/styleguide/pyguide.html) for documentation.


## Technical Guide to Contributing
This section goes over some specific technical guides when contributing new code.
### Handling file paths
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

