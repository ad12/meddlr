#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from os import path
from setuptools import find_packages, setup

import torch
import torchvision

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert [1, 5] <= torch_ver, "Requires torch >=1.5"
tv_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert [0, 6] <= tv_ver, "Requires torchvision >=0.6"


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "ss_recon", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        l.strip() for l in init_py if l.startswith("__version__")  # noqa: E741
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("SSRECON_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [
            l for l in init_py if not l.startswith("__version__")  # noqa: E741
        ]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="ss_recon",
    version=get_version(),
    author="Arjun Desai & Batu Ozturkler",
    url="",
    description="SSRecon is a framework for semi-supervised MR reconstruction",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "tensorboard",
        "fvcore",
        "mridata",
        "scikit-image",
        "sigpy",
        "ismrmrd",
        "pandas",
        "silx",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "flake8",
            "isort",
            "black==19.3b0",
            "flake8-bugbear",
            "flake8-comprehensions",
        ]
    },
)
