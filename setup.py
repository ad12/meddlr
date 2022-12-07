import os
import sys
from os import path
from shutil import rmtree

from setuptools import Command, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


class UploadCommand(Command):
    """Support setup.py upload.
    Adapted from https://github.com/robustness-gym/meerkat.
    """

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(get_version()))
        os.system("git push --tags")

        sys.exit()


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "meddlr", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]  # noqa: E741
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


# ---------------------------------------------------
# Setup Information
# ---------------------------------------------------
NAME = "meddlr"
DESCRIPTION = (
    "Meddlr is a config-driven framework built to simplify ML-based "
    "medical image reconstruction and analysis."
)
VERSION = get_version()
AUTHOR = "The Meddlr Team"
EMAIL = "arjundd@stanford.edu"
URL = "https://github.com/ad12/meddlr"
REQUIRES_PYTHON = ">=3.6"

REQUIRED = [
    "pyxb",  # need to install before ismrmrd
    "h5py",
    "matplotlib",
    "numpy",
    "tensorboard",
    "fvcore",
    "mridata",
    "scikit-image>=0.18.2",
    "sigpy>=0.1.17",
    "ismrmrd",
    "pandas",
    "silx",
    "tqdm",
    "omegaconf",
    "torchmetrics>=0.5.1",
    "iopath",
    "packaging",
]

EXTRAS = {
    "dev": [
        # Formatting
        "coverage",
        "flake8",
        "isort",
        "black==22.3.0",
        "flake8-bugbear",
        "flake8-comprehensions",
        "pre-commit>=2.9.3",
        # Testing
        "pytest",
        "medpy",
        "pooch",
        "gdown<4.6.0",
        "parameterized",
        # tifffile==2022.7.28 not reading scipy data.
        # TODO (arjundd): Investigate tifffile issue.
        "tifffile<=2022.5.4",
        # Documentation
        "sphinx",
        "sphinxcontrib-bibtex",
        "sphinx-rtd-theme",
        "m2r2",
    ],
    "benchmarking": ["medpy"],
    "deployment": ["gdown<4.6.0", "requests", "iocursor"],
    "docs": ["sphinx", "sphinxcontrib.bibtex", "sphinx-rtd-theme", "m2r2"],
    "metrics": ["lpips"],
    "modeling": ["monai"],
}

base_extras = [EXTRAS[k] for k in EXTRAS.keys() if k not in ["dev", "docs"]]
EXTRAS["all"] = list(set(sum(base_extras, [])))
EXTRAS["alldev"] = list(set(sum(EXTRAS.values(), [])))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("configs", "tests", "*.tests", "*.tests.*", "tests.*")),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
