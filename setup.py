import os
import subprocess
import sys
from os import path
from shutil import rmtree

from setuptools import Command, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
_INIT_FILE = path.join(path.abspath(path.dirname(__file__)), "meddlr", "__init__.py")


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


class BumpVersionCommand(Command):
    """
    To use: python setup.py bumpversion -v <version>

    This command will push the new version directly and tag it.
    """

    description = "Installs the foo."
    user_options = [
        ("version=", "v", "the new version number"),
    ]

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        self.version = None

    def finalize_options(self):
        # This package cannot be imported at top level because it
        # is not recognized by Github Actions.
        from packaging import version

        if self.version is None:
            raise ValueError("Please specify a version number.")

        current_version = get_version()
        if not version.Version(self.version) > version.Version(current_version):
            raise ValueError(
                f"New version ({self.version}) must be greater than "
                f"current version ({current_version})."
            )

    def _undo(self):
        os.system("git restore --staged meddlr/__init__.py")
        os.system("git checkout -- meddlr/__init__.py")

    def run(self):
        self.status("Checking current branch is 'main'")
        current_branch = get_git_branch()
        if current_branch != "main":
            raise RuntimeError(
                "You can only bump the version from the 'main' branch. "
                "You are currently on the '{}' branch.".format(current_branch)
            )

        self.status("Pulling latest changes from origin")
        err_code = os.system("git pull")
        if err_code != 0:
            raise RuntimeError("Failed to pull from origin.")

        self.status("Checking working directory is clean")
        err_code = os.system("git diff --exit-code")
        err_code += os.system("git diff --cached --exit-code")
        if err_code != 0:
            raise RuntimeError("Working directory is not clean.")

        # TODO: Add check to see if all tests are passing on main.

        # Change the version in __init__.py
        self.status(f"Updating version {get_version()} -> {self.version}")
        update_version(self.version)
        if get_version() != self.version:
            self._undo()
            raise RuntimeError("Failed to update version.")

        self.status("Adding meddlr/__init__.py to git")
        err_code = os.system("git add meddlr/__init__.py")
        if err_code != 0:
            self._undo()
            raise RuntimeError("Failed to add file to git.")

        # Commit the file with a message '[bumpversion] v<version>'.
        self.status(f"Commit with message '[bumpversion] v{self.version}'")
        err_code = os.system("git commit -m '[bumpversion] v{}'".format(get_version()))
        if err_code != 0:
            self._undo()
            raise RuntimeError("Failed to commit file to git.")

        # Push the commit to origin.
        # self.status("Pushing commit to origin")
        # err_code = os.system("git push")
        # if err_code != 0:
        #     # TODO: undo the commit automatically.
        #     raise RuntimeError("Failed to push commit to origin.")

        sys.exit()


def get_version():
    init_py = open(_INIT_FILE, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


def update_version(version):
    init_py = [
        line if not line.startswith("__version__") else f'__version__ = "{version}"\n'
        for line in open(_INIT_FILE, "r").readlines()
    ]
    with open(_INIT_FILE, "w") as f:
        f.writelines(init_py)


def get_git_branch():
    """Return the name of the current branch."""
    proc = subprocess.Popen(["git branch"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    if err is not None:
        raise RuntimeError(f"Error finding git branch: {err}")
    out = out.decode("utf-8").split("\n")
    current_branch = [line for line in out if line.startswith("*")][0]
    current_branch = current_branch.replace("*", "").strip()
    return current_branch


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
    # Pin the numpy version because other libraries dont handle the np.float dtype yet.
    "numpy<=1.23.5",
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
    "torchmetrics>=0.5.1,<=0.11.4",
    "iopath",
    "packaging",
]

EXTRAS = {
    "dev": [
        # Formatting
        "coverage",
        "flake8",
        "isort",
        "black==22.12.0",
        "flake8-bugbear",
        "flake8-comprehensions",
        "pre-commit>=2.9.3",
        # Testing
        "pytest",
        "medpy",
        "wrapt",
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
    cmdclass={"upload": UploadCommand, "bumpversion": BumpVersionCommand},
)
