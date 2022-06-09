import copy
import inspect
import logging
import types
from collections import defaultdict
from typing import List

from tabulate import tabulate

from meddlr.utils.logger import log_first_n

__all__ = ["DatasetCatalog", "MetadataCatalog"]


class DatasetCatalog(object):
    """A catalog that stores information about the datasets and how to fetch them.

    This catalog makes it easy to choose different datasets based on a config string.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "mridata_knee_2019_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Meddlr Dataset format
    (See datasets/README.md for details). For a list of built-in datasets,
    see `data/datasets/builtin.py`.

    This catalog also supports aliases for different datasets. This is useful for
    backwards compatibility and easy config access.

    Adapted from https://github.com/facebookresearch/detectron2.

    TODO (arjundd): Make this based off Registry.
    """

    _REGISTERED = {}
    _ALIASES = {}

    @staticmethod
    def register(name, func_or_name):
        """
        Args:
            name (str): The name that identifies a dataset,
                e.g. "mridata_knee_2019_train".
            func_or_name (callable | str): A callable that returns a list of dicts
                or the name of an existing dataset. If the latter, it will be registered
                as a soft alias to the underlying dataset.
        """
        if name in DatasetCatalog._REGISTERED:
            raise ValueError("Dataset '{}' is already registered!".format(name))
        if isinstance(func_or_name, str):
            assert (
                func_or_name in DatasetCatalog._REGISTERED
            ), f"Target dataset '{func_or_name}' is not registered"
            DatasetCatalog._ALIASES[name] = func_or_name
            return

        assert callable(
            func_or_name
        ), "You must register a function with `DatasetCatalog.register`!"
        DatasetCatalog._REGISTERED[name] = (func_or_name, inspect.signature(func_or_name))

    @staticmethod
    def get(name, *args, **kwargs):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset,
                e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.0
        """
        try:
            dname = DatasetCatalog._ALIASES.get(name, name)
            f, sig = DatasetCatalog._REGISTERED[dname]
        except KeyError:
            raise KeyError(
                "Dataset '{}' is not registered! "
                "Available datasets are: {}".format(
                    name, ", ".join(DatasetCatalog._REGISTERED.keys())
                )
            )

        logger = logging.getLogger(__name__)
        parameters = sig.parameters
        has_kwargs = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in parameters.values())
        if not has_kwargs:
            missing_kwargs = [k for k in kwargs if k not in parameters]
            valid_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
            if missing_kwargs:
                logger.warning(
                    "Function to load dataset {} does not support following kwargs. "
                    "Ignoring them...\n\t{}".format(name, missing_kwargs)
                )
            kwargs = valid_kwargs
        return f(*args, **kwargs)

    @staticmethod
    def list() -> List[str]:
        """List all registered datasets (including aliases).

        For a more decorated output, see :func:`DatasetCatalog.__repr__`.

        Returns:
            list[str]
        """
        return list(DatasetCatalog._REGISTERED.keys()) + list(DatasetCatalog._ALIASES.keys())

    @staticmethod
    def clear(name=None):
        """Remove the registered dataset and it's aliases.

        Args:
            name (str): The name of the dataset to be removed.
                If None, all datasets are removed.
        """
        if name is None:
            DatasetCatalog._REGISTERED.clear()
            DatasetCatalog._ALIASES.clear()
            return

        aliases = [name] + [k for k, v in DatasetCatalog._ALIASES.items() if v == name]
        DatasetCatalog._REGISTERED.pop(name, None)
        for alias in aliases:
            DatasetCatalog._ALIASES.pop(alias, None)

    def __repr__(self) -> str:
        base_to_aliases = defaultdict(list)
        for alias, base in self._ALIASES.items():
            base_to_aliases[base].append(alias)

        datasets = [
            {
                "name": name,
                "aliases": base_to_aliases.get(name, None),
                **(
                    MetadataCatalog.get(name).__dict__
                    if name in MetadataCatalog._NAME_TO_META
                    else {}
                ),
            }
            for name in self._REGISTERED
        ]
        table = tabulate(datasets, headers="keys", tablefmt="fancy_grid")
        return "Cataloged Datasets:\n{}".format(table)


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible
    globally.

    Examples:

    .. code-block:: python

        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors
    # will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        raise AttributeError(
            "Attribute '{}' does not exist in the metadata of '{}'. "
            "Available keys are {}.".format(key, self.name, str(self.__dict__.keys()))
        )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the
        Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        Otherwise return default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MetadataCatalog:
    """
    MetadataCatalog provides access to "Metadata" of a given dataset.

    The metadata associated with a certain name is a singleton: once created,
    the metadata will stay alive and will be returned by future calls to
    `get(name)`.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the
    execution of the program, e.g.: the class names in COCO.

    Adapted from https://github.com/facebookresearch/detectron2.
    """

    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
        assert len(name)
        if name in MetadataCatalog._NAME_TO_META:
            ret = MetadataCatalog._NAME_TO_META[name]
            return ret
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m
