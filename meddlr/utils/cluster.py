"""Detect clusters and machines.

DO NOT MOVE THIS FILE.
"""
import logging
import os
import re
import shutil
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union

import yaml
from iopath.common.file_io import PathHandler

import meddlr as mr
from meddlr.utils import env

# Path to the repository directory.
# TODO: make this cleaner
_REPO_DIR = os.path.join(os.path.dirname(__file__), "../..")
_PATH_MANAGER = env.get_path_manager()
_LOGGER = logging.getLogger(__name__)

__all__ = ["Cluster", "set_cluster"]


class Cluster:
    """Manages configurations for different nodes/clusters.

    This class is helpful for managing different cluster configurations
    (e.g. storage paths) across different nodes/clusters without the
    overhead of duplicating the codebase across multiple nodes.

    A cluster is defined as a combination of a set of nodes (i.e. machines)
    that share configuration properties, such as storage paths. This class helps
    manage the configuration properties of these sets of nodes together. For example,
    let's say machines with hostnames *nodeA* and *nodeB* share some data and results
    paths. We can define a new cluster *MyCluster* to manage these:

    >>> cluster = Cluster(
    ...     'MyCluster', patterns=['nodeA', 'nodeB'],
    ...     data_dir="/path/to/datasets", results_dir="/path/to/results"
    ... )

    To use the configurations of a particular cluster, set the working cluster:

    >>> cluster.use()

    Configs can be persisted by saving to a file. If the config has been saved,
    future sessions will attempt to auto-detect the set of saved configs:

    >>> cluster.save()  # save cluster configuration to file
    >>> cluster.delete()  # deletes cluster configuration from file

    To get the file where the configs are stored, run:

    >>> Cluster.config_file()  # get the config file for the cluster

    To identify the cluster config to use, we inspect the hostname of the current node.
    This can be problematic if two machines have the same hostname, though
    this has not been an issue as of yet.

    Note:
        DO NOT use the machine's public ip address to identify it. While this is
        definitely more robust, there are security issues associated with this.

    Note:
        This class is not thread safe. Saving/deleting configs should be done on
        the main thread.
    """

    def __init__(
        self,
        name: str = None,
        patterns: Union[str, Sequence[str]] = None,
        data_dir: str = None,
        results_dir: str = None,
        cache_dir: str = None,
        **cfg_kwargs,
    ):
        """
        Args:
            name (str): The name of the cluster. Name is case-sensitive.
            patterns (Sequence[str]): Regex pattern(s) for identifying nodes
                in the cluster. Cluster will be identified by
                ``any(re.match(p, socket.gethostname()) for p in patterns)``.
                If ``None``, defaults to current hostname.
            data_dir (str, optional): The data directory. Defaults to
                ``os.environ['MEDDLR_DATASETS_DIR']`` or ``"./datasets"``.
            results_dir (str, optional): The results directory. Defaults to
                `"os.environ['MEDDLR_RESULTS_DIR']"` or ``"./results"``.
            cfg_kwargs (optional): Any other configurations you would like to
                store for the cluster.
        """
        if name is None:
            name = socket.gethostname()
        self.name = name

        if patterns is None:
            patterns = socket.gethostname()
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = patterns

        self._data_dir = data_dir
        self._results_dir = results_dir
        self._cache_dir = cache_dir
        self._cfg_kwargs = cfg_kwargs

    @property
    def data_dir(self):
        path = self._data_dir
        path = os.environ.get("MEDDLR_DATASETS_DIR", path if path else "./datasets")
        return _PATH_MANAGER.get_local_path(path)

    @property
    def datasets_dir(self):
        """Alias for ``self.data_dir``."""
        return self.data_dir

    @property
    def results_dir(self):
        path = self._results_dir
        path = os.environ.get("MEDDLR_RESULTS_DIR", path if path else "./results")
        return _PATH_MANAGER.get_local_path(path)

    @property
    def cache_dir(self):
        path = self._cache_dir
        path = os.environ.get("MEDDLR_CACHE_DIR", path if path else "~/cache/meddlr")
        return _PATH_MANAGER.get_local_path(path)

    def set(self, **kwargs):
        """Set cluster configuration properties.

        Args:
            kwargs: Keyword arguments to set.

        Examples:
            >>> cluster.set(data_dir="/path/to/datasets", results_dir="/path/to/results")

        Note:
            Setting attributes for the cluster will not override environment variables.
            For example, if ``os.environ['MEDDLR_DATASETS_DIR']`` is set, setting ``data_dir``
            will have no effect.
        """
        for k, v in kwargs.items():
            private_key = f"_{k}"
            if hasattr(self, private_key):
                setattr(self, private_key, v)
            else:
                self._cfg_kwargs[k] = v

    def __getattr__(self, attr: str):
        attr_env = f"MEDDLR_{attr.upper()}"
        try:
            value = os.environ.get(attr_env, self._cfg_kwargs[attr])
        except KeyError:
            raise AttributeError(f"Attribute {attr} not specified for cluster {self.name}.")
        return value

    def save(self):
        """Save cluster config to yaml file."""
        filepath = self.config_file()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cluster_cfgs = {}
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                cluster_cfgs = yaml.safe_load(f)

        data = {(k[1:] if k.startswith("_") else k): v for k, v in self.__dict__.items()}
        cluster_cfgs[self.name] = data

        with open(filepath, "w") as f:
            yaml.safe_dump(cluster_cfgs, f)

    def delete(self):
        """Deletes cluster config from yaml file."""
        filepath = self.config_file()
        if not os.path.isfile(filepath):
            return

        with open(filepath, "r") as f:
            cluster_cfgs: Dict[str, Any] = yaml.safe_load(f)

        if self.name not in cluster_cfgs:
            return

        cluster_cfgs.pop(self.name)
        with open(filepath, "w") as f:
            yaml.safe_dump(cluster_cfgs, f)

    def get_path(self, key):
        """Legacy method for fetching cluster-specific paths."""
        return getattr(self, key)

    @classmethod
    def all_clusters(cls) -> List["Cluster"]:
        return cls.from_config(name=None)

    @classmethod
    def cluster(cls):
        """Searches saved clusters by regex matching with hostname.

        Returns:
            Cluster: The current cluster.

        Note:
            The cluster must have been saved to a config file. Also, if
            there are multiple cluster matches, only the first (sorted alphabetically)
            will be returned.
        """
        try:
            clusters = cls.all_clusters()
        except FileNotFoundError:
            return _UNKNOWN
        hostname = socket.gethostname()
        for clus in clusters:
            if any(re.match(p, hostname) for p in clus.patterns):
                return clus
        return _UNKNOWN

    @classmethod
    def from_config(cls, name):
        """Loads cluster from config.

        Args:
            name (str | Sequence[str] | None): Cluster name(s) to load.
                If ``None``, all clusters will be loaded.

        Returns:
            Cluster: The Cluster object(s).
        """
        filepath = cls.config_file()
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)

        if name is None:
            return [cls(**cluster_cfg) for cluster_cfg in cfg.values()]
        elif isinstance(name, str):
            return cls(**cfg[name])
        else:
            return type(name)([cls(**cfg[n]) for n in name])

    @staticmethod
    def config_file():
        return os.path.join(env.settings_dir(), "clusters.yaml")

    @staticmethod
    def working_cluster() -> "Cluster":
        return _CLUSTER

    def use(self):
        """Sets ``self`` to be the working cluster of the project.

        The working cluster is the default cluster that is used to manage
        paths and other configuration variables.

        Examples:
            >>> cluster.use()

        Note:
            This function does not override environment variables.
            All environment variables will take priority over this clusters
        """
        set_cluster(self)

    def __repr__(self):
        return "Cluster({})".format(
            ", ".join("{}={}".format(k, v) for k, v in self.__dict__.items())
        )


def set_cluster(cluster: Union[str, Cluster] = None):
    """Sets the working cluster.
    Args:
        cluster (`str` or `Cluster`): The cluster name or cluster.
            If ``None``, will reset cluster to _UNKNOWN, meaning default
            data and results dirs will be used.
    """
    if cluster is None:
        cluster = _UNKNOWN
    elif isinstance(cluster, str):
        if cluster.lower() == _UNKNOWN.name.lower():
            cluster = _UNKNOWN
        else:
            cluster = Cluster.from_config(cluster)
    global _CLUSTER
    _CLUSTER = cluster


_UNKNOWN = Cluster("UNKNOWN", [])  # Unknown cluster
_CLUSTER = Cluster.cluster()  # Working cluster


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(self._root_dir(), name)

    def _open(self, path, mode="r", **kwargs):
        return _PATH_MANAGER.open(self._get_local_path(path), mode, **kwargs)

    def _mkdirs(self, path: str, **kwargs):
        os.makedirs(self._get_local_path(path), exist_ok=True)

    @abstractmethod
    def _root_dir(self) -> str:
        pass


class DataHandler(GeneralPathHandler):
    PREFIX = "data://"

    def _root_dir(self):
        return _CLUSTER.get_path("data_dir")


class ResultsHandler(GeneralPathHandler):
    PREFIX = "results://"

    def _root_dir(self):
        return _CLUSTER.get_path("results_dir")


class CacheHandler(GeneralPathHandler):
    PREFIX = "cache://"

    def _root_dir(self):
        return _CLUSTER.get_path("cache_dir")


class AnnotationsHandler(GeneralPathHandler):
    PREFIX = "ann://"

    def _get_cache_path(self):
        return os.path.join(
            env.get_path_manager().get_local_path(_CLUSTER.cache_dir),
            f"annotations/v{mr.__version__}",
        )

    def _root_dir(self):
        local_path = os.path.abspath(os.path.join(_REPO_DIR, "annotations"))
        if os.path.isdir(local_path):
            return local_path

        cache_path = self._get_cache_path()
        if not os.path.isdir(cache_path):
            self._download()
        return cache_path

    def _download(self):
        """Downloads annotations from meddlr github.

        Note:
            This method downloads the full repository of the current
            meddlr version and then copies the annotation files to
            the respective directory. This may not be very efficient,
            but it ensures that the annotation download process is
            less onerous on the user.
        """
        _LOGGER.info("Downloading annotations...")
        repo_path = download_repository()
        cache_path = env.get_path_manager().get_local_path(self._get_cache_path())
        os.makedirs(cache_path, exist_ok=True)
        retval = os.system(f"cp -r {repo_path}/annotations {cache_path}/")
        if retval != 0:
            raise RuntimeError("Could not download annotations.")


_PATH_MANAGER.register_handler(DataHandler())
_PATH_MANAGER.register_handler(ResultsHandler())
_PATH_MANAGER.register_handler(CacheHandler())
_PATH_MANAGER.register_handler(AnnotationsHandler())


def download_repository(version=None, path="cache://github-repo/{version}", force=False) -> str:
    """Downloads the repository from GitHub.

    Args:
        version (str): The version to download. Defaults to the
            version of the current codebase.
        path (str): The path to download the repository to.
            Defaults to the the cache directory.
        force (bool): Whether to overwrite existing files.

    Returns:
        str: The path to the downloaded repository.
    """
    if version is None:
        version = f"v{mr.__version__}"
    path = str(path)
    path = path.format(version=version)
    path = env.get_path_manager().get_local_path(path)

    dir_exists_and_not_empty = os.path.isdir(path) and os.listdir(path)
    if dir_exists_and_not_empty:
        if not force:
            return path
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    retval = os.system(f"wget -O - {env.get_github_url()}/tarball/{version} | tar xz -C {path}")
    if retval != 0:
        raise RuntimeError("Could not download repository.")

    folders = os.listdir(path)
    assert len(folders) == 1
    curr_path = os.path.join(path, folders[0])
    retval = os.system(f"mv {curr_path}/* {path}")
    if retval != 0:
        raise RuntimeError("Could not download repository.")
    shutil.rmtree(curr_path)

    return path
