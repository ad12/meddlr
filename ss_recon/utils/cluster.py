"""Detect clusters and machines.

DO NOT MOVE THIS FILE.
"""
import getpass
import os
import re
import socket
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from fvcore.common.file_io import PathHandler, PathManager

# Path to the repository directory.
# TODO: make this cleaner
_REPO_DIR = os.path.join(os.path.dirname(__file__), "../..")


class Cluster(Enum):
    """Hacky way to keep track of the cluster you are working on.

    To identify the cluster, we inspect the hostname.
    This can be problematic if two clusters have the same hostname, though
    this has not been an issue as of yet.

    DO NOT use the machine's public ip address to identify it. While this is
    definitely more robust, there are security issues associated with this.

    Useful for developing with multiple people working on same and different
    machines.
    """

    UNKNOWN = 0, []
    ROMA = 1, ["roma"]
    VIGATA = 2, ["vigata"]
    NERO = 3, ["slurm-gpu-compute.*"]
    SHERLOCK = 4, ["sh[0-9]+.*"]
    SAIL = 5, ["sc.*stanford.edu", "pasteur[0-9].stanford.edu"]
    HARBIN = 6, ["harbin"]
    MRLEARNING = 7, ["mrlearning"]
    AUTOFOCUS = 8, ["autofocus"]
    SIENA = 9, ["siena"]
    TORINO = 10, ["torino"]
    CINE = 11, ["cine"]

    def __new__(cls, value: int, patterns: List[str]):
        """
        Args:
            value (int): Unique integer value.
            patterns (`List[str]`): List of regex patterns that would match the
                hostname on the compute cluster. There can be multiple hostnames
                per compute cluster because of the different nodes.
        """
        obj = object.__new__(cls)
        obj._value_ = value

        obj.patterns = patterns
        obj.dir_map = {}

        return obj

    @classmethod
    def cluster(cls):
        hostname = socket.gethostname()

        for clus in cls:
            for p in clus.patterns:
                if re.match(p, hostname):
                    return clus

        return cls.UNKNOWN

    def register_user(self, user_id: str, data_dir: str = "", results_dir: str = ""):
        """Register user preferences for paths.

        Args:
            user_id (str): User id found on the machine.
            data_dir (str): Default data directory.
                Paths starting with "data://" will be formated to this
                directory as the root. For example if `data_dir=/my/path`,
                then file path "data://data1" will be "/my/path/data1".
            results_dir (str): Default results directory.
                Performance is like that of data_dir expect with "results://"
                prefix.
        """
        if not data_dir:
            data_dir = os.path.abspath(os.path.join(_REPO_DIR, "datasets"))
        if not results_dir:
            results_dir = os.path.abspath(os.path.join(_REPO_DIR, "results"))

        self.dir_map[user_id] = {
            "data_dir": data_dir,
            "results_dir": results_dir,
        }

    def get_path(self, key):
        user_id = getpass.getuser()
        if user_id not in self.dir_map:
            raise ValueError("User {} is not registered on cluster {}".format(user_id, self.name))
        return self.dir_map[user_id][key]


# Environment variable for the current cluster that is being used.
CLUSTER = Cluster.cluster()


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(self._root_dir(), name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

    def _mkdirs(self, path: str, **kwargs):
        os.makedirs(self._get_local_path(path), exist_ok=True)

    @abstractmethod
    def _root_dir(self) -> str:
        pass


class DataHandler(GeneralPathHandler):
    PREFIX = "data://"

    def _root_dir(self):
        return CLUSTER.get_path("data_dir")


class ResultsHandler(GeneralPathHandler):
    PREFIX = "results://"

    def _root_dir(self):
        return CLUSTER.get_path("results_dir")


class AnnotationsHandler(GeneralPathHandler):
    PREFIX = "ann://"

    def _root_dir(self):
        return os.path.abspath(os.path.join(_REPO_DIR, "annotations"))


PathManager.register_handler(DataHandler())
PathManager.register_handler(ResultsHandler())
PathManager.register_handler(AnnotationsHandler())

# Paths are in order data, results
_USER_PATHS = {
    "arjundd": {
        CLUSTER.ROMA: (
            "/dataNAS/people/arjun/data",
            "/bmrNAS/people/arjun/results/ss_recon",
        ),
        CLUSTER.VIGATA: (
            "/dataNAS/people/arjun/data",
            "/bmrNAS/people/arjun/results/ss_recon",
        ),
        CLUSTER.NERO: (
            "/share/pi/bah/data",
            "/share/pi/bah/arjundd/results/ss_recon",
        ),
        CLUSTER.SIENA: (
            # "/data/datasets",  # mounted on siena only
            "/dataNAS/people/arjun/data",
            "/bmrNAS/people/arjun/results/ss_recon",
        ),
        CLUSTER.TORINO: (
            "/dataNAS/people/arjun/data",
            "/bmrNAS/people/arjun/results/ss_recon",
        ),
    },
    "ozt": {
        CLUSTER.HARBIN: (
            "/mnt/dense/ozt/dl-ss-recon/data",
            "/home/ozt/dl-ss-recon/results/ss_recon",
        ),
        CLUSTER.MRLEARNING: (
            "/mnt/dense/ozt/dl-ss-recon/data",
            "/home/ozt/dl-ss-recon/results/ss_recon",
        ),
        CLUSTER.AUTOFOCUS: (
            "/mnt/dense/ozt/dl-ss-recon/data",
            "/home/ozt/dl-ss-recon/results/ss_recon",
        ),
        CLUSTER.CINE: (
            "/mnt/dense/ozt/dl-ss-recon/data",
            "/home/ozt/dl-ss-recon/results/ss_recon",
        ),
    },
    "bgunel": {
        CLUSTER.ROMA: (
            "/dataNAS/people/arjun/data",
            "/dataNAS/people/bgunel/results/mrs",
        ),
        CLUSTER.VIGATA: (
            "/dataNAS/people/arjun/data",
            "/dataNAS/people/bgunel/results/mrs",
        ),
        CLUSTER.SIENA: (
            "/dataNAS/people/arjun/data",
            "/dataNAS/people/bgunel/results/mrs",
        ),
        CLUSTER.TORINO: (
            "/dataNAS/people/arjun/data",
            "/dataNAS/people/bgunel/results/mrs",
        ),
    },
    # New users add path preference below.
}


# Register default user paths.
_USER = getpass.getuser()
if _USER in _USER_PATHS:
    for cluster, (data_dir, results_dir) in _USER_PATHS[_USER].items():
        cluster.register_user(_USER, data_dir, results_dir)
