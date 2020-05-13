import re
import socket
from enum import Enum
from typing import List


class Cluster(Enum):
    """Compute cluster.

    TODO (arjundd): make the paths configurable via experiments/preferences.
    """

    UNKNOWN = 0, [], None
    ROMA = 1, ["roma"], "/bmrNAS/people/arjun/results"
    VIGATA = 2, ["vigata"], "/bmrNAS/people/arjun/results"
    NERO = 3, ["slurm-gpu-compute.*"], "/share/pi/bah/arjundd/results"

    def __new__(cls, value: int, patterns: List[str], save_dir: str):
        """
        Args:
            value (int): Unique integer value.
            patterns (`List[str]`): List of regex patterns that would match the
                hostname on the compute cluster. There can be multiple hostnames
                per compute cluster because of the different nodes.
            save_dir (str): Directory to save data to.
        """
        obj = object.__new__(cls)
        obj._value_ = value

        obj.patterns = patterns
        obj.save_dir = save_dir

        return obj

    @classmethod
    def cluster(cls):
        hostname = socket.gethostname()

        for clus in cls:
            for p in clus.patterns:
                if re.match(p, hostname):
                    return clus

        return cls.UNKNOWN


# Environment variable for the current cluster that is being used.
CLUSTER = Cluster.cluster()
