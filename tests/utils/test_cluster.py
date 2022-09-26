import os
import socket
import unittest

from meddlr.utils.cluster import Cluster

from ..util import temp_env


class TestCluster(unittest.TestCase):
    @temp_env
    def test_basic(self):
        """Test basic cluster creation."""
        # Clear the environment to avoid reading from user's preset environment.
        os.environ.clear()

        cluster = Cluster()

        assert cluster.name == socket.gethostname()
        assert cluster.patterns == [socket.gethostname()]

        assert cluster._data_dir is None
        assert cluster._results_dir is None
        assert cluster._cache_dir is None
        assert cluster._cfg_kwargs == {}

        assert cluster.data_dir == "./datasets"
        assert cluster.datasets_dir == cluster.data_dir
        assert cluster.results_dir == "./results"
        assert cluster.cache_dir == os.path.expanduser("~/.cache/meddlr")

    @temp_env
    def test_set(self):
        """Test setting configuration properties."""
        # Clear the environment to avoid reading from user's preset environment.
        os.environ.clear()

        cluster = Cluster()

        cluster.set(data_dir="my-data", results_dir="my-results", cache_dir="my-cache")
        assert cluster.data_dir == "my-data"
        assert cluster.results_dir == "my-results"
        assert cluster.cache_dir == "my-cache"

        cluster.set(foo="bar")
        assert cluster.foo == "bar"

    @temp_env
    def test_environment_variables(self):
        """Test that environment variables are overrides for cluster configurations."""
        # Clear the environment to avoid reading from user's preset environment.
        os.environ.clear()

        cluster = Cluster()

        cluster.set(data_dir="my-data", results_dir="my-results", cache_dir="my-cache")
        assert cluster.data_dir == "my-data"
        assert cluster.results_dir == "my-results"
        assert cluster.cache_dir == "my-cache"

        os.environ["MEDDLR_DATASETS_DIR"] = "foo-bar"
        assert cluster.data_dir == "foo-bar"

    @temp_env
    def test_peristence(self):
        """Test that cluster configuration can be persisted (saved, loaded, deleted).

        Note:
            This test is not thread-safe. Please do not run multiple instances of
            this test simultaneously with multiprocessing.
        """
        # Clear the environment to avoid reading from user's preset environment.
        os.environ.clear()

        name = "test-foobar"
        cluster = Cluster(name=name)

        cluster.set(data_dir="my-data", results_dir="my-results", cache_dir="my-cache")

        cluster.save()
        cluster2 = cluster.from_config(name)
        assert cluster2 is not cluster
        assert cluster2.data_dir == "my-data"
        assert cluster2.results_dir == "my-results"
        assert cluster2.cache_dir == "my-cache"

        cluster.delete()
        with self.assertRaises(KeyError):
            # Once deleted the cluster should not be found.
            cluster2 = cluster.from_config(name)
