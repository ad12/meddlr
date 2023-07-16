.. _path:

Managing Paths
===============

Hard-coding relative, absolute, and remote paths can be quite annoying, especially
when coordinating experiments with collaborators or running experiments across multiple
clusters.

Meddlr provides an ecosystem for automatically resolving absolute paths given user
preferences. This ecosystem allows you to coordinate directories for results, data,
and caches, as well as any custom paths you would like to configure.


Meddlr Paths: Results, Data, and Caches
----------------------------------------
There are three primary kinds of paths that Meddlr uses:

1. dataset paths: the path to the directory holding the datasets
2. result paths: the path to the directory where results should be stored
3. cache paths: the path where cachable data is stored

Setting Paths with Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One way of setting these paths is with environment variables:

.. code-block:: python

    import os
    os.environ["MEDDLR_DATASETS_DIR"] = "/path/to/datasets"
    os.environ["MEDDLR_RESULTS_DIR"] = "/path/to/results"
    os.environ["MEDDLR_CACHE_DIR"] = "/path/to/cache"

    from meddlr.utils import env
    env.get_path_manager().get_local_path("results://exp1")  # returns "/path/to/results/exp1"

If these parameters are set in the terminal, they will also be read:

.. code-block:: bash

    export MEDDLR_DATASETS_DIR="/path/to/datasets"
    export MEDDLR_RESULTS_DIR="/path/to/results"
    export MEDDLR_CACHE_DIR="/path/to/cache"

The recommended way of doing this is setting the environment variable in the `.bashrc`
of your user profile.


Setting Paths with `Cluster`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Setting environment variables can be quite tedious when working with multiple machines.
Meddlr also provides a way to automatically load paths for clusters of machines/nodes:

.. code-block:: python

    cluster = Cluster(
        'MyCluster', patterns=['nodeA', 'nodeB'],
        data_dir="/path/to/datasets", results_dir="/path/to/results",
        cache_dir="/path/to/cache"
    )

    # Saves the cluster to a file, so it will be automatically loaded the next time
    # you use meddlr.
    cluster.save()

    # To get the current working cluster.
    print(Cluster.working_cluster())

See :class:`meddlr.utils.Cluster` for more information.


Managing Remote Paths (BETA)
----------------------------
Meddlr has support (in BETA) for fetching files from remote paths
(e.g. Amazon S3, Google Drive, generic URLs via `wget`).

.. code-block:: python

    from meddlr.utils import env
    s3_file = env.get_path_manager().get_local_path("s3://bucket/path/to/file")
    url_file = env.get_path_manager().get_local_path("http://example.com/path/to/file")

We do not recommend using Google Drive to store files as it throttles the download
and will not scale to multiple users.