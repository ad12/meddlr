.. _models:

Models
==========


MONAI Interoperability
^^^^^^^^^^^^^^^^^^^^^^
Meddlr's configuration files can also be used to build models from `monai <https://docs.monai.io/en/stable/>``.

First, find ``model_name``, the class name of the model you want to build.
The casing of the string must match the casing of the class name in MONAI.

Then identify the arguments that the model requires for intantiation.
To find the argument names, look at the ``__init__`` method of the model class.

.. code-block:: yaml

    MODEL:
        name: monai/<model_name>
        MONAI:
            <model_name>:
                <arg1_name>: <value1>
                <arg2_name>: <value2>
                ...


An example configuration for building a MONAI ``VNet`` model could look like:

.. code-block:: yaml

    MODEL:
        name: monai/VNet
        MONAI:
            VNet:
                in_channels: 1
                out_channels: 1
                spatial_dims: 2
                dropout_dim: 2
                dropout_prob: 0.2


Building the models from the config is identical to how we do it for built-in models:

.. code-block:: python

    from meddlr.config import get_cfg, CfgNode

    cfg = get_cfg()

    # If the config is loaded from a file
    cfg.merge_from_file("config.yaml")

    # If the config is built in python.
    cfg = get_cfg()
    cfg.MODEL.MONAI.VNet = CfgNode()
    cfg.MODEL.MONAI.VNet.in_channels = 1
    cfg.MODEL.MONAI.VNet.out_channels = 1
    cfg.MODEL.MONAI.VNet.spatial_dims = 2
    cfg.MODEL.MONAI.VNet.dropout_dim = 2
    cfg.MODEL.MONAI.VNet.dropout_prob = 0.2

    # The torch.nn.Module.
    model = build_model(cfg)