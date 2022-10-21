Config
=================

Meddlr offers a key-value based config system which enables instantiating and connecting objects in a standardized way.
This enables common behaviors during data loading, model training, and evaluation.

This config system is built with YAML and `yacs <https://github.com/rbgirshick/yacs>`.
Note that not all Python functionality can be emulated by the dictionary-like YAML
format. In these cases, use the Meddlr API to compose your logic.

This structure is inspired by the YAML-based configuration in
`detectron2 <https://detectron2.readthedocs.io/en/latest/tutorials/configs.html#yacs-configs>``


Basic Usage
-----------------
Configs are structured nested dictionaries that are represented by nested :class:`meddlr.config.CfgNode` in Meddlr.
These configs can be controlled in a number of ways:

.. code-block:: python

   from meddlr.config import get_cfg

   cfg = get_cfg()

   # Edit the config in code.
   cfg.SOLVER.LR = 0.01  # override the learning rate in the config
   cfg.SOLVER.TRAIN_BATCH_SIZE = 32  # override the training batch size

   # Read the config from a file.
   cfg.merge_from_file("my_config_file.yaml")

   # Edit the config from a flattened list of key value pairs.
   cfg.merge_from_list(["SOLVER.LR", 0.01, "SOLVER.TRAIN_BATCH_SIZE", 32])

   # Write the config to a file.
   with open("new_config_file.yaml", "w") as f:
      cfg.dump(f)


When initializing configs from YAML files, users can specify a special `_BASE_: base_config.yaml` field,
which will load a base config file first. Values in the base config will be overwritten by values specified
in the config. This can be useful for setting up common settings across multiple
experiments. A trove of base config files can be found
`on GitHub <https://github.com/ad12/meddlr/tree/main/configs>`.

You can find more details about the specific config options in
`meddlr/config/config.py`

Extending the Config System
--------------------------------------------
As developers, we may often want to extend the default config system to support new features.

Projects that use the Meddlr library should extend the default config in their own code.

.. code-block:: python

   from meddlr.config import get_cfg, set_cfg

   def add_new_fields(cfg):
      cfg.NEW_FIELD = 0
      cfg.NEW_FIELD_STR = "default"
   
   cfg = get_cfg()
   cfg = add_new_fields(cfg)

   # Set the default config to the updated config.
   set_cfg(cfg)

   # Now you can use the config.
   cfg.merge_from_file("my_project_experiment1.yaml")



Recommended Practices
----------------------
1. Keep config files lightweight. Refactor common config fields into a base config.
2. Save your config file in your experiments. This will be done for you if you use `meddlr.engine.default_setup`
3. Reuse config fields in your code when possible. Do not duplicate existing config fields.


Default Fields in Meddlr
--------------------------
.. toggle::

   .. csv-table:: Meddlr default config
      :file: ../../assets/temp/config-docs.csv
      :widths: 30, 70
      :header-rows: 1