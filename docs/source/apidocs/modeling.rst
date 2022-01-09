.. _api_modeling:

meddlr.modeling
================
Meddlr ships with some popular models for image reconstruction and analysis tasks.
These models can be configured using meddlr configuration files, which allows for faster
iteration and hyperparameter management.

Meddlr also provides model blocks and layers, which are modularized components for different models.
This affords the flexibility of creating new model architectures without having to re-implement useful
blocks/layers.

.. Meddlr is also compatible with third-party model hubs like `MONAI <https://github.com/Project-MONAI/MONAI>`_, PyTorch Hub, and detectron2.

meddlr.modeling.meta_arch
==========================
Meddlr is composed of a series of `meta-architectures`, which can be leveraged in different applications.
While some of these architectures are unique to particular applications (e.g. unrolled networks for image reconstruction),
most networks are built for plug-and-play use for any application (e.g. U-Net). Some of these meta architectures
are wrappers around more basic architectures. For example, ``VortexModel`` and ``N2RModel`` wrap around other architectures.

.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.modeling.meta_arch.CSModel
    meddlr.modeling.meta_arch.DenoisingModel
    meddlr.modeling.meta_arch.GeneralizedUNet
    meddlr.modeling.meta_arch.GeneralizedUnrolledCNN
    meddlr.modeling.meta_arch.N2RModel
    meddlr.modeling.meta_arch.SSDUModel
    meddlr.modeling.meta_arch.VortexModel


meddlr.modeling.layers
==========================
Layers are the most elementary components of the model architecture.
Meddlr provides a unified interface for building both default PyTorch and custom layers
implemented in Meddlr.

Custom layers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.modeling.layers.ConvWS2d
    meddlr.modeling.layers.ConvWS3d
    meddlr.modeling.layers.GaussianBlur

Build utils
-----------
.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.modeling.layers.get_layer_type
    meddlr.modeling.layers.get_layer_kind


meddlr.modeling.blocks
==========================
Layers can be assembled together to make *blocks*. The default blocks
provided in meddlr can be constructed using sequences of strings and optional
keyword arguments.

.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.modeling.blocks.SimpleConvBlockNd
    meddlr.modeling.blocks.SimpleConvBlock2d
    meddlr.modeling.blocks.SimpleConvBlock3d
    meddlr.modeling.blocks.ResBlockNd
    meddlr.modeling.blocks.ResBlock2d
    meddlr.modeling.blocks.ResBlock3d
    meddlr.modeling.blocks.ConcatBlockNd
    meddlr.modeling.blocks.ConcatBlock2d
    meddlr.modeling.blocks.ConcatBlock3d
