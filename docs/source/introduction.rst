.. _introduction:

**This guide is still under construction**

Introduction
================================================================================
Meddlr is a Python framework built to simplify experimentation with medical
image reconstruction and analysis problems.

Meddlr is designed to facilitate the experimentation lifecycle for new ML-based
methods by unifying data formats and interfaces,
standardizing metric computation, and distributing state-of-the-art ML methods for these problems.
Meddlr is also config-driven, which can simplify launching, monitoring, and iterating on experiments.


Features
--------------------------------------------------------------------------------

Unified Data Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data are represented as dictionaries with keys for inputs, outputs, and metadata.
This unified representation simplifies synthesizing using different datasets during
training, validation, and evaluation.

Meddlr also distributes intelligent data samplers that simplify batching data based
on particular metadata. For example, if all images in the batch should be of the same
dimensions, the :class:`meddlr.data.samplers.GroupSampler` can be used to sample data
in such a manner.

Standardized Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Meddlr metrics extend :module:torchmetrics to provide a standardized interface for
computing metrics across image patches (e.g. per slice) or across the entire scan.
It also provides pandas and dictionary export functions to simplify metric I/O on
a per-example basis.

Built-in Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^
Meddlr offers default implementations for common baselines for MRI reconstruction
and image analysis (e.g. segmentation). 
