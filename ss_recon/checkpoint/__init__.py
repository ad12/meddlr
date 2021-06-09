# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File:


from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from .detection_checkpoint import DetectionCheckpointer

__all__ = ["Checkpointer", "PeriodicCheckpointer", "DetectionCheckpointer"]
