# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import pickle

from fvcore.common.checkpoint import Checkpointer, _strip_prefix_if_present
from fvcore.common.file_io import PathManager

logger = logging.getLogger(__name__)


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron
    & detectron2 model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=True, **checkpointables)

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        model_state_dict = checkpoint["model"]
        incompatible = super()._load_model(checkpoint)
        if not _has_incompatible_keys(incompatible):
            return

        # Load models that are wrapped in an nn.Module.
        # Following convention, these wrapped models begin with the "model." prefix.
        logger.warn("Attempting to load wrapped model from model state dict.")
        _strip_prefix_if_present(model_state_dict, "module.")
        _strip_prefix_if_present(model_state_dict, "model.")
        incompatible = super()._load_model({"model": model_state_dict})
        if _has_incompatible_keys(incompatible):
            raise ValueError("Found incompatible keys or shapes:\n{}".format(incompatible))


def _has_incompatible_keys(incompatible):
    return (
        bool(incompatible.missing_keys)
        | bool(incompatible.unexpected_keys)
        | bool(incompatible.incorrect_shapes)
    )
