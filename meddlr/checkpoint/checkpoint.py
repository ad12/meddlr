import logging

from fvcore.common.checkpoint import Checkpointer as _Checkpointer
from fvcore.common.checkpoint import _strip_prefix_if_present

logger = logging.getLogger(__name__)


class Checkpointer(_Checkpointer):
    """
    Extension of fvcore's Checkpointer class that handles backwards compatibility
    and systematic state_dict unpacking.

    To use the full functionality of this class, all saved models should follow
    the convention that wrapped models begin with the prefix ``'model.'``.
    See :cls:`meddlr.modeling.meta_arch.GeneralizedUnrolledCNN`.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=True, **checkpointables)

    def _load_file(self, filename):
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
