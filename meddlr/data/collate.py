from typing import Dict, List

from torch.utils.data.dataloader import default_collate as _default_collate

__all__ = ["default_collate", "collate_by_supervision"]


def default_collate(batch: List[Dict]):
    """Default collate function for meddlr.

    This collate function handles metadata appropriately by returning
    metadata as a list of dictionaries instead of a dictionary of tensors.
    This is done because not all metadata (e.g. string values) can be
    tensorized.

    Metadata is only handled if at least one example in the batch has
    a ``'metadata'`` key.

    Args:
        batch (list): The list of dictionaries.

    Returns:
        Dict
    """
    metadata = None
    if any("metadata" in b for b in batch):
        metadata = [b.pop("metadata", None) for b in batch]
    out_dict = _default_collate(batch)
    if metadata is not None:
        out_dict["metadata"] = metadata
    return out_dict


def collate_by_supervision(batch: list):
    """Collate supervised/unsupervised batch examples.

    This collate function is required when training with semi-supervised
    models, such as :cls:`N2RModel` and :cls:`VortexModel`.

    Args:
        batch (list): The list of dictionaries.

    Returns:
        Dict[str, Dict]: A dictionary with 2 keys, ``'supervised'`` and ``'unsupervised'``.
    """
    supervised = [x for x in batch if not x.get("is_unsupervised", False)]
    unsupervised = [x for x in batch if x.get("is_unsupervised", False)]

    out_dict = {}
    if len(supervised) > 0:
        supervised = default_collate(supervised)
        out_dict["supervised"] = supervised
    if len(unsupervised) > 0:
        unsupervised = default_collate(unsupervised)
        out_dict["unsupervised"] = unsupervised
    assert len(out_dict) > 0
    return out_dict
