from torch.utils.data import DistributedSampler

from meddlr.data.samplers.group_sampler import (
    AlternatingGroupSampler,
    DistributedGroupSampler,
    GroupSampler,
)
from meddlr.data.samplers.sampler import AlternatingSampler

__all__ = ["build_train_sampler", "build_val_sampler"]


def build_train_sampler(cfg, dataset, distributed=False):
    """Builds the training sampler from the config.

    Args:
        cfg (CfgNode): The config. The ``cfg.DATALOADER.SAMPLER_TRAIN``field
            will be used to determine the sampler type.
        dataset (torch.data.Dataset): The training dataset. This dataset
            must follow the Meddlr dataset convention. See :cls:`SliceDataset`
            for more information.
        distributed (bool, optional): Whether to use a distributed sampler.
            Note custom samplers (e.g. AlternatingSampler/GroupSampler)
            do not support this argument.

    Returns:
        torch.utils.data.sampler.Sampler: The sampler.
    """
    sampler = cfg.DATALOADER.SAMPLER_TRAIN
    is_batch_sampler = False
    seed = cfg.SEED if cfg.SEED > -1 else None
    if sampler == "AlternatingSampler":
        sampler = AlternatingSampler(
            dataset,
            T_s=cfg.DATALOADER.ALT_SAMPLER.PERIOD_SUPERVISED,
            T_us=cfg.DATALOADER.ALT_SAMPLER.PERIOD_UNSUPERVISED,
            seed=seed,
        )
    elif sampler == "GroupSampler":
        is_batch_sampler = cfg.DATALOADER.GROUP_SAMPLER.AS_BATCH_SAMPLER
        sampler = GroupSampler(
            dataset,
            batch_by=cfg.DATALOADER.GROUP_SAMPLER.BATCH_BY,
            batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
            as_batch_sampler=is_batch_sampler,
            drop_last=cfg.DATALOADER.DROP_LAST,
            shuffle=True,
            seed=seed,
        )
    elif sampler == "AlternatingGroupSampler":
        is_batch_sampler = cfg.DATALOADER.GROUP_SAMPLER.AS_BATCH_SAMPLER
        sampler = AlternatingGroupSampler(
            dataset,
            T_s=cfg.DATALOADER.ALT_SAMPLER.PERIOD_SUPERVISED,
            T_us=cfg.DATALOADER.ALT_SAMPLER.PERIOD_UNSUPERVISED,
            batch_by=cfg.DATALOADER.GROUP_SAMPLER.BATCH_BY,
            batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
            as_batch_sampler=is_batch_sampler,
            drop_last=cfg.DATALOADER.DROP_LAST,
            seed=seed,
        )
    elif sampler in ("", None):
        sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    else:
        raise ValueError("Unknown Sampler {}".format(sampler))

    return sampler, is_batch_sampler


def build_val_sampler(cfg, dataset, distributed: bool = False, dist_group_by="file_name"):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    sampler = cfg.DATALOADER.SAMPLER_TRAIN
    is_batch_sampler = False
    seed = cfg.SEED if cfg.SEED > -1 else None
    if (
        sampler in ("GroupSampler", "AlternatingGroupSampler")
        and cfg.DATALOADER.GROUP_SAMPLER.BATCH_BY
    ):
        is_batch_sampler = True
        sampler = GroupSampler(
            dataset,
            batch_by=cfg.DATALOADER.GROUP_SAMPLER.BATCH_BY,
            batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
            as_batch_sampler=is_batch_sampler,
            drop_last=False,
            shuffle=False,
            seed=seed,
        )
    elif sampler in ("", None) and distributed and dist_group_by:
        sampler = DistributedGroupSampler(dataset, group_by=dist_group_by, shuffle=False)
    else:
        sampler = None

    return sampler, is_batch_sampler
