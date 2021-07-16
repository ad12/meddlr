from ss_recon.data.samplers.group_sampler import GroupSampler
from ss_recon.data.samplers.sampler import AlternatingSampler


def build_train_sampler(cfg, dataset):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
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
    elif sampler == "":
        sampler = None
    else:
        raise ValueError("Unknown Sampler {}".format(sampler))

    return sampler, is_batch_sampler


def build_val_sampler(cfg, dataset):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    sampler = cfg.DATALOADER.SAMPLER_TRAIN
    is_batch_sampler = False
    seed = cfg.SEED if cfg.SEED > -1 else None
    if sampler == "GroupSampler" and cfg.DATALOADER.GROUP_SAMPLER.BATCH_BY:
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
    else:
        sampler = None

    return sampler, is_batch_sampler
