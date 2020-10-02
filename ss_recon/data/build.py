import itertools
from typing import Sequence
from collections import defaultdict
import random
import logging

from torch.utils.data import DataLoader
import numpy as np
from .catalog import DatasetCatalog
from .slice_dataset import SliceData, collate_by_supervision, default_collate
from .transforms import transform as T
from .transforms.subsample import build_mask_func
from .samplers import AlternatingSampler


def get_recon_dataset_dicts(
    dataset_names: Sequence[str],
    num_scans_total: int = -1,
    num_scans_subsample: int = 0,
    seed: int = 1000,
    accelerations = (),
    filter_by=(),
):
    """Get recon datasets and perform filtering.

    Given the same seed, scans will be selected in a fixed order. For example,
    let `num_scans_total=3` in experiment A and `num_scans_total=4` in
    experiment B. If the seed is the same for both experiments, then the scans
    selected in experiment A will be a subset of the scans selected in
    experiment B. This is to simulate data as a growing set- we can never
    "lose" data, only add to the existing set.

    Args:
        dataset_names (str(s)): datasets to load.
        num_scans_total (int): Number of total scans to return.
            If `-1`, ignored.
        num_scans_subsample (int): Number of scans to mark as only subsample
            scans. These scans will not have a ground truth scan.
        seed (int): the deterministic seed for filtering which scans to select.
        accelerations (sequence): the range of accelerations for this dataset.
            The maximum in the range will be used for retrospective
            undersampling of the unsupervised subset of scans.
    """
    assert len(dataset_names)
    dataset_dicts = [
        DatasetCatalog.get(dataset_name) for dataset_name in dataset_names
    ]
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    num_before = len(dataset_dicts)

    logger = logging.getLogger(__name__)

    state = random.Random(seed)
    if num_scans_total > 0 or num_scans_subsample > 0:
        # Sort to ensure same order for all users.
        # Shuffle for randomness.
        dataset_dicts = sorted(dataset_dicts, key=lambda x: x["file_name"])
        state.shuffle(dataset_dicts)

    if filter_by:
        for k, v in filter_by:
            num_before = len(dataset_dicts)
            if not isinstance(v, Sequence):
                v = (v,)
            dataset_dicts = [dd for dd in dataset_dicts if dd[k] in v]
            num_after = len(dataset_dicts)
            logger.info(
                f"Filtered by {k}: Dropped {num_before - num_after} scans. "
                f"{num_after} scans remaining"
            )

    if num_scans_total > 0:
        dataset_dicts = dataset_dicts[:num_scans_total]

    num_after = len(dataset_dicts)
    logger.info(
        "Dropped {} scans. {} scans remaining".format(
            num_before - num_after, num_after
        )
    )

    num_scans_subsample = max(0, num_scans_subsample)
    if num_scans_subsample > 0:
        if num_scans_subsample > len(dataset_dicts):
            raise ValueError("")
        for dd in dataset_dicts[:num_scans_subsample]:
            dd["_is_unsupervised"] = True
            # Select the maximum acceleration when doing undersampling.
            dd["_acceleration"] = max(accelerations)
    logger.info(
        "Dropped references for {}/{} scans. "
        "{} scans with reference remaining".format(
            num_scans_subsample,
            num_after,
            num_after - num_scans_subsample,
        )
    )

    return dataset_dicts


def _build_dataset(cfg, dataset_dicts, data_transform, dataset_type=None, is_eval=False):
    keys = cfg.DATALOADER.DATA_KEYS
    if keys:
        assert all(len(x) == 2 for x in keys), "cfg.DATALOADER.DATA_KEYS should be sequence of tuples of len 2"
        keys = {k: v for k, v in keys}

    if dataset_type is None:
        dataset_type = SliceData
    return dataset_type(dataset_dicts, data_transform, keys=keys, include_metadata=is_eval)


def build_recon_train_loader(cfg, dataset_type=None):
    dataset_dicts = get_recon_dataset_dicts(
        dataset_names=cfg.DATASETS.TRAIN,
        num_scans_total=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL,
        num_scans_subsample=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED,
        seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
        accelerations=cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS,
        filter_by=cfg.DATALOADER.FILTER.BY,
    )
    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(cfg, mask_func, is_test=False, add_noise=cfg.AUG_TRAIN.USE_NOISE)

    train_data = _build_dataset(cfg, dataset_dicts, data_transform, dataset_type)
    is_semi_supervised = len(train_data.get_unsupervised_idxs()) > 0
    collate_fn = collate_by_supervision if is_semi_supervised else default_collate

    # Build sampler.
    sampler = cfg.DATALOADER.SAMPLER_TRAIN
    shuffle = False  # shuffling should be handled by sampler, if specified.
    seed = cfg.SEED if cfg.SEED > -1 else None
    if sampler == "AlternatingSampler":
        sampler = AlternatingSampler(
            train_data,
            T_s=cfg.DATALOADER.ALT_SAMPLER.PERIOD_SUPERVISED,
            T_us=cfg.DATALOADER.ALT_SAMPLER.PERIOD_UNSUPERVISED,
            seed=seed,
        )
    elif sampler == "":
        sampler = None
        shuffle = True
    else:
        raise ValueError("Unknown Sampler {}".format(sampler))

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=cfg.DATALOADER.DROP_LAST,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return train_loader


def build_recon_val_loader(cfg, dataset_name, as_test: bool = False, add_noise: bool = False):
    dataset_dicts = get_recon_dataset_dicts(
        dataset_names=[dataset_name],
        filter_by=cfg.DATALOADER.FILTER.BY,
        num_scans_total=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL,
        seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
    )
    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(cfg, mask_func, is_test=as_test, add_noise=add_noise)

    train_data = _build_dataset(cfg, dataset_dicts, data_transform, is_eval=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
        collate_fn=default_collate,
    )
    return train_loader


def build_data_loaders_per_scan(cfg, dataset_name, accelerations=None):
    """Creates a data loader for each unique scan.

    TODO: Deprecate this function and incorporate scan action into standard
    testing pipeline
    """
    dataset_dicts = get_recon_dataset_dicts(dataset_names=[dataset_name])

    if accelerations is None:
        accelerations = cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS
        accelerations = list(np.arange(accelerations[0], accelerations[1]))

    assert len(cfg.AUG_TRAIN.UNDERSAMPLE.CENTER_FRACTIONS) <= 1, (
        "Currently only support single center fraction during testing"
    )

    loaders = defaultdict(dict)
    for acc in accelerations:
        aug_cfg = cfg.AUG_TRAIN.clone()
        aug_cfg.defrost()
        aug_cfg.UNDERSAMPLE.ACCELERATIONS = (acc,)
        aug_cfg.freeze()
        for dataset_dict in dataset_dicts:
            mask_func = build_mask_func(aug_cfg)
            data_transform = T.DataTransform(
                cfg, mask_func, is_test=True
            )
            train_data = _build_dataset(cfg, [dataset_dict], data_transform, is_eval=True)
            loader = DataLoader(
                dataset=train_data,
                batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=True,
                collate_fn=default_collate,
            )
            loaders["{}x".format(acc)][dataset_dict["file_name"]] = loader

    return loaders
