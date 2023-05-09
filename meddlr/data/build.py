import itertools
import logging
import random
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from meddlr.data.catalog import DatasetCatalog
from meddlr.data.collate import collate_by_supervision, default_collate
from meddlr.data.samplers.build import build_train_sampler, build_val_sampler
from meddlr.data.slice_dataset import SliceData
from meddlr.data.transforms import transform as T
from meddlr.data.transforms.subsample import build_mask_func
from meddlr.utils import env

__all__ = ["build_recon_train_loader", "build_recon_val_loader"]


def get_recon_dataset_dicts(
    dataset_names: Sequence[str],
    num_scans_total: Union[int, Tuple] = -1,
    num_scans_subsample: int = 0,
    seed: int = 1000,
    accelerations=(),
    filter_by=(),
):
    """Get recon datasets and perform filtering.

    Given the same seed, scans will be selected in a fixed order. For example,
    let ``num_scans_total=3`` in experiment A and ``num_scans_total=4`` in
    experiment B. If the seed is the same for both experiments, then the scans
    selected in experiment A will be a subset of the scans selected in
    experiment B. This is to simulate data as a growing set- we can never
    "lose" data, only add to the existing set.

    Args:
        dataset_names (str(s)): Datasets to load. See meddlr/data/datasets/builtin.py
            for built-in datasets.
        num_scans_total (int): Number of total scans to return. If ``-1``, ignored.
        num_scans_subsample (int): Number of scans to mark as only undersampled
            (i.e. unsupervised) scans. These scans will not have a ground truth scan.
        seed (int): The deterministic seed for filtering which scans to select.
            For reproducibility, this seed should be set.
        accelerations (Tuple[float, float], optional): The range of accelerations
            for this dataset. The maximum in the range will be used for retrospective
            undersampling of the unsupervised subset of scans.
        filter_by (Tuple): Metadata key to filter on and the value(s) to filter by.
            If the value for a particular key is a list, any one of the elements in
            the list is a valid entry.
            The schema is ``((k1, v1), (k2, v2), (k3, v3), ...)``.

    Returns:
        List[Dict]: The dataset dictionaries.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    num_before = len(dataset_dicts)

    logger = logging.getLogger(__name__)

    state = random.Random(seed)
    limit_scans = (
        num_scans_total > 0 if isinstance(num_scans_total, (float, int)) else bool(num_scans_total)
    )
    if limit_scans or num_scans_subsample != 0:
        # Sort to ensure same order for all users.
        # Shuffle for randomness.
        dataset_dicts = sorted(dataset_dicts, key=lambda x: x["file_name"])
        state.shuffle(dataset_dicts)

    if filter_by:
        for k, v in filter_by:
            num_before = len(dataset_dicts)
            if not isinstance(v, (list, tuple)):
                v = (v,)
            else:
                # This helps us ignore casting differences between
                # list and tuple.
                extra_v = []
                for _v in v:
                    if isinstance(_v, list):
                        extra_v.append(tuple(_v))
                    elif isinstance(_v, tuple):
                        extra_v.append(list(_v))
                v = list(v) + extra_v
            dataset_dicts = [
                dd
                for dd in dataset_dicts
                if dd.get(k, None) in v or dd.get("_metadata", {}).get(k, None) in v
            ]
            num_after = len(dataset_dicts)
            logger.info(
                f"Filtered by {k}: Dropped {num_before - num_after} scans. "
                f"{num_after} scans remaining"
            )

    num_after_filter = len(dataset_dicts)
    if isinstance(num_scans_total, int) and num_scans_total > 0:
        dataset_dicts = dataset_dicts[:num_scans_total]
    elif isinstance(num_scans_total, Tuple) and num_scans_total:
        dataset_dicts = _limit_data_by_group(dataset_dicts, num_scans_total)

    num_after = len(dataset_dicts)
    logger.info(
        "Dropped {} scans. {} scans remaining".format(num_after_filter - num_after, num_after)
    )

    if num_scans_subsample == -1:
        num_scans_subsample = len(dataset_dicts)
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
            num_scans_subsample, num_after, num_after - num_scans_subsample
        )
    )

    return dataset_dicts


def _limit_data_by_group(dataset_dicts, num_scans_total: Tuple[str, Dict]):
    if len(num_scans_total) > 1:
        raise ValueError("Currently can only limit scans by one field")
    metadata_key, cat_to_limit = num_scans_total[0]
    if not isinstance(cat_to_limit, Mapping):
        if not isinstance(cat_to_limit, Sequence) or len(cat_to_limit) % 2 != 0:
            raise ValueError(
                "Categories and limits must be provided as a dictionary or even-length tuple. "
                "The tuple should be (group1, limit1, group2, limit2, ...)"
            )
        cat_to_limit = {k: v for k, v in zip(cat_to_limit[::2], cat_to_limit[1::2])}
    cat_to_num = defaultdict(int)
    new_dataset_dicts = []
    for dd in dataset_dicts:
        metadata_val = dd.get(metadata_key, dd.get("_metadata", {}).get(metadata_key, None))
        for k in cat_to_limit:
            if isinstance(k, (tuple, list)) and metadata_val in k:
                metadata_val = k
                break
        if metadata_val is None or metadata_val not in cat_to_limit:
            new_dataset_dicts.append(dd)
            continue
        if cat_to_num.get(metadata_val, 0) >= cat_to_limit[metadata_val]:
            # Quota for this metadata field reached.
            continue
        new_dataset_dicts.append(dd)
        cat_to_num[metadata_val] += 1
    return new_dataset_dicts


def _build_dataset(cfg, dataset_dicts, data_transform, dataset_type=None, is_eval=False, **kwargs):
    keys = cfg.DATALOADER.DATA_KEYS
    if keys:
        assert all(
            len(x) == 2 for x in keys
        ), "cfg.DATALOADER.DATA_KEYS should be sequence of tuples of len 2"
        keys = {k: v for k, v in keys}

    if dataset_type is None:
        dataset_type = SliceData
    return dataset_type(
        dataset_dicts, data_transform, keys=keys, include_metadata=is_eval, **kwargs
    )


def _get_default_dataset_type(dataset_name):
    """Returns the default dataset type based on the dataset name.

    TODO: This function and its call hierarchy need to be refactored.
    """
    return SliceData


def build_recon_train_loader(cfg, dataset_type=None):
    if (
        cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL > 0
        and cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP
    ):
        raise ValueError(
            "`DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL` and "
            "`DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP` are mutually exclusive."
        )
    num_scans_total = (
        cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP
        if cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP
        else cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL
    )

    dataset_dicts = get_recon_dataset_dicts(
        dataset_names=cfg.DATASETS.TRAIN,
        num_scans_total=num_scans_total,
        num_scans_subsample=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED,
        seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
        accelerations=cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS,
        filter_by=cfg.DATALOADER.FILTER.BY,
    )
    if dataset_type is None:
        dataset_type = _get_default_dataset_type(cfg.DATASETS.TRAIN[0])

    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(
        cfg,
        mask_func,
        is_test=False,
        add_noise=cfg.AUG_TRAIN.USE_NOISE,
        add_motion=cfg.AUG_TRAIN.USE_MOTION,
    )

    train_data = _build_dataset(cfg, dataset_dicts, data_transform, dataset_type)
    # TODO: make this cleaner
    is_semi_supervised = (len(train_data.get_unsupervised_idxs()) > 0) | (
        cfg.MODEL.META_ARCHITECTURE == "N2RModel"
    )
    collate_fn = collate_by_supervision if is_semi_supervised else default_collate

    # Build sampler.
    sampler, is_batch_sampler = build_train_sampler(cfg, train_data)
    shuffle = not sampler  # shuffling should be handled by sampler, if specified.
    if is_batch_sampler:
        dl_kwargs = {"batch_sampler": sampler}
    else:
        dl_kwargs = {
            "sampler": sampler,
            "batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
            "shuffle": shuffle,
            "drop_last": cfg.DATALOADER.DROP_LAST,
        }

    num_workers = cfg.DATALOADER.NUM_WORKERS
    prefetch_factor = cfg.DATALOADER.PREFETCH_FACTOR
    if env.pt_version() >= "2.0" and num_workers == 0:
        prefetch_factor = None

    train_loader = DataLoader(
        dataset=train_data,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        **dl_kwargs,
    )
    return train_loader


def build_recon_val_loader(
    cfg,
    dataset_name,
    as_test: bool = False,
    add_noise: bool = False,
    add_motion: bool = False,
    dataset_type=None,
):
    if (
        cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL > 0
        and cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP
    ):
        raise ValueError(
            "`DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL` and "
            "`DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP` are mutually exclusive."
        )
    num_scans_total = (
        cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP
        if cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP
        else cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL
    )

    dataset_dicts = get_recon_dataset_dicts(
        dataset_names=[dataset_name],
        filter_by=cfg.DATALOADER.FILTER.BY,
        num_scans_total=num_scans_total,
        seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
    )
    if dataset_type is None:
        dataset_type = _get_default_dataset_type(dataset_name)

    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(
        cfg, mask_func, is_test=as_test, add_noise=add_noise, add_motion=add_motion
    )

    val_data = _build_dataset(
        cfg, dataset_dicts, data_transform, is_eval=True, dataset_type=dataset_type
    )

    # Build sampler.
    sampler, is_batch_sampler = build_val_sampler(cfg, val_data)
    if is_batch_sampler:
        dl_kwargs = {"batch_sampler": sampler}
    else:
        dl_kwargs = {
            "sampler": sampler,
            "batch_size": cfg.SOLVER.TEST_BATCH_SIZE,
            "shuffle": False,
            "drop_last": False,
        }

    val_loader = DataLoader(
        dataset=val_data,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=default_collate,
        prefetch_factor=cfg.DATALOADER.PREFETCH_FACTOR,
        **dl_kwargs,
    )
    return val_loader


def build_data_loaders_per_scan(cfg, dataset_name, accelerations=None):
    """Creates a data loader for each unique scan.

    TODO: Deprecate this function and incorporate scan action into standard
    testing pipeline
    """
    dataset_dicts = get_recon_dataset_dicts(dataset_names=[dataset_name])

    if accelerations is None:
        accelerations = cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS
        accelerations = list(np.arange(accelerations[0], accelerations[1]))

    assert (
        len(cfg.AUG_TRAIN.UNDERSAMPLE.CENTER_FRACTIONS) <= 1
    ), "Currently only support single center fraction during testing"

    loaders = defaultdict(dict)
    for acc in accelerations:
        aug_cfg = cfg.AUG_TRAIN.clone()
        aug_cfg.defrost()
        aug_cfg.UNDERSAMPLE.ACCELERATIONS = (acc,)
        aug_cfg.freeze()
        for dataset_dict in dataset_dicts:
            mask_func = build_mask_func(aug_cfg)
            data_transform = T.DataTransform(cfg, mask_func, is_test=True)
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
