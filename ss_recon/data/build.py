import itertools
from typing import Sequence
from collections import defaultdict
import random
import logging

from torch.utils.data import DataLoader
import numpy as np
from .catalog import DatasetCatalog
from .slice_dataset import SliceData
from .transforms import transform as T
from .transforms.subsample import build_mask_func


def get_recon_dataset_dicts(
    dataset_names: Sequence[str],
    num_scans_total: int = -1,
    num_scans_subsample: int = 0,
    seed: int = 1000,
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
            dd["_use_subsampled"] = True
    logger.info(
        "Dropped references for {}/{} scans. "
        "{} scans with reference remaining".format(
            num_scans_subsample,
            num_after,
            num_after - num_scans_subsample,
        )
    )

    return dataset_dicts


def build_recon_train_loader(cfg):
    dataset_dicts = get_recon_dataset_dicts(
        dataset_names=cfg.DATASETS.TRAIN,
        num_scans_total=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL,
        num_scans_subsample=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED,
        seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
    )
    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(cfg.AUG_TRAIN, mask_func, is_test=False)

    train_data = SliceData(dataset_dicts, data_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=cfg.DATALOADER.DROP_LAST,
        pin_memory=True,
    )
    return train_loader


def build_recon_test_loader(cfg, dataset_name):
    dataset_dicts = get_recon_dataset_dicts(dataset_names=[dataset_name])
    mask_func = build_mask_func(cfg.AUG_TRAIN)
    data_transform = T.DataTransform(cfg.AUG_TRAIN, mask_func, is_test=True)
    train_data = SliceData(dataset_dicts, data_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
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

    loaders = defaultdict(dict)
    for acc in accelerations:
        aug_cfg = cfg.AUG_TRAIN.clone()
        aug_cfg.defrost()
        aug_cfg.UNDERSAMPLE.ACCELERATIONS = (acc,)
        aug_cfg.freeze()
        for dataset_dict in dataset_dicts:
            mask_func = build_mask_func(aug_cfg)
            data_transform = T.DataTransform(
                cfg.AUG_TRAIN, mask_func, is_test=True
            )
            train_data = SliceData([dataset_dict], data_transform)
            loader = DataLoader(
                dataset=train_data,
                batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=True,
            )
            loaders["{}x".format(acc)][dataset_dict["file_name"]] = loader

    return loaders
