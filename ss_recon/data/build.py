import itertools
from typing import Sequence
from collections import defaultdict

from torch.utils.data import DataLoader
import numpy as np
from .catalog import DatasetCatalog
from .slice_dataset import SliceData
from .transforms import transform as T
from .transforms.subsample import build_mask_func


def get_recon_dataset_dicts(dataset_names: Sequence[str]):
    assert len(dataset_names)
    dataset_dicts = [
        DatasetCatalog.get(dataset_name) for dataset_name in dataset_names
    ]
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    # TODO: Drop datasets, tag datasets with "drop_target" flag, etc.

    return dataset_dicts


def build_recon_train_loader(cfg):
    dataset_dicts = get_recon_dataset_dicts(dataset_names=cfg.DATASETS.TRAIN)
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
