from torch.utils.data import DataLoader
from typing import Sequence
import itertools
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
    mask_func = build_mask_func(cfg.AUG_TRAIN.UNDERSAMPLE)
    data_transform = T.DataTransform(cfg.AUG_TRAIN, mask_func, is_test=False)

    train_data = SliceData(dataset_dicts, data_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.DATALOADER.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=cfg.DATALOADER.DROP_LAST,
        pin_memory=True,
    )
    return train_loader


def build_recon_test_loader(cfg, dataset_name):
    pass
