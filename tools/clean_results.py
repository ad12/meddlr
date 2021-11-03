"""
This script finds top weights and optionall removes all other weights
from your results directory.

Usage:

    # Find top val_psnr weights and traverse directories interactively
    $ python clean_results.py --dir /path/to/directory --top_k 1

    # Find top 3 val_psnr and val_l1 weights interactively. Delete all other weights
    $ python clean_results.py --dir /path/to/directory --top_k 3 --metrics psnr l1 \
        --interactive --remove
"""
import argparse
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Collection, Sequence, Union

import pandas as pd
from tabulate import tabulate

from meddlr.config import get_cfg
from meddlr.evaluation.testing import find_weights
from meddlr.utils import cluster  # noqa
from meddlr.utils.general import find_experiment_dirs


def clean_results(
    dirpath: Union[str, Path],
    top_k: int,
    metrics: Union[str, Sequence[str]] = None,
    iter_limit: Union[int, Sequence[str]] = None,
    interactive: bool = False,
    remove: bool = False,
):
    """Remove all extra weight files that are no longer needed.

    We do not remove the ``'model_final.pth'`` file as this is one way we
    identify which runs are complete.

    Args:
        dirpath: Directory to recursively search.
        top_k (int): The number of top weights to keep.
        metrics: The metrics(s) to use for finding weights.
        interactive (bool, optional): If `True`, this method will
            wait for your input before deleting all weights.
        remove (bool, optional): If ``True``, weight will be deleted. Otherwise,
            the weights will just be printed out.
    """
    if isinstance(metrics, str) or not isinstance(metrics, Collection):
        metrics = [metrics]
    if not isinstance(iter_limit, Collection):
        iter_limit = [iter_limit]

    all_exp_paths = find_experiment_dirs(dirpath, completed=True)
    for exp_path in all_exp_paths:
        exp_path = os.path.abspath(exp_path)
        print(exp_path)
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(exp_path, "config.yaml"))

        data = defaultdict(list)
        for _metric, _iter_limit in itertools.product(metrics, iter_limit):
            if str(_metric).lower() == "none":
                _metric = None
            if str(_iter_limit).lower() == "none":
                _iter_limit = None
            filepaths, criterion, best_vals = find_weights(
                cfg, criterion=_metric, iter_limit=_iter_limit, top_k=top_k
            )
            if isinstance(filepaths, (str, Path)):
                filepaths = [filepaths]
                best_vals = [best_vals]
            data["criterion"].extend([criterion] * len(filepaths))
            data["iter_limit"].extend([str(_iter_limit)] * len(filepaths))
            data["filepath"].extend(filepaths)
            data["criterion_val"].extend(best_vals)

        data = pd.DataFrame(data)
        print(tabulate(data, headers=data.columns))

        filepaths_to_keep = set(data["filepath"].tolist()) | {
            os.path.join(exp_path, "model_final.pth")
        }
        all_model_paths = {
            os.path.abspath(os.path.join(exp_path, x))
            for x in os.listdir(exp_path)
            if x.endswith(".pth")
        }
        remove_files = all_model_paths - filepaths_to_keep

        print(
            "Found {}/{} files to remove from {}:\n\t{}".format(
                len(remove_files), len(all_model_paths), exp_path, "\n\t".join(remove_files)
            )
        )
        if interactive:
            if remove:
                key = input(
                    "Remove {}/{} files? (y|[n]) ".format(len(remove_files), len(all_model_paths))
                )
                if key not in ["y"]:
                    sys.exit(0)
            else:
                key = input("Press any key to continue")

        if remove:
            for fp in remove_files:
                os.remove(fp)


def main():
    parser = argparse.ArgumentParser("Clean out results folder")
    parser.add_argument(
        "--dir", help="Results directory to recursively search", required=True, type=str
    )
    parser.add_argument("--top_k", help="Top k weights to store", required=True, type=int)
    parser.add_argument("--iter_limit", help="Iteration limits", default=None, nargs="*")
    parser.add_argument("--metrics", help="Metrics to use", default=None, nargs="*")
    parser.add_argument("--interactive", help="Run interactively", action="store_true")
    parser.add_argument("--remove", help="Remove weights files", action="store_true")

    args = parser.parse_args()

    clean_results(
        args.dir,
        args.top_k,
        args.metrics,
        iter_limit=args.iter_limit,
        interactive=args.interactive,
        remove=args.remove,
    )


if __name__ == "__main__":
    main()
