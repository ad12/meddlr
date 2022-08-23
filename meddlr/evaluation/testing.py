import json
import logging
import os
import pprint
import re
from numbers import Number
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import torch

from meddlr.config import CfgNode


def print_csv_format(results: Dict[str, Dict[str, Number]]):  # pragma: no cover
    """Print metrics for easy copypaste.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, Dict), results  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    important_res = [(k, v) for k, v in results.items() if "-" not in k]
    logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
    logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))

    # Additional formatting
    logger.info(
        "Metrics (comma delimited): \n{}\n{}".format(
            ",".join([k[0] for k in important_res]),
            ",".join(["{0:.4f}".format(k[1]) for k in important_res]),
        )
    )


def verify_results(cfg: CfgNode, results: Dict[str, Any]) -> bool:  # pragma: no cover
    """Verify that the results are consistent with what is expected in the config.

    Adapted from detectron2:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/testing.py

    Args:
        results: A mapping from metrics -> scores.

    Returns:
        bool: Whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    results = flatten_results_dict(results)

    ok = True
    for metric, expected, tolerance in expected_results:
        actual = results[metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results: Dict[str, Any], delimiter: str = "/") -> Dict[str, Number]:
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + delimiter + kk] = vv
        else:
            r[k] = v
    return r


# Supported validation metrics and the operation to perform on them.
# TODO (arjundd): Do not hardcode these. Match to metrics in meddlr instead.
_METRICS_TO_OPERATION = {
    "l1": "min",
    "l2": "min",
    "psnr": "max",
    "ssim": "max",
    "iteration": "max",  # find the last checkpoint
    "loss": "min",
    "mae": "min",
}


def find_metrics(
    cfg: CfgNode, criterion: str, operation: str = "auto", iter_limit: int = None, top_k: int = 1
):
    """Find the iteration(s) resulting in best metrics for a given criterion.

    Args:
        cfg: The config.
        criterion: The criterion that we use to select weights.
        operation: The operation for selecting the best value of the criterion.
            One of `'auto'`, `'min'`, `'max'`.
        iter_limit: If specified, only iterations ``<=iter_limit` will be searched.
            If this value is negative, it is interpreted as the epoch limit.
            A positive value is recommended for avoiding ambiguity.
        top_k: The number of (iteration, metric) pairs to return.
            If ``-1``, return all pairs in sorted in order of best to worst.

    Returns:
        List[Tuple[int, float]] | Tuple[int, float]: A list of tuples of the
            form (iteration, metric) if ``top_k > 1``. Otherwise, a tuple of
            the form (iteration, metric).

    Note:
        This function is experimental. The API may change without warning.
    """
    metric_details = _find_metric_details(
        cfg=cfg,
        criterion=criterion,
        operation=operation,
        iter_limit=iter_limit,
        top_k=-1,
    )
    best_iter_and_values = metric_details["best_iter_and_values"][:top_k]
    return best_iter_and_values[0] if top_k == 1 else best_iter_and_values


def find_weights(
    cfg: CfgNode,
    criterion: str,
    operation: str = "auto",
    iter_limit: int = None,
    file_name_fmt: str = "model_{:07d}.pth",
    top_k: int = 1,
) -> Tuple[Union[str, List[str]], str, Union[float, List[float]]]:
    """Find the best weights based on a metric criterion.

    TODO (arjundd): Use :func:`find_metrics` to find the appropriate metrics.

    Args:
        cfg: The config.
        criterion (str): The criterion that we use to select weights.
        operation (str, optional): The operation for the best value of the criterion.
            One of `'auto'`, `'min'`, `'max'`.
        iter_limit (int, optional): If specified, all weights will be before
            this iteration. If this value is negative, it is
            interpreted as the epoch limit.
        file_name_fmt (int, optional): The naming format for checkpoint files.
        top_k (int, optional): The number of top checkpoints to return.

    Returns:
        Tuple: ``k`` filepath(s), selection criterion, and ``k`` criterion value(s).
            If ``k=1``, filepath is a string and value is a float.

    Examples:
        >>> checkpoint_file, criterion, value = find_weights(cfg, "val_loss", "min")

    Note:
        This function is experimental. The API may change without warning.
    """
    logger = logging.getLogger(__name__)

    metric_details = _find_metric_details(
        cfg=cfg,
        criterion=criterion,
        operation=operation,
        iter_limit=iter_limit,
        top_k=-1,
    )
    best_iter_and_values = metric_details["best_iter_and_values"][:top_k]
    last_iter = metric_details["last_iter"]

    all_filepaths = []
    all_values = []
    potential_ckpt_files = os.listdir(cfg.OUTPUT_DIR)
    for best_iter, best_value in best_iter_and_values:
        file_name = file_name_fmt.format(best_iter)

        matched_files = [x for x in potential_ckpt_files if re.match(file_name, x)]
        if len(matched_files) > 1:
            raise ValueError(
                f"Too many potential checkpoint files found for iter={best_iter}, "
                f"criterion={criterion}, value={best_value}:\n\t{matched_files}"
            )
        if len(matched_files) == 0:
            if best_iter == last_iter:
                file_name = "model_final.pth"
            matched_files = [file_name for x in potential_ckpt_files if re.match(file_name, x)]
        if len(matched_files) == 0:
            raise ValueError(
                f"Could not find potential checkpoint files for iter={best_iter}, "
                f"criterion={criterion}, value={best_value}."
            )
        file_name = matched_files[0]

        file_path = os.path.join(cfg.OUTPUT_DIR, file_name)

        if not os.path.isfile(file_path):
            raise ValueError("Model for iteration {} does not exist".format(best_iter))

        all_filepaths.append(file_path)
        all_values.append(best_value)

        logger.info("Weights: {} - {}: {:0.4f}".format(file_name, criterion, best_value))

    if top_k == 1:
        return all_filepaths[0], criterion, all_values[0]
    else:
        return all_filepaths, criterion, all_values


def check_consistency(state_dict: Dict[str, Any], model: torch.nn.Module):
    """Verifies that the proper weights were loaded into the model.

    Related to issue that loading weights from checkpoints of a cuda model
    does not properly load when model is on cpu. It may also result in
    warnings for contiguous tensors, but this is not always the case.
    https://github.com/pytorch/pytorch/issues/42300

    Args:
        state_dict (Dict): A model state dict.
        model (nn.Module): A Pytorch model.
    """
    _state_dict = model.state_dict()
    for k in state_dict:
        assert k in _state_dict, f"{k} not in model state_dict: {_state_dict.keys()}"
        assert torch.equal(state_dict[k], _state_dict[k]), f"Mismatch values: {k}"


def _metrics_from_x(metrics_file, criterion):
    metrics_file = os.path.splitext(metrics_file)[0]
    if os.path.isfile(f"{metrics_file}.json"):
        return _metrics_from_json(f"{metrics_file}.json", criterion)
    elif os.path.isfile(f"{metrics_file}.csv"):
        return _metrics_from_csv(f"{metrics_file}.csv", criterion)
    else:
        raise ValueError(f"metrics file not found - {metrics_file}")


def _metrics_from_json(metrics_file, criterion):
    metrics = []
    with open(metrics_file, "r") as f:
        metrics = [json.loads(line.strip()) for line in f]
    metrics = [m for m in metrics if criterion in m or any(k.endswith(criterion) for k in m.keys())]
    is_metric_wrapped = criterion not in metrics[0]
    if is_metric_wrapped:
        metrics = [
            (
                int(m["iteration"]),
                np.mean(
                    [m[k] for k in m if k.endswith(criterion) and "test" not in k.split("/")[0]]
                ).item(),
            )
            for m in metrics
        ]
    else:
        metrics = [(m["iteration"], m[criterion]) for m in metrics]
    return metrics


def _metrics_from_csv(metrics_file, criterion):
    def _row_has_criterion(row: pd.Series):
        index = row.index.tolist()
        return criterion in index and not pd.isna(row[criterion])

    metrics = pd.read_csv(metrics_file)
    metrics = [m for _, m in metrics.iterrows() if _row_has_criterion(m)]
    metrics = [(int(m["step"]), m[criterion]) for m in metrics]
    return metrics


def get_iters_per_epoch_eval(cfg) -> int:
    """Get number of iterations per epoch for evaluation purposes.

    This function expects a metrics named ``{cfg.OUTPUT_DIR}/metrics.json``
    that is written during training.

    Args:
        cfg: The config.

    Return:
        int: Number of iterations per epoch.
    """
    exp_path = cfg.OUTPUT_DIR

    eval_period = cfg.TEST.EVAL_PERIOD
    time_scale = cfg.TIME_SCALE
    assert time_scale in ["epoch", "iter"]
    if (time_scale == "epoch" and eval_period < 0) or (time_scale == "iter" and eval_period > 0):
        raise ValueError("Evaluation period is not in # epochs")
    eval_period = abs(eval_period)

    with open(os.path.join(exp_path, "metrics.json"), "r") as f:
        metrics = [json.loads(line.strip()) for line in f.readlines()]
    # Filter to only have evaluation metrics.
    metrics = [m for m in metrics if "eval_time" in m]
    # last eval could be at end of training which doesnt have same
    # spacing of iters_per_epoch - skip it
    iterations = np.diff([m["iteration"] for m in metrics][:-1])
    if len(iterations) == 0:
        raise ValueError("Could not determine iters_per_epoch - too few evaluations")
    if len(iterations) >= 2 and not np.all(iterations[:-1] == iterations[0]):
        raise ValueError("Not all iteration spacings are equal - {iterations}")

    iters_per_epoch = iterations[0] / eval_period
    assert iters_per_epoch == int(iters_per_epoch)
    return int(iters_per_epoch)


def _find_metric_details(
    cfg: CfgNode,
    criterion: str,
    operation: str = "auto",
    iter_limit: int = None,
    top_k: int = 1,
):
    """Find the iteration(s) resulting in best metrics for a given criterion.

    Args:
        cfg: The config.
        criterion: The criterion that we use to select weights.
        operation: The operation for selecting the best value of the criterion.
            One of `'auto'`, `'min'`, `'max'`.
        iter_limit: If specified, only iterations ``<=iter_limit` will be searched.
            If this value is negative, it is interpreted as the epoch limit.
            A positive value is recommended for avoiding ambiguity.
        top_k: The number of (iteration, metric) pairs to return.
            If ``-1``, return all pairs in sorted in order of best to worst.

    Returns:
        List[Tuple[int, float]] | Tuple[int, float]: A list of tuples of the
            form (iteration, metric) if ``top_k > 1``. Otherwise, a tuple of
            the form (iteration, metric). Also returns last iteration if requested.

    Note:
        This function is experimental. The API may change without warning.
    """
    logger = logging.getLogger(__name__)

    if operation not in ["min", "max", "auto"]:
        raise ValueError(f"Invalid operation: {operation}. Expected one of 'min', 'max', 'auto'.")

    # Negative iter_limit is interpreted as epoch limit.
    if iter_limit is not None and iter_limit < 0:
        iter_limit = int(abs(iter_limit) * get_iters_per_epoch_eval(cfg))

    ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.TEST.EVAL_PERIOD
    if (
        ckpt_period * eval_period <= 0  # same sign (i.e. same time scale)
        or abs(eval_period) % abs(ckpt_period) != 0  # checkpoint period is multiple of eval period
    ):
        raise ValueError(  # pragma: no cover
            "Cannot find weights if checkpoint/eval periods "
            "at different time scales or eval period is not "
            "a multiple of checkpoint period."
        )

    if operation == "auto":
        operation = [
            op for name, op in _METRICS_TO_OPERATION.items() if name.lower() in criterion.lower()
        ]
        if len(operation) == 0:
            raise ValueError(f"Could not find operation for criterion '{criterion}'.")
        if len(operation) > 1:
            raise ValueError(f"Found multiple operations for criterion '{criterion}': {operation}")
        operation = operation[0]
    assert operation in ["min", "max"]

    logger.info("Finding best weights in {} using {}...".format(cfg.OUTPUT_DIR, criterion))

    # Filter metrics to find reporting of real validation metrics.
    # If metric is wrapped (e.g. "mridata_knee_2019_val/val_l1"), that means
    # multiple datasets were validated on.
    # We filter out metrics from datasets that contain the word "test".
    # The criterion from all other datasets are averaged and used as the
    # target criterion.
    metrics_file = os.path.join(cfg.OUTPUT_DIR, "metrics")
    try:
        metrics = _metrics_from_x(metrics_file, criterion)
    except IndexError:
        raise ValueError(f"No metrics found matching criterion '{criterion}'.")

    # Last iteration that was logged.
    # Note if the run did not complete, `model_final.pth` may not correspond
    # with this iteration. Use with caution.
    last_iter = metrics[-1][0]

    # Filter out all metrics calculated above iter limit.
    if iter_limit:
        metrics = [x for x in metrics if x[0] < iter_limit]

    # Retraining does not overwrite the metrics file.
    # We make sure that the metrics correspond only to the most
    # recent run.
    metrics_recent_order = metrics[::-1]
    iterations = [m[0] for m in metrics_recent_order]
    intervals = np.diff(iterations)
    if any(intervals > 0):
        stop_idx = np.argmax(intervals > 0) + 1
        metrics_recent_order = metrics_recent_order[:stop_idx]
        metrics = metrics_recent_order[::-1]

    # Note that resuming can sometimes report metrics for the same
    # iteration. We handle this by taking the most recent metric for the
    # iteration __after__ filtering out old training runs.
    # metrics = {iteration: value for iteration, value in metrics}
    metrics = [(iteration, value) for iteration, value in metrics]

    best_iter_and_values = sorted(metrics, key=lambda x: x[1], reverse=operation == "max")
    if top_k > 0:
        best_iter_and_values = best_iter_and_values[:top_k]
    return {
        "best_iter_and_values": best_iter_and_values,
        "last_iter": last_iter,
    }
