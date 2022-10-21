import copy
import itertools
import logging
import os
import time
from collections import defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import silx.io.dictdump as sio
import torch
from tqdm import tqdm

import meddlr.utils.comm as comm
from meddlr.config.config import CfgNode
from meddlr.data.transforms.transform import build_normalizer
from meddlr.evaluation.scan_evaluator import ScanEvaluator, structure_scans
from meddlr.forward.mri import hard_data_consistency
from meddlr.metrics.build import build_metrics
from meddlr.metrics.collection import MetricCollection
from meddlr.ops import complex as cplx


class ReconEvaluator(ScanEvaluator):
    """Image reconstruction evaluator.

    This evaluator can be used for image reconstruction, recovery and generation tasks.
    It uses image quality metrics (e.g. SSIM, PSNR) to evaluate the quality of
    the reconstructed images. For more details on metrics, see :mod:`meddlr.metrics.build`.

    This evaluator also supports restacking slices into volumes.
    To compute a metric on the full volume (i.e. scan), use metrics with the
    suffix ``_scan`` (e.g. ``'psnr_scan'``).
    """

    def __init__(
        self,
        dataset_name: str,
        cfg: CfgNode,
        distributed: bool = False,
        sync_outputs: bool = False,
        aggregate_scans: bool = True,
        group_by_scan: bool = False,
        output_dir: Optional[str] = None,
        skip_rescale: bool = False,
        save_scans: bool = False,
        metrics: Sequence[str] = None,
        flush_period: int = None,
        to_cpu: bool = False,
        channel_names: Optional[Sequence[str]] = None,
        eval_in_process: bool = False,
        structure_channel_by=None,
        prefix: str = "val",
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset.
            distributed (bool, optional): If ``True``, collect results from all
                ranks for evaluation. Otherwise, will evaluate the results in the
                current process. If using ``DistributedDataParallel``, this should likely
                be ``True``.
            sync_outputs (bool, optional): If ``True``, synchronizes all predictions
                before evaluation. If ``False``, synchronizes metrics before reduction.
                Ignored if ``distributed=False``.
            aggregate_scans (bool, optional): If ``True``, also computes metrics per
                scan under the label `'scan_{metric}'` (e.g. scan_l1).
            group_by_scan (bool, optional): If `True`, groups metrics by scan.
                `self.evaluate()` will return a dict of scan_id -> dict[metric name, metric value]
            skip_rescale (bool, optional): If `True`, skips rescaling the output and target
                by the mean/std.
            save_scans (bool, optional): If `True`, saves predictions to .npy file.
            metrics (Sequence[str], optional): To avoid computing metrics, set to ``False``.
                Defaults to all supported recon metrics. To process metrics on the full scan,
                append ``'_scan'`` to the metric name (e.g. `'psnr_scan'`).
            flush_period (int, optional): The approximate period over which predictions
                are cleared and running results are computed. This parameter helps
                mitigate OOM errors. The period is equivalent to number of examples
                (not batches).
            to_cpu (bool, optional): If ``True``, move all data to the cpu to do computation.
            eval_in_process (bool, optional): If ``True``, run slice/patch evaluation
                while processing. This may increase overall speed.
            prefix (str): prefix to add to metric names.
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = build_normalizer(cfg)
        self._group_by_scan = group_by_scan
        self._distributed = distributed
        self._sync_outputs = sync_outputs
        self._aggregate_scans = aggregate_scans
        self._skip_rescale = skip_rescale
        self._channel_names = channel_names
        self._structure_channel_by = structure_channel_by
        self._prefix = prefix
        self._postprocess = cfg.TEST.POSTPROCESSOR.NAME
        self.device = cfg.MODEL.DEVICE

        if save_scans and (not output_dir or not aggregate_scans):
            raise ValueError("`output_dir` and `aggregate_scans` must be specified to save scans.")
        self._save_scans = save_scans
        self._save_scan_dir = os.path.join(self._output_dir, "pred") if self._output_dir else None
        if self._save_scan_dir:
            os.makedirs(self._save_scan_dir, exist_ok=True)

        if metrics is False:
            metrics = []
        elif metrics in (None, True):
            metrics = self.default_metrics()
        self._metric_names = metrics

        self._results = None

        if flush_period is None:
            flush_period = cfg.TEST.FLUSH_PERIOD
        self.flush_period = flush_period
        self.to_cpu = to_cpu
        self.eval_in_process = eval_in_process

        self._remaining_state = None
        self._predictions = []
        self._is_flushing = False

        # Memory
        self._memory = defaultdict(list)

    @classmethod
    def default_metrics(cls) -> List[str]:
        """The default metrics processed by this class."""
        metrics = ["psnr", "psnr_mag", "ssim (Wang)", "nrmse", "nrmse_mag"]
        metrics.extend([f"{x}_scan" for x in metrics])
        return metrics

    def reset(self):
        self._remaining_state = None
        self._predictions = []
        self._is_flushing = False
        self._memory = defaultdict(list)

        metrics = self._metric_names
        prefix = self._prefix + "_" if self._prefix else ""
        slice_metrics = [m for m in metrics if not m.endswith("_scan")]
        scan_metrics = [m[:-5] for m in metrics if m.endswith("_scan")]
        self.slice_metrics = build_metrics(
            slice_metrics,
            fmt=prefix + "{}",
            channel_names=self._channel_names,
        ).to(self.device)
        self.scan_metrics = build_metrics(
            scan_metrics,
            fmt=prefix + "{}_scan",
            channel_names=self._channel_names,
        ).to(self.device)

        self.slice_metrics.eval()
        self.scan_metrics.eval()

    def exit_prediction_scope(self):
        ret_val = super().exit_prediction_scope()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return ret_val

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a recon model (e.g., GeneralizedRCNN).
                Currently this should be an empty dictionary.
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.

        Note:
            All elements in ``inputs`` and ``outputs`` should already
            be detached from the computational graph.
        """
        N = outputs["pred"].shape[0]

        preds: torch.Tensor
        targets: torch.Tensor

        # Hacky way to postprocess the targets with hard data consistency (if specified).
        if "hard_dc" in self._postprocess:
            outputs["pred"] = hard_data_consistency(
                outputs["pred"],
                acq_kspace=inputs["kspace"],
                mask=inputs["postprocessing_mask"],
                maps=inputs["maps"],
            )

        if self._skip_rescale:
            # Do not rescale the outputs
            preds = outputs["pred"]
            targets = outputs["target"]
        else:
            normalized = self._normalizer.undo(
                image=outputs["pred"],
                target=outputs["target"],
                mean=inputs["mean"],
                std=inputs["std"],
                channels_last=True,
            )
            preds = normalized["image"]
            targets = normalized["target"]

        if self.to_cpu:
            preds = preds.to(self._cpu_device, non_blocking=True)
            targets = targets.to(self._cpu_device, non_blocking=True)

        if self.eval_in_process:
            self.evaluate_prediction(
                {"pred": preds, "target": targets},
                self.slice_metrics,
                [
                    "-".join([str(md[field]) for field in ("scan_id", "slice_id")])
                    for md in inputs["metadata"]
                ],
                is_batch=True,
            )

        self._predictions.extend(
            [
                {
                    "pred": preds[i],
                    "target": targets[i],
                    "metadata": inputs["metadata"][i] if "metadata" in inputs else {},
                }
                for i in range(N)
            ]
        )
        self._append_memory("after_predictions")

        has_num_examples = self.flush_period > 0 and len(self._predictions) >= self.flush_period
        has_num_scans = self.flush_period < 0 and len(
            {x["metadata"]["scan_id"] for x in self._predictions}
        ) > abs(self.flush_period)
        if has_num_examples or has_num_scans:
            self.flush(enter_prediction_scope=True, skip_last_scan=True)
            self._append_memory("after_flush")

    def structure_scans(self, verbose=True):
        """Structure scans into volumes to be used to evaluation."""
        structure_channel_by = self._structure_channel_by
        structure_by = {0: "slice_id"}
        if structure_channel_by is not None:
            # This does not work when predictions are real/imaginary are separate channels
            # TODO: Fix this.
            structure_by[-1] = structure_channel_by
        to_struct = ("pred", "target")

        # Making a tensor contiguous can be an expensive operation.
        # We want to do it as few times as possible. Because we have to
        # do it anyway when we squeeze the tensor when structuring by channel,
        # we opt not to do when first structuring the scans.
        contiguous = structure_channel_by is None

        out = structure_scans(
            self._predictions,
            to_struct=to_struct,
            dims=structure_by,
            contiguous=contiguous,
            verbose=verbose,
        )

        if structure_channel_by is not None:
            for scan_id in out:
                out[scan_id].update(
                    {k: out[scan_id][k].squeeze(-2).contiguous() for k in to_struct}
                )

        return out

    def synchronize_predictions(self):
        comm.synchronize()
        self._predictions = comm.gather(self._predictions, dst=0)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            self._predictions = []

    def evaluate(self):
        # Sync predictions (if applicable)
        if self._distributed and self._sync_outputs:
            self.synchronize_predictions()
            if not self._predictions:
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[ReconEvaluator] Did not receive valid predictions.")
            return {}

        # Compute metrics per slice (if not already done during process step).
        if not self.eval_in_process:
            for pred in tqdm(
                self._predictions, desc="Slice metric", disable=not comm.is_main_process()
            ):
                self.evaluate_prediction(
                    pred,
                    self.slice_metrics,
                    "-".join([str(pred["metadata"][x]) for x in ("scan_id", "slice_id")]),
                )

        # Compute metrics per scan.
        has_metadata = bool(self._predictions[0]["metadata"])
        if self._aggregate_scans and has_metadata:
            scans = self.structure_scans()
            for scan_id, pred in tqdm(
                scans.items(), desc="Scan Metrics", disable=not comm.is_main_process()
            ):
                self.evaluate_prediction(pred, self.scan_metrics, scan_id)
                if self._save_scans:
                    sio.dicttoh5(
                        {"pred": pred["pred"].cpu()},
                        os.path.join(self._save_scan_dir, f"{scan_id}.h5"),
                    )

        if self._group_by_scan:
            pred_vals = self._group_results_by_scan()
        else:
            pred_vals = self.slice_metrics.to_dict()
            pred_vals.update(self.scan_metrics.to_dict())

        self._results = pred_vals

        if not self._is_flushing:
            self.log_summary()

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _group_results_by_scan(self):
        """Get results grouping by scan."""
        pred_vals = defaultdict(dict)

        slice_metrics = self.slice_metrics.to_dict(group_by=["id", "Metric"])
        agg_slice_metrics = defaultdict(lambda: defaultdict(list))
        for (slice_id, metric_name), value in slice_metrics.items():
            agg_slice_metrics[slice_id.rsplit("-", 1)[0]][metric_name].append(value)
        for scan_id, metrics_dict in agg_slice_metrics.items():
            for metrics_name, values in metrics_dict.items():
                pred_vals[scan_id][metrics_name] = np.mean(values)

        scan_metrics = self.scan_metrics.to_dict(group_by=["id", "Metric"])
        for (scan_id, metric_name), value in scan_metrics.items():
            pred_vals[scan_id][metric_name] = value

        return pred_vals

    def log_summary(self):
        if not comm.is_main_process():
            return

        output_dir = self._output_dir
        self._logger.info(
            "[{}] Slice metrics summary:\n{}".format(
                type(self).__name__, self.slice_metrics.summary()
            )
        )
        # TODO: Make this based off if metrics has scans
        if self._aggregate_scans:
            self._logger.info(
                "[{}] Scan metrics summary:\n{}".format(
                    type(self).__name__, self.scan_metrics.summary()
                )
            )

        if not output_dir:
            return

        dirpath = output_dir
        os.makedirs(dirpath, exist_ok=True)
        test_results_summary_path = os.path.join(dirpath, "results.txt")
        slice_metrics_path = os.path.join(dirpath, "slice_metrics.csv")
        scan_metrics_path = os.path.join(dirpath, "scan_metrics.csv")

        # Write details to test file
        with open(test_results_summary_path, "w+") as f:
            f.write("Results generated on %s\n" % time.strftime("%X %x %Z"))
            # f.write("Weights Loaded: %s\n" % os.path.basename(self._config.TEST_WEIGHT_PATH))

            f.write("--" * 40)
            f.write("\n")
            f.write("Slice Metrics:\n")
            f.write(self.slice_metrics.summary())
            f.write("--" * 40)
            f.write("\n")
            if self._aggregate_scans:
                f.write("Scan Metrics:\n")
                f.write(self.scan_metrics.summary())
            f.write("--" * 40)
            f.write("\n")

        df: pd.DataFrame = self.slice_metrics.to_pandas()
        df.to_csv(slice_metrics_path, header=True, index=True)

        df: pd.DataFrame = self.scan_metrics.to_pandas()
        df.to_csv(scan_metrics_path, header=True, index=True)

    def evaluate_prediction(
        self,
        prediction,
        metrics: MetricCollection,
        ex_id: Union[str, Sequence[str]],
        is_batch=False,
    ):
        output, target = prediction["pred"], prediction["target"]
        if not is_batch:
            output, target = output.unsqueeze(0), target.unsqueeze(0)
            ex_id = [ex_id]
        output, target = cplx.channel_first(output), cplx.channel_first(target)
        metrics(preds=output, targets=target, ids=ex_id)
        # TODO (arjundd): Add support for multiple metrics
        # Hacky way to return an empty dict when metrics are not supported.
        # TODO (arjundd): Handle metric-less evaluation appropriately
        try:
            return metrics.to_dict()
        except ValueError:  # pragma: no cover
            return {}

    def _append_memory(self, key):
        if not torch.cuda.is_available():
            return
        self._memory[key].append(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        mem = self._memory[key]
        if len(mem) > 1 and (mem[-1] - mem[-2] > 500):
            self._logger.info(f"Memory exceeded '{key}'- previous 5 logs: {mem[-5:]}")
            # self._logger.info(torch.cuda.memory_stats())
