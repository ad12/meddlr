import copy
import logging
import os
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from ss_recon.data.transforms.transform import build_normalizer
from ss_recon.evaluation.metrics import (
    compute_l2,
    compute_nrmse,
    compute_psnr,
    compute_ssim,
    compute_vifp_mscale,
)
from ss_recon.ops import complex as cplx
from ss_recon.utils.transforms import center_crop

from .evaluator import DatasetEvaluator


class ReconEvaluator(DatasetEvaluator):
    """
    Evaluate reconstruction quality using the metrics listed below:

    - reconstruction loss (as specified by `loss_computer`)
    - L1, L2
    - Magnitude PSNR
    - Complex PSNR
    - SSIM (to be implemented)
    """

    def __init__(
        self,
        dataset_name,
        cfg,
        output_dir=None,
        group_by_scan=False,
        skip_rescale=False,
        save_scans=False,
        metrics=None,
        flush_period: int = None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            output_dir (str, optional): an output directory to dump all
                results predicted on the dataset. Currently not used.
            group_by_scan (bool, optional): If `True`, groups metrics by scan.
            skip_rescale (bool, optional): If `True`, skips rescaling the output and target
                by the mean/std.
            save_scans (bool, optional): If `True`, saves predictions to .npy file.
            metrics (Sequence[str], optional): To avoid computing metrics, set to ``False``.
                Defaults to all supported recon metrics.
                To process metrics on the full scan, append ``'_scan'`` to the metric name
                (e.g. `'psnr_scan'`). Supported metrics include:
                * 'psnr': Complex peak signal-to-noise ratio
                * 'psnr_mag': Magnitude peak signal-to-noise ratio
                * 'ssim_old': Old calculation for SSIM
                * 'ssim (Wang)': SSIM following Wang, et al. protocol.
                    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
                * 'nrmse': Complex normalized root-mean-squared-error
                * 'nrmse_mag': Magnitude normalized root-mean-squared-error.
                * 'vif_mag': Visual information fidelity on magnitude images.
                * 'vif_phase': Visual information fidelity on phase images.
            flush_period (int, optional): The approximate period over which predictions
                are cleared and running results are computed. This parameter helps
                mitigate OOM errors. The period is equivalent to number of examples
                (not batches).
        """
        # self._tasks = self._tasks_from_config(cfg)
        self._output_dir = output_dir
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = build_normalizer(cfg)
        self._group_by_scan = group_by_scan
        self._skip_rescale = skip_rescale
        self._save_scans = save_scans

        if metrics is not False:
            self._slice_metrics = (
                [m for m in metrics if not m.endswith("_scan")] if metrics else None
            )
            self._scan_metrics = (
                [m[:-5] for m in metrics if m.endswith("_scan")] if metrics else None
            )
        else:
            self._slice_metrics = []
            self._scan_metrics = []
        self._results = None
        self._running_results = None

        if flush_period is None:
            flush_period = cfg.TEST.FLUSH_PERIOD
        if flush_period is None or flush_period < 0:
            flush_period = 0
        self.flush_period = flush_period

        # TODO: Uncomment when metadata is supported
        # self._metadata = MetadataCatalog.get(dataset_name)
        # if not hasattr(self._metadata, "json_file"):
        #     self._logger.warning(
        #         f"json_file was not found in MetaDataCatalog for '{dataset_name}'")  # noqa
        #
        #     cache_path = os.path.join(output_dir,
        #                               f"{dataset_name}_coco_format.json")
        #     self._metadata.json_file = cache_path
        #     convert_to_coco_json(dataset_name, cache_path)

    @classmethod
    def default_metrics(cls):
        """The default metrics processed by this class."""
        metrics = ["psnr", "psnr_mag", "ssim_old", "ssim (Wang)", "nrmse", "nrmse_mag"]
        metrics.extend([f"{x}_scan" for x in metrics])
        return metrics

    def reset(self):
        self._predictions = []
        self._scan_map = defaultdict(dict)
        self.scans = None
        self._results = None
        self._running_results = None

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

        if self._skip_rescale:
            # Do not rescale the outputs
            preds = outputs["pred"].to(self._cpu_device, non_blocking=True)
            targets = outputs["target"].to(self._cpu_device, non_blocking=True)
        else:
            normalized = self._normalizer.undo(
                image=outputs["pred"],
                target=outputs["target"],
                mean=inputs["mean"],
                std=inputs["std"],
            )
            preds = normalized["image"].to(self._cpu_device, non_blocking=True)
            targets = normalized["target"].to(self._cpu_device, non_blocking=True)

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

        if self.flush_period > 0 and len(self._predictions) >= self.flush_period:
            self.flush(skip_last_scan=True)

        # preds = outputs["pred"].to(self._cpu_device)
        # targets = outputs["target"].to(self._cpu_device)
        # means = inputs["mean"].to(self._cpu_device)
        # stds = inputs["std"].to(self._cpu_device)
        # for i in range(N):
        #     pred, target = preds[i], targets[i]
        #     mean = means[i]
        #     std = stds[i]
        #     pred = pred * std + mean
        #     target = target * std + mean

        #     # probably isn't best practice to hang onto each prediction.
        #     prediction = {"pred": pred, "target": target}

        #     self._predictions.append(prediction)

    def flush(self, skip_last_scan: bool = True):
        """Clear predictions and computing running metrics.

        Results are added to ``self._running_results``.

        Args:
            skip_last_scan (bool, optional): If ``True``, does not flush
                most recent scan. This avoids prematurely computing metrics
                before all slices of the scan are available.
        """
        remaining_preds = []

        if skip_last_scan:
            try:
                scan_ids = np.asarray([p["metadata"]["scan_id"] for p in self._predictions])
            except KeyError:
                raise ValueError(
                    "Cannot skip last scan. metadata does not contain 'scan_id' keyword."
                )

            change_idxs = np.where(scan_ids[1:] != scan_ids[:-1])[0]
            if len(change_idxs) == 0:
                warnings.warn(
                    "Flushing skipped. All current predictions are from the same scan. "
                    "To force flush, set `skip_last_scan=True`."
                )
                return

            last_idx = int(change_idxs[-1] + 1)
            remaining_preds = self._predictions[last_idx:]
            self._predictions = self._predictions[:last_idx]

        self._logger.info("Flushing results...")

        self.evaluate()
        self._predictions = remaining_preds

    def structure_scans(self):
        """Structure scans into volumes to be used to evaluation."""
        self._logger.info("Structuring slices into volumes...")
        scan_map = defaultdict(dict)
        for pred in self._predictions:
            scan_map[pred["metadata"]["scan_id"]][pred["metadata"]["slice_id"]] = pred

        scans = {}
        for scan_id, slice_idx_to_pred in tqdm(scan_map.items()):
            min_slice, max_slice = min(slice_idx_to_pred.keys()), max(slice_idx_to_pred.keys())
            slice_predictions = [slice_idx_to_pred[i] for i in range(min_slice, max_slice + 1)]
            pred = {
                k: torch.stack(
                    [slice_pred[k] for slice_pred in slice_predictions], dim=0
                ).contiguous()
                for k in ("pred", "target")
            }
            scans[scan_id] = pred
        return scans

    def evaluate(self):
        if self._group_by_scan:
            return self._evaluate_by_group()

        if len(self._predictions) == 0:
            self._logger.warning("[ReconEvaluator] Did not receive valid predictions.")
            return {}

        pred_vals = defaultdict(list)
        for pred in tqdm(self._predictions, desc="Slice metrics"):
            val = self.evaluate_prediction(pred, self._slice_metrics)
            for k, v in val.items():
                pred_vals[f"val_{k}"].append(v)

        scans = self.structure_scans()
        self.scans = scans
        scans = scans.values()
        for pred in tqdm(scans, desc="Scan Metrics"):
            val = self.evaluate_prediction(pred, self._scan_metrics)
            for k, v in val.items():
                pred_vals[f"val_{k}_scan"].append(v)

        if self._running_results is None:
            self._running_results = defaultdict(list)
        for k, v in pred_vals.items():
            self._running_results[k].extend(v)

        self._results = OrderedDict({k: np.mean(v) for k, v in self._running_results.items()})

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _evaluate_by_group(self):
        """Keeping this separate for now to avoid breaking existing functionality."""
        if len(self._predictions) == 0:
            self._logger.warning("[ReconEvaluator] Did not receive valid predictions.")
            return {}

        # Per slice evaluation.
        pred_vals = defaultdict(lambda: defaultdict(list))
        for pred in tqdm(self._predictions, desc="Slice metrics"):
            scan_id = pred["metadata"]["scan_id"]
            val = self.evaluate_prediction(pred, self._slice_metrics)
            for k, v in val.items():
                pred_vals[scan_id][f"{k}"].append(v)

        # Full scan evaluation
        if self._scan_metrics or self._save_scans:
            scans = self.structure_scans()
        else:
            scans = {}
        for scan_id, pred in tqdm(scans.items(), desc="Scan metrics"):
            val = self.evaluate_prediction(pred, self._scan_metrics)
            for k, v in val.items():
                pred_vals[scan_id][f"{k}_scan"].append(v)

        if self._save_scans:
            self._logger.info("Saving data...")
            assert self._output_dir
            for scan_id, pred in tqdm(scans.items()):
                np.save(os.path.join(self._output_dir, f"{scan_id}.npy"), pred["pred"])

        if self._running_results is None:
            self._running_results = defaultdict(lambda: defaultdict(list))
        for scan_id, metrics in pred_vals.items():
            for k, v in metrics.items():
                self._running_results[scan_id][k].extend(v)

        self._results = OrderedDict(
            {
                scan_id: {k: np.mean(v) for k, v in metrics.items()}
                for scan_id, metrics in self._running_results.items()
            }
        )

        results = copy.deepcopy(self._results)

        # if self._output_dir:
        #     output_file = os.path.join(self._output_dir, "results.pt")
        #     _results = {"results": results, "pred_vals": pred_vals}
        #     import pdb; pdb.set_trace()
        #     torch.save(_results, output_file)

        # Copy so the caller can do whatever with results
        return results

    def evaluate_prediction(self, prediction, metric_names=None):
        output, target = prediction["pred"], prediction["target"]
        metric_names = list(metric_names) if metric_names is not None else None
        metrics = {}

        # Compute metrics magnitude images
        abs_error = cplx.abs(output - target)
        metrics["l1"] = torch.mean(abs_error).item()
        metrics["l2"] = compute_l2(target, output).item()
        if metric_names is None or "psnr" in metric_names:
            metrics["psnr"] = compute_psnr(target, output).item()
        if metric_names is None or "psnr_mag" in metric_names:
            metrics["psnr_mag"] = compute_psnr(target, output, magnitude=True).item()
        if metric_names is None or "ssim_old" in metric_names:
            metrics["ssim_old"] = compute_ssim(
                target,
                output,
                data_range="x-range",
                gaussian_weights=False,
                use_sample_covariance=True,
            )

        # Compute ssim following Wang, et al. protocol.
        # https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
        # Both the target and predicted reconstructions are
        # normalized by the maximum value of the magnitude of the target
        if metric_names is None or "ssim (Wang)" in metric_names:
            metrics["ssim (Wang)"] = compute_ssim(
                target,
                output,
                data_range="ref-maxval",
                gaussian_weights=True,
                use_sample_covariance=False,
            )
        if metric_names is None or "ssim50 (Wang)" in metric_names:
            shape = target.shape[:-1] if cplx.is_complex_as_real(target) else target.shape
            shape = tuple(x // 2 if x > 1 else 1 for x in shape)
            metrics["ssim50 (Wang)"] = compute_ssim(
                center_crop(target, shape),
                center_crop(output, shape),
                data_range="ref-maxval",
                gaussian_weights=True,
                use_sample_covariance=False,
            )
        if metric_names is None or "nrmse" in metric_names:
            metrics["nrmse"] = compute_nrmse(target, output).item()
        if metric_names is None or "nrmse_mag" in metric_names:
            metrics["nrmse_mag"] = compute_nrmse(target, output, magnitude=True).item()

        if metric_names is None or "vif_mag" in metric_names:
            metrics["vif_mag"] = compute_vifp_mscale(target, output, im_type="magnitude")
        if metric_names is None or "vif_phase" in metric_names:
            metrics["vif_phase"] = compute_vifp_mscale(target, output, im_type="phase")

        # Make sure all metrics are handled.
        if metric_names is None:
            metric_names = ()
        remaining = set(metric_names) - set(metrics.keys())
        if len(remaining) > 0:
            raise ValueError(f"Cannot handle metrics {remaining}")

        return metrics
        # return {
        #     "l1": l1, "l2": l2, "psnr": psnr,
        #     "ssim_old": ssim_old, "ssim (Wang)": ssim_wang,
        #     "nrmse": nrmse, "psnr_mag": psnr_mag, "nrmse_mag": nrmse_mag
        # }

    def evaluate_prediction_old(self, prediction):
        warnings.warn(
            "`evaluate_prediction_old` is deprecated and is staged for removal in v0.0.2",
            DeprecationWarning,
        )
        output, target = prediction["pred"], prediction["target"]

        # Compute metrics magnitude images
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error).item()
        l2 = torch.sqrt(torch.mean(abs_error ** 2)).item()
        psnr = 20 * torch.log10(cplx.abs(target).max() / l2).item()

        output, target = cplx.abs(output), cplx.abs(target)
        target = target.squeeze(-1).numpy()
        output = output.squeeze(-1).numpy()
        ssim = structural_similarity(target, output, data_range=output.max() - output.min())

        return {"l1": l1, "l2": l2, "psnr": psnr, "ssim": ssim}
