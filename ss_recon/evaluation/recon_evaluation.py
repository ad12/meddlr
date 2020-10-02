import copy
import logging
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity

from ss_recon.data.transforms.transform import build_normalizer
from ss_recon.evaluation.metrics import compute_mse, compute_l2, compute_psnr, compute_nrmse, compute_ssim
from ss_recon.utils import complex_utils as cplx

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

    def __init__(self, dataset_name, cfg, output_dir=None, group_by_scan=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            output_dir (str, optional): an output directory to dump all
                results predicted on the dataset. Currently not used.
            group_by_scan (bool, optional): If `True`, groups metrics by scan.
        """
        # self._tasks = self._tasks_from_config(cfg)
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = build_normalizer(cfg)
        self._group_by_scan = group_by_scan

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

    def reset(self):
        self._predictions = []
        self._scan_map = defaultdict(dict)
        self.scans = None

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given
                configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a recon model (e.g., GeneralizedRCNN).
                Currently this should be an empty dictionary.
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        N = outputs["pred"].shape[0]

        normalized = self._normalizer.undo(
            image=outputs["pred"], target=outputs["target"], mean=inputs["mean"], std=inputs["std"]
        )
        preds = normalized["image"].to(self._cpu_device, non_blocking=True)
        targets = normalized["target"].to(self._cpu_device, non_blocking=True)

        self._predictions.extend([
            {
                "pred": preds[i], 
                "target": targets[i], 
                "metadata": inputs["metadata"][i] if "metadata" in inputs else {}
            } 
            for i in range(N)
        ])

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

    def structure_scans(self):
        """Structure scans into volumes to be used to evaluation."""
        self._logger.info("Structuring slices into volumes...")
        scan_map = defaultdict(dict)
        for pred in self._predictions:
            scan_map[pred["metadata"]["scan_id"]][pred["metadata"]["slice_id"]] = pred

        scans = {}
        for scan_id, slice_idx_to_pred in tqdm(scan_map.items()):
            min_slice, max_slice = min(slice_idx_to_pred.keys()), max(slice_idx_to_pred.keys())
            slice_predictions = [slice_idx_to_pred[i] for i in range(min_slice, max_slice+1)]
            pred = {
                k: torch.stack([slice_pred[k] for slice_pred in slice_predictions], dim=0) 
                for k in ("pred", "target")
            }
            scans[scan_id] = pred
        return scans

    def evaluate(self):
        if self._group_by_scan:
            return self._evaluate_by_group()

        if len(self._predictions) == 0:
            self._logger.warning(
                "[ReconEvaluator] Did not receive valid predictions."
            )
            return {}

        pred_vals = defaultdict(list)
        for pred in self._predictions:
            val = self.evaluate_prediction(pred)
            for k, v in val.items():
                pred_vals[f"val_{k}"].append(v)

        scans = self.structure_scans()
        self.scans = scans
        scans = scans.values()
        for pred in scans:
            val = self.evaluate_prediction(pred)
            for k, v in val.items():
                pred_vals[f"val_{k}_scan"].append(v)

        self._results = OrderedDict(
            {k: np.mean(v) for k, v in pred_vals.items()}
        )
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    
    def _evaluate_by_group(self):
        """Keeping this separate for now to avoid breaking existing functionality."""
        if len(self._predictions) == 0:
            self._logger.warning(
                "[ReconEvaluator] Did not receive valid predictions."
            )
            return {}

        # Per slice evaluation.
        pred_vals = defaultdict(lambda: defaultdict(list))
        for pred in self._predictions:
            scan_id = pred["metadata"]["scan_id"]
            val = self.evaluate_prediction(pred)
            for k, v in val.items():
                pred_vals[scan_id][f"{k}"].append(v)

        # Full scan evaluation
        scans = self.structure_scans()
        for scan_id, pred in scans.items():
            val = self.evaluate_prediction(pred)
            for k, v in val.items():
                pred_vals[scan_id][f"{k}_scan"].append(v)

        self._results = OrderedDict({
            scan_id: {k: np.mean(v) for k, v in metrics.items()} 
            for scan_id, metrics in pred_vals.items()
        })

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def evaluate_prediction(self, prediction):
        output, target = prediction["pred"], prediction["target"]

        # Compute metrics magnitude images
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error).item()
        l2 = compute_l2(target, output).item()
        psnr = compute_psnr(target, output).item()
        ssim_old = compute_ssim(
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
        ssim_wang = compute_ssim(
            target,
            output,
            data_range="ref-maxval",
            gaussian_weights=True,
            use_sample_covariance=False,
        )
        nrmse = compute_nrmse(target, output).item()

        return {
            "l1": l1, "l2": l2, "psnr": psnr, 
            "ssim_old": ssim_old, "ssim (Wang)": ssim_wang, 
            "nrmse": nrmse,
        }

    def evaluate_prediction_old(self, prediction):
        warnings.warn(
            "`evaluate_prediction_old` is deprecated and is staged for removal in v0.0.2", 
            DeprecationWarning
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
        ssim = structural_similarity(
            target, output, data_range=output.max() - output.min()
        )

        return {"l1": l1, "l2": l2, "psnr": psnr, "ssim": ssim}
