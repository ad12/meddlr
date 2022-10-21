import copy
import itertools
import logging
import os
import time
import uuid

import torch
from tqdm import tqdm

import meddlr.ops as oF
import meddlr.utils.comm as comm
from meddlr.data.transforms.transform import build_normalizer
from meddlr.evaluation.scan_evaluator import ScanEvaluator, structure_scans
from meddlr.metrics.build import build_metrics

_DEFAULT_METRICS = (
    "DSC",
    "VOE",
    "CV",
    "DSC_scan",
    "VOE_scan",
    "CV_scan",
    "ASSD_scan",
)


class SemSegEvaluator(ScanEvaluator):
    """
    Evaluate semantic segmentation quality.
    """

    def __init__(
        self,
        dataset_name,
        cfg,
        distributed=False,
        sync_outputs=False,
        aggregate_scans=True,
        output_dir=None,
        group_by_scan: bool = False,
        save_seg: bool = False,
        metrics=None,
        flush_period: bool = None,
        to_cpu=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            distributed (bool, optional): If `True`, collect results from all
                ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
                If using `DistributedDataParallel`, this should likely be `True`.
            sync_outputs (bool, optional): If `True`, synchronizes all predictions
                before evaluation.
                If `False`, synchronizes metrics before reduction.
                Ignored if `distributed=False`
            aggregate_scans (bool, optional): If `True`, also computes metrics per
                scan under the label `'scan_{metric}'` (e.g. scan_l1).

        Not yet supported:
            running_eval (bool, optional): If `True`, evaluates scans while processing.
                This reduces the memory overhead. Note if distributed=`True`, sync_outputs
                must be `False`.
        """
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = build_normalizer(cfg)
        self._distributed = distributed
        self._sync_outputs = sync_outputs
        self._aggregate_scans = aggregate_scans
        self._group_by_scan = group_by_scan
        self._save_seg = save_seg
        self.to_cpu = to_cpu
        self._is_flushing = False

        self.activation = cfg.MODEL.SEG.ACTIVATION
        self.threshold = 0.5
        # if self._distributed and self._sync_outputs and running_eval:
        #     raise ValueError("running_eval not possible when outputs must be synced")

        self.has_bg = cfg.MODEL.SEG.INCLUDE_BACKGROUND or cfg.MODEL.SEG.ACTIVATION == "softmax"
        if self.has_bg:
            # Add "background" to class names if desired.
            self._class_names = (
                cfg.MODEL.SEG.CLASSES
                if any(x.lower() in ["background", "bg"] for x in cfg.MODEL.SEG.CLASSES)
                else ("bg",) + tuple(cfg.MODEL.SEG.CLASSES)
            )
        else:
            self._class_names = cfg.MODEL.SEG.CLASSES

        self._metric_names = []
        if metrics is not False:
            if not metrics:
                metrics = self.default_metrics()
            self._metric_names = metrics

        if flush_period is None:
            flush_period = cfg.TEST.FLUSH_PERIOD
        self.flush_period = flush_period

        self._remaining_state = None
        self._predictions = []
        self._is_flushing = False

    @classmethod
    def default_metrics(cls):
        """The default metrics processed by this class."""
        # TODO: Add support for ASSD_scan
        return _DEFAULT_METRICS

    def reset(self):
        self._predictions = []
        self._remaining_state = None
        self._is_flushing = False

        metrics = self._metric_names
        slice_metrics = [m for m in metrics if not m.endswith("_scan")]
        scan_metrics = [m[:-5] for m in metrics if m.endswith("_scan")]

        self.slice_metrics = build_metrics(slice_metrics, channel_names=self._class_names)
        self.scan_metrics = build_metrics(
            scan_metrics, fmt="{}_scan", channel_names=self._class_names
        )
        self.slice_metrics.eval()
        self.scan_metrics.eval()

    def structure_scans(self, verbose=True):
        """Structure scans into volumes to be used to evaluation."""
        out = structure_scans(self._predictions, verbose=verbose, dims={1: "slice_id"})
        return out

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a recon model (e.g., GeneralizedRCNN).
                Currently this should be an empty dictionary.
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        labels: torch.Tensor = self._parse_sem_seg_label(inputs)
        pred: torch.Tensor = self._parse_sem_seg_pred(outputs)

        N = pred.shape[0]
        preds = pred.type(torch.bool)
        targets = labels.type(torch.bool)
        if self.to_cpu:
            preds = preds.to(self._cpu_device, non_blocking=True)
            targets = targets.to(self._cpu_device, non_blocking=True)

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

        has_num_examples = self.flush_period > 0 and len(self._predictions) >= self.flush_period
        has_num_scans = self.flush_period < 0 and len(
            {x["metadata"]["scan_id"] for x in self._predictions}
        ) > abs(self.flush_period)
        if has_num_examples or has_num_scans:
            self.flush(enter_prediction_scope=True, skip_last_scan=True)

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
            self._logger.warning(f"[{type(self).__name__}] Did not receive valid predictions.")
            return {}

        # TODO: Add support in metrics manager for appending metrics to the same scan_id
        for _idx, pred in enumerate(tqdm(self._predictions, desc="Slice Metrics")):
            if "metadata" in pred:
                scan_id = "/".join(
                    str(pred["metadata"].get(k, uuid.uuid4())) for k in ["scan_id", "slice_id"]
                )
                voxel_spacing = pred["metadata"].get("voxel_spacing", None)
                if voxel_spacing:
                    voxel_spacing = voxel_spacing[-(pred["pred"].ndim - 1) :]
            else:
                scan_id = str(uuid.uuid4())
                voxel_spacing = None
            self.slice_metrics(
                ids=[scan_id],
                preds=pred["pred"].unsqueeze(0),
                targets=pred["target"].unsqueeze(0),
                voxel_spacing=voxel_spacing,
                spacing=voxel_spacing,
            )

        # Compute metrics per scan.
        if self._aggregate_scans:
            scans = self.structure_scans()
            for scan_id, pred in tqdm(scans.items(), desc="Scan Metrics"):
                self.scan_metrics(
                    ids=[scan_id],
                    preds=pred["pred"].unsqueeze(0),
                    targets=pred["target"].unsqueeze(0),
                    category_dim=0,
                    voxel_spacing=pred.get("voxel_spacing", None),
                    spacing=pred.get("voxel_spacing", None),
                )

        pred_vals = self.slice_metrics.to_dict()
        pred_vals.update(self.scan_metrics.to_dict())
        self._results = pred_vals

        if not self._is_flushing:
            self.log_summary()

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def log_summary(self):
        slice_metrics_summary = self.slice_metrics.summary()
        scan_metrics_summary = self.scan_metrics.summary()
        slice_metrics_df = self.slice_metrics.to_pandas()
        scan_metrics_df = self.scan_metrics.to_pandas()

        if not comm.is_main_process():
            return

        output_dir = self._output_dir
        self._logger.info(
            "[{}] Slice metrics summary:\n{}".format(type(self).__name__, slice_metrics_summary)
        )
        self._logger.info(
            "[{}] Scan metrics summary:\n{}".format(type(self).__name__, scan_metrics_summary)
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
            f.write(slice_metrics_summary)
            f.write("--" * 40)
            f.write("\n")
            f.write("Scan Metrics:\n")
            f.write(scan_metrics_summary)
            f.write("--" * 40)
            f.write("\n")

        slice_metrics_df.to_csv(slice_metrics_path, header=True, index=True)
        scan_metrics_df.to_csv(scan_metrics_path, header=True, index=True)

    def _parse_sem_seg_pred(self, input):
        # TODO: Update these values
        num_classes = 1
        channel_dim = 1

        keys = ["sem_seg_pred", "sem_seg_probs", "sem_seg_logits", "pred", "probs", "logits"]
        existing_keys = [k for k in keys if k in input.keys()]
        if len(existing_keys) == 0:
            raise ValueError(f"`input` must have one of keys: {keys}.\n\tGot: {list(input.keys())}")
        key = existing_keys[0]
        assert input[key].shape[1] == len(self._class_names), input[key].shape

        if key.endswith("pred"):
            pred = input[key]
        elif key.endswith("logits"):
            logits = input[key]
            if self.activation == "sigmoid":
                probs = torch.sigmoid(logits)
                pred = (probs >= self.threshold).type(torch.long)
            elif self.activation == "softmax":
                pred = oF.pred_to_categorical(
                    logits, activation=self.activation, channel_dim=1, threshold=0.5
                )
                pred = oF.to_onehot(pred, len(num_classes) + 1)
        elif key.endswith("probs"):
            probs = input[key]
            if self.activation == "sigmoid":
                pred = probs >= self.threshold
            elif self.activation == "softmax":
                pred = oF.to_onehot(
                    torch.argmax(probs, dim=channel_dim),
                    num_classes=len(self._class_names) + 1,
                )

        return pred.type(torch.uint8)

    def _parse_sem_seg_label(self, output):
        keys = ["sem_seg", "sem_seg_labels", "sem_seg_target", "labels", "target"]
        existing_keys = [k for k in keys if k in output.keys()]
        if len(existing_keys) == 0:
            raise ValueError(
                f"`output` must have one of keys: {keys}.\n\tGot: {list(output.keys())}"
            )
        key = existing_keys[0]
        assert output[key].shape[1] == len(self._class_names), f"Key '{key}': {output[key].shape}"
        return output[key].type(torch.uint8)
