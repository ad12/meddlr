import copy
import logging
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from skimage.metrics import structural_similarity

from ss_recon.data.transforms.transform import build_normalizer
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

    def __init__(self, dataset_name, cfg, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        # self._tasks = self._tasks_from_config(cfg)
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = build_normalizer(cfg)

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
        self._coco_results = []

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
            outputs["pred"], outputs["target"], mean=inputs["mean"], std=inputs["std"]
        )
        preds = normalized["image"].to(self._cpu_device, non_blocking=True)
        targets = normalized["target"].to(self._cpu_device, non_blocking=True)

        self._predictions.extend([{"pred": preds[i], "target": targets[i]} for i in range(N)])

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

    def evaluate(self):
        if len(self._predictions) == 0:
            self._logger.warning(
                "[ReconEvaluator] Did not receive valid predictions."
            )
            return {}

        pred_vals = defaultdict(list)
        for pred in self._predictions:
            val = self.evaluate_prediction(pred)
            for k, v in val.items():
                pred_vals[k].append(v)

        self._results = OrderedDict(
            {k: np.mean(v) for k, v in pred_vals.items()}
        )
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def evaluate_prediction(self, prediction):
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

        return {"val_l1": l1, "val_l2": l2, "val_psnr": psnr, "val_ssim": ssim}
