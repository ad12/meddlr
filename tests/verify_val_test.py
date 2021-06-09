"""Verify that validation works like it is supposed to.'

There was an issue where the performance at validation time
was different than at test time even though the same evaluator
was being used.

This script is to be used to debug through this.

Note:
    Some things to keep track of:
        - gradient accumulation flushing before/after checkpointing
        - is model after eval the same as model from checkpoint

Findings:
    1. In a single run, performance across same dataset is the same (EXECPTED)
    2. Running testing multiple times with the same weights file has same performance (EXPECTED)
    3. Val / Test performance and models are the same for the last iteration
        w/ cudnn.deterministic=True (EXPECTED)

To verify:
    1. Val / Test performance and models are the same for the last iteration w/ cudnn.deterministic=False
    2. Val / Test performance and models are the same for intermediate checkpoints (iteration 40/80)
    3. Val / Test performance and models are the same for the last iteration w/ gradient accumulation
"""
import os
import sys
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy

import torch
from fvcore.common.file_io import PathManager
from tqdm import tqdm

import ss_recon.utils.complex_utils as cplx
from ss_recon.checkpoint.detection_checkpoint import DetectionCheckpointer
from ss_recon.config import get_cfg
from ss_recon.engine import DefaultTrainer, default_setup
from ss_recon.evaluation.recon_evaluation import ReconEvaluator
from ss_recon.evaluation.testing import find_weights
from ss_recon.utils.logger import setup_logger

# Set seed and cuda deterministic to true to be able to reproduce.
SEED = 0
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class ComparisonEvaluator(ReconEvaluator):
    def __init__(self, dataset_name, cfg, output_dir, group_by_scan=False):
        super().__init__(dataset_name, cfg, output_dir=output_dir, group_by_scan=group_by_scan)

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
                k: torch.stack([slice_pred[k] for slice_pred in slice_predictions], dim=0)
                for k in ("pred", "target", "mask", "kspace")
            }
            scans[scan_id] = pred
        return scans

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

        self._predictions.extend(
            [
                {
                    "kspace": inputs["kspace"][i],
                    "mask": cplx.get_mask(inputs["kspace"][i]),
                    "pred": preds[i],
                    "target": targets[i],
                    "metadata": inputs["metadata"][i] if "metadata" in inputs else {},
                }
                for i in range(N)
            ]
        )

    def evaluate(self):
        results = super().evaluate()

        # Save predictions and volumes.
        torch.save(self._predictions, os.path.join(self._output_dir, "2d_predictions.pt"))
        torch.save(self.scans, os.path.join(self._output_dir, "scan_predictions.pt"))

        return results


class ComparisonTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        """
        Returns:
            DatasetEvaluator

        It is not implemented by default.
        """
        if not output_dir:
            output_dir = cfg.OUTPUT_DIR
        return ComparisonEvaluator(dataset_name, cfg, output_dir)


def train_and_val():
    """This does training and validation."""
    # train_args = deepcopy(args)
    default_setup(cfg, args, save_cfg=True)

    # Set again for completeness.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    trainer = ComparisonTrainer(cfg)
    return trainer.train(), trainer.model.cpu()


def _test(run_setup=False, weights=""):
    """This does testing."""
    eval_args = deepcopy(args)
    eval_args.eval_only = True
    if run_setup:
        default_setup(cfg, args, save_cfg=False)

    logger = setup_logger(name=__name__)
    model = ComparisonTrainer.build_model(cfg)
    init_model = ComparisonTrainer.build_model(cfg)

    # model = model.to(cfg.DEVICE)

    # Get metrics for the last checkpoint.
    if not weights:
        weights = find_weights(cfg, criterion="iteration", iter_limit=None)
    # data = torch.load(weights)

    model = model.to(cfg.DEVICE)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(weights, resume=False)
    # For future comparision - deep copy does not work
    # Model must also be on the same device as the params
    # See https://github.com/pytorch/pytorch/issues/42300
    init_model = init_model.cuda()
    DetectionCheckpointer(init_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(weights, resume=False)

    # Compare models at the beginning.
    # If these are not the same, there is a pytorch bug.
    if compare_models(model, init_model):
        logger.info("test: Loaded model and copied version are the same.")
    else:
        sys.exit(1)

    # This is exactly the call that is made by validation.
    # So it gives us a benchmark to compare against.
    output_dir = os.path.join(cfg.OUTPUT_DIR, "testing")
    os.makedirs(output_dir, exist_ok=True)
    evaluator = ComparisonEvaluator(cfg.DATASETS.VAL[0], cfg, output_dir)
    results = ComparisonTrainer.test(cfg, model, evaluators=evaluator, use_val=True)
    # results = None
    # Check that model that was loaded and model after eval are the same
    assert compare_models(init_model, model)  # model changed during eval

    return results, model.cpu()


def compare_models(model_1, model_2):
    """From https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5"""
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
                print(f"{key_item_1[1]} vs {key_item_2[1]}")
                return False
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")
        return True
    else:
        return False


def compare_outputs():
    """Compare scan inputs (kspace, masks, target) and outputs (pred) between val / test."""
    val_scans = torch.load(os.path.join(cfg.OUTPUT_DIR, "scan_predictions.pt"))
    test_scans = torch.load(os.path.join(cfg.OUTPUT_DIR, "testing/scan_predictions.pt"))

    scan_ids = val_scans.keys()

    for scan_id in scan_ids:
        val = val_scans[scan_id]
        test = test_scans[scan_id]
        assert torch.equal(val["mask"], test["mask"])
        assert torch.equal(val["kspace"], test["kspace"])
        assert torch.equal(val["target"], test["target"])
        assert torch.equal(val["pred"], test["pred"])


# # train_results, trained_model = train_and_val()
# # torch.cuda.empty_cache()
# test_results, test_model = test(True)

# # Compare models
# # compare_models(trained_model, test_model)

# # print(train_results)
# print(test_results)

# # for k in train_results:
# #     tr_val = train_results[k]
# #     tst_val = test_results[k]
# #     assert tr_val == tst_val, "{} - train: {}, test:{}".format(tr_val, tst_val)

# compare_outputs()
# print("All outputs between val and test are the same")

if __name__ == "__main__":
    # Initialize config
    cfg = get_cfg()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "GeneralizedUnrolledCNN"
    cfg.OUTPUT_DIR = PathManager.get_local_path("results://tests/verify_val")
    cfg.DATASETS.TRAIN = ("mridata_knee_2019_train",)
    cfg.DATASETS.VAL = ("mridata_knee_2019_val",)
    cfg.DATASETS.TEST = ()
    cfg.SOLVER.MAX_ITER = 80
    cfg.SOLVER.CHECKPOINT_PERIOD = 80
    cfg.TEST.EVAL_PERIOD = 80
    cfg.TIME_SCALE = "iter"
    cfg.SOLVER.TRAIN_BATCH_SIZE = 1
    cfg.SOLVER.TEST_BATCH_SIZE = 8
    cfg.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 12  # ensure deterministic
    cfg.SEED = 0  # ensure deterministic
    cfg.freeze()

    # Initialize args namespace.
    # Serve as corollary for ss_recon.engine.defaults.default_argument_parser
    args = Namespace(
        debug=True,
        resume=False,
        num_gpus=1,
        devices=None,
        eval_only=False,  # for eval, set this to true.
    )
    test_results, test_model = _test(
        True,
        weights="/bmrNAS/people/arjun/results/ss_recon/prelim_exps/baseline_12x_maxbatch/sub-1/model_0151999.pth",  # noqa: B950
    )
    print(test_results)
