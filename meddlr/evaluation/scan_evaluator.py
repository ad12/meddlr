import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from meddlr.data.data_utils import structure_patches
from meddlr.evaluation.evaluator import DatasetEvaluator

_logger = logging.getLogger(__name__)


class ScanEvaluator(DatasetEvaluator):
    _remaining_state: Optional[Dict[str, Any]]
    _predictions: List[Dict[str, Any]]
    _is_flushing: bool
    _logger: logging.Logger

    def enter_prediction_scope(self, skip_last_scan: bool = True):
        if self._remaining_state is not None:
            raise ValueError(
                "You must exit the prediction scope by calling `exit_prediction_scope`."
            )

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
                self._logger.warning(
                    "Flushing skipped. All current predictions are from the same scan. "
                    "To force flush, set `skip_last_scan=False`."
                )
                return False

            last_idx = int(change_idxs[-1] + 1)
            remaining_preds = self._predictions[last_idx:]
            self._predictions = self._predictions[:last_idx]

        self._remaining_state = {"_predictions": remaining_preds}

    def exit_prediction_scope(self):
        if self._remaining_state is None:
            raise ValueError(
                "You must enter prediction scope first by calling `enter_prediction_scope`"
            )

        remaining_state = {k: v for k, v in self._remaining_state.items() if hasattr(self, k)}
        for k, v in remaining_state.items():
            setattr(self, k, v)
        self._remaining_state = None

    def flush(self, enter_prediction_scope: bool = True, skip_last_scan: bool = True):
        """Clear predictions and computing running metrics.

        Results are added to ``self._running_results``.

        Args:
            enter_prediction_scope (bool, optional): If ``True``, enter the
                prediction scope.
            skip_last_scan (bool, optional): If ``True``, does not flush
                most recent scan. This avoids prematurely computing metrics
                before all slices of the scan are available.
        """
        if enter_prediction_scope:
            success = self.enter_prediction_scope(skip_last_scan=skip_last_scan)
            # TODO: this means we haven't properly set the state, which is likely because
            # we dont want to execute this flush. Make the API clearer.
            if success is False:
                return

        self._logger.info("Flushing results...")

        self._is_flushing = True
        self.evaluate()
        self._is_flushing = False

        if enter_prediction_scope:
            self.exit_prediction_scope()

    def structure_scans(self, verbose=True):
        """Structure scans into volumes to be used to evaluation."""
        return structure_scans(self._predictions, verbose=verbose)


def structure_scans(
    predictions,
    to_struct=("pred", "target"),
    dims: Dict[int, str] = None,
    metadata=("voxel_spacing", "affine"),
    contiguous: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Structure patches into single objects.

    Note:
        This may not currently be a very fast implementation as stacking occurs multiple times
        instead of single tensor allocation.

    TODO: Benchmark speeds for future releases.
    """
    if not dims:
        dims = {0: "slice_id"}

    if not bool(predictions[0]["metadata"]):
        raise ValueError("Cannot aggregate scans without metadata")
    if len(set(to_struct) & set(metadata)) > 0:
        raise ValueError(
            "Cannot overlap between keys to structure (`to_struct`) "
            "and keys for metadata (`metadata`)."
        )

    if verbose:
        _logger.info("Structuring slices into volumes...")

    scan_map = defaultdict(dict)
    for pred in predictions:
        values = tuple(pred["metadata"][k] for k in dims.values())
        scan_map[pred["metadata"]["scan_id"]][values] = pred

    if verbose:
        _logger.info("Structuring slices into volumes...")

    scans = {}
    for scan_id, pred_dict in tqdm(scan_map.items()):
        coords = list(pred_dict.keys())
        pred = {
            k: structure_patches([pred_dict[c][k] for c in coords], coords=coords, dims=dims)
            for k in to_struct
        }
        if contiguous:
            pred = {k: v.contiguous() for k, v in pred.items()}
        sample_dict = pred_dict[coords[0]]
        pred.update({k: sample_dict["metadata"].get(k, None) for k in metadata})
        scans[scan_id] = pred
    return scans
