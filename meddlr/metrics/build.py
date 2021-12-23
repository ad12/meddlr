import inspect
import logging
from copy import deepcopy
from typing import Any, Dict

from meddlr.metrics.collection import MetricCollection
from meddlr.metrics.image import MSE, NRMSE, PSNR, RMSE, SSIM
from meddlr.metrics.sem_seg import ASSD, CV, DSC, VOE

logger = logging.getLogger(__name__)


# The builtin metrics that will be recognized by every
# evaluator based on the string completion.
# We specify the class and any keyword args required to
# construct the object.
# fmt: off
_BUILTIN_METRICS = {
    # =========== Image =========== #
    # PSNR
    "psnr": (PSNR, {"im_type": None}),
    "psnr_mag": (PSNR, {"im_type": "magnitude"}),
    "psnr_real": (PSNR, {"im_type": "real"}),
    "psnr_phase": (PSNR, {"im_type": "phase"}),
    "psnr_imag": (PSNR, {"im_type": "imag"}),
    # NRMSE
    "nrmse": (NRMSE, {"im_type": None}),
    "nrmse_mag": (NRMSE, {"im_type": "magnitude"}),
    "nrmse_real": (NRMSE, {"im_type": "real"}),
    "nrmse_phase": (NRMSE, {"im_type": "phase"}),
    "nrmse_imag": (NRMSE, {"im_type": "imag"}),
    # MSE
    "mse": (MSE, {"im_type": None}),
    "mse_mag": (MSE, {"im_type": "magnitude"}),
    "mse_real": (MSE, {"im_type": "real"}),
    "mse_phase": (MSE, {"im_type": "phase"}),
    "mse_imag": (MSE, {"im_type": "imag"}),
    # RMSE
    "rmse": (RMSE, {"im_type": None}),
    "rmse_mag": (RMSE, {"im_type": "magnitude"}),
    "rmse_real": (RMSE, {"im_type": "real"}),
    "rmse_phase": (RMSE, {"im_type": "phase"}),
    "rmse_imag": (RMSE, {"im_type": "imag"}),
    # SSIM
    "ssim (Wang)": (SSIM, {"method": "wang"}),

    # =========== Semantic Segmentation =========== #
    "DSC": (DSC, {}),
    "VOE": (VOE, {}),
    "ASSD": (ASSD, {"connectivity": 1}),
    "CV": (CV, {})
}
# fmt: on


def build_metrics(metric_names, fmt: str = None, **kwargs) -> MetricCollection:
    metrics = {}
    for name in metric_names:
        if name not in _BUILTIN_METRICS.keys():
            raise ValueError(
                "Metric '{}' not found in built-in metrics. "
                "Built-in metrics include:\n{}".format(name, _BUILTIN_METRICS.keys())
            )

        klass: type
        base_kwargs: Dict[str, Any]
        klass, base_kwargs = _BUILTIN_METRICS[name]
        signature = inspect.signature(klass)
        klass_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

        shared_keys = base_kwargs.keys() & klass_kwargs.keys()
        if shared_keys:
            logger.warning(
                "Metric '{}' default values for following arguments will be overridden. "
                "This is may lead to non-reproducible results:\n{}".format(name, shared_keys)
            )
        base_kwargs = deepcopy(base_kwargs)
        base_kwargs.update(klass_kwargs)

        mname = fmt.format(name) if fmt else name
        metrics[mname] = klass(**base_kwargs)

    return MetricCollection(metrics)
