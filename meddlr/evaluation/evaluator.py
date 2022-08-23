import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterable, Mapping, Sequence

import torch

from meddlr.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the
    inputs/outputs.

    This class will accumulate information of the inputs/outputs
    (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass  # pragma: no cover

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(input)`
        """
        pass  # pragma: no cover

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output
        pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass  # pragma: no cover


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators, as_list=False):
        assert len(evaluators)
        super().__init__()
        if isinstance(evaluators, Mapping):
            evaluators = {k: v for k, v in evaluators.items() if v is not None}
        else:
            evaluators = [x for x in evaluators if x is not None]
        self._evaluators = evaluators
        self.as_list = as_list

    def items(self):
        if isinstance(self._evaluators, Mapping):
            return self._evaluators.items()
        return list(enumerate(self._evaluators))

    def values(self) -> Sequence[DatasetEvaluator]:
        if isinstance(self._evaluators, Mapping):
            return self._evaluators.values()
        else:
            return self._evaluators

    def reset(self):
        for evaluator in self.values():
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self.values():
            evaluator.process(input, output)

    def evaluate(self):
        results = [] if self.as_list else OrderedDict()
        for evaluator in self.values():
            result = evaluator.evaluate()
            if self.as_list:
                if result is not None:
                    results.append(result)
                continue

            if result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results " "with the same key {}".format(k)
                    results[k] = v
        return results

    def __getitem__(self, index):
        if isinstance(index, str):
            if not isinstance(self._evaluators, Mapping):
                raise ValueError("Cannot index sequence of evaluators with string key.")
            return self._evaluators[index]
        if isinstance(self._evaluators, Mapping):
            return list(self.values())[index]
        else:
            return self._evaluators[index]

    def __len__(self):
        return len(self._evaluators)

    def __contains__(self, obj):
        return obj in self.values()


def inference_on_dataset(
    model: torch.nn.Module, data_loader: Iterable, evaluator: DatasetEvaluator
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model: a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set
            to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and
            `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)

    total = len(data_loader)  # inference data loader must have a fixed length
    logger.info(f"Start inference on {total} batches")
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            total_compute_time += time.perf_counter() - start_compute_time

            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start  # noqa
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )  # noqa
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: "
        "{} ({:.6f} s / batch)".format(
            total_time_str,
            total_time / (total - num_warmup),
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: "
        "{} ({:.6f} s / batch)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
        )
    )

    eval_start_time = time.perf_counter()
    results = evaluator.evaluate()
    evaluation_time = time.perf_counter() - eval_start_time
    logger.info(f"Evaluation Time: {evaluation_time:.6f} s")
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier
    # for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
