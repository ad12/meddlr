"""Utilities for partitioning data into different splits.

DO NOT SHARE THIS FILE EXTERNALLY WITHOUT PRIOR APPROVAL.
"""
import logging
import os
import random
from copy import deepcopy
from typing import Any, List, Union

import numpy as np


def _random_balance(
    bins: List[List[Any]],
    expected_dist: Union[List[float], np.ndarray],
    buffer: float,
    weights_dict: dict,
    iters: int = 100,
):
    """Randomly balance bins.

    Args:
        bins (:obj:`List[List[Any]]`): Separated bins.
        expected_dist (array-like): Expected distribution/split percentages. Must sum up to 1.
        buffer (:obj:`float`): If bin percentages are within +/-:obj:`buffer`, the split is valid.
        weights_dict (:obj:`dict`): Mapping from value to weight.
        iters (:obj:`int`, optional): Number of iterations to try. Defaults to ``100``.

    Returns:
        :obj:`List[List[Any]]`: Balanced bins.
    """
    if not np.allclose(np.sum(expected_dist), 1):
        raise ValueError("expected_dist must sum to 1.")

    curr_weights = np.asarray([np.sum([weights_dict[elem] for elem in c_bin]) for c_bin in bins])
    total_weight = np.sum(curr_weights)
    buffer = buffer * total_weight
    expected_dist = np.asarray(expected_dist) * total_weight

    assert len(bins) == len(
        expected_dist
    ), "Size mismatch: {} bins, {} expected distributions".format(len(bins), len(expected_dist))

    for _ in range(iters):
        for c_ind in range(len(bins)):
            c_bin = bins[c_ind]  # current bin

            # If bin is within expected buffer range, then discount.
            delta = curr_weights[c_ind] - expected_dist[c_ind]
            if np.abs(delta) <= buffer:
                continue

            is_bin_overfilled = delta > 0

            # Select candidate bin. If c_bin is overfilled (delta > 0),
            # pick an underfilled bin. And vice versa.
            candidate_indexes = (
                np.argwhere(curr_weights - expected_dist < 0).flatten()
                if is_bin_overfilled
                else np.argwhere(curr_weights - expected_dist > 0).flatten()
            )
            candidate_indexes = list(set(candidate_indexes) - {c_ind})  # exclude current bin
            candidate_ind = np.random.choice(candidate_indexes)
            candidate_bin = bins[candidate_ind]

            # Select element at random from current bin.
            c_bin_elem = c_bin[np.random.choice(np.arange(len(c_bin)))]
            ce_weight = weights_dict[c_bin_elem]

            # Select candidate element.
            # If overfilled, pick element with weight < ce_weight.
            # If underfilled, pick element with weight > ce_weight.
            # If no such candidate element is found, we continue.
            candidate_bin_weights = np.asarray([weights_dict[elem] for elem in candidate_bin])
            candidate_elems_idxs = (
                np.argwhere(candidate_bin_weights < ce_weight).flatten()
                if is_bin_overfilled
                else np.argwhere(candidate_bin_weights > ce_weight).flatten()
            )
            if len(candidate_elems_idxs) == 0:
                continue
            candidate_elem_ind = np.random.choice(candidate_elems_idxs)
            candidate_elem = candidate_bin[candidate_elem_ind]
            candidate_elem_weight = candidate_bin_weights[candidate_elem_ind]

            # Swap elements and adjust weights.
            c_bin.remove(c_bin_elem)
            candidate_bin.remove(candidate_elem)
            c_bin.append(candidate_elem)
            candidate_bin.append(c_bin_elem)

            curr_weights[c_ind] += candidate_elem_weight - ce_weight
            curr_weights[candidate_ind] += ce_weight - candidate_elem_weight

            if np.all(np.abs(curr_weights - expected_dist) < buffer):
                return bins

    raise ValueError("Random balancing failed. Try increasing the number of iterations.")


def _greedy_balance(
    bins: List[List[Any]],
    expected_dist: Union[List[float], np.ndarray],
    buffer: float,
    weights_dict: dict,
):
    """Greedily balance bins.

    Fills the most underfilled bin from most overfilled bin at each iteration.

    Args:
        bins (:obj:`List[List[Any]]`): Separated bins.
        expected_dist (array-like): Expected distribution/split percentages. Must sum up to 1.
        buffer (:obj:`float`): If bin percentages are within +/-:obj:`buffer`, the split is valid.
        weights_dict (:obj:`dict`): Mapping from value to weight.

    Returns:
        :obj:`List[List[Any]]`: Balanced bins.
    """
    if not np.allclose(np.sum(expected_dist), 1):
        raise ValueError("expected_dist must sum to 1.")

    curr_weights = np.asarray([np.sum([weights_dict[elem] for elem in c_bin]) for c_bin in bins])
    total_weight = np.sum(curr_weights)
    buffer = buffer * total_weight
    expected_dist = np.asarray(expected_dist) * total_weight

    # Sort elements in bins in ascending order of weight.
    # First element will always have the lowest weight among all elements in the bin.
    sorted_bins = [
        sorted([(val, weights_dict[val]) for val in c_bin], key=lambda x: x[-1]) for c_bin in bins
    ]
    bins = [[val for val, _ in x] for x in sorted_bins]

    # Create list of { -> num elements with weight}.
    while True:
        deltas = curr_weights - expected_dist
        if np.all(np.abs(deltas) < buffer):
            break

        # Select the most underfilled bin and the delta weight.
        uf_bin_ind = np.argmin(deltas)  # index of most underfilled bin.
        uf_bin = bins[uf_bin_ind]
        uf_delta_mag = np.abs(deltas[uf_bin_ind])

        # Select most overfilled bin that has at least one element.
        # An overfilled bin has a positive delta.
        of_bin_ind = -1
        for ind in np.argsort(deltas).flatten()[::-1]:
            c_delta = deltas[ind]
            if c_delta <= 0:
                break

            # Check if there is at least one element with a weight <= min(c_delta, uf_delta_mag).
            if weights_dict[bins[ind][0]] <= min(c_delta, uf_delta_mag):
                of_bin_ind = ind
                break

        if of_bin_ind == -1:
            break

        assert of_bin_ind != uf_bin_ind, "Bin cannot be both overfilled and underfilled"
        max_weight = min(uf_delta_mag, deltas[of_bin_ind])
        of_bin = bins[of_bin_ind]

        # Select the element greedily from the overfilled bin
        # that has a weight <= abs(delta[uf_bin]).
        candidate_elems = []
        candidate_weight = 0
        for ind, elem in enumerate(of_bin):  # noqa: B007
            e_weight = weights_dict[elem]
            if e_weight > max_weight or e_weight < candidate_weight:
                continue

            if e_weight == candidate_weight:
                candidate_elems.append(elem)
            elif e_weight > candidate_weight:
                candidate_weight = e_weight
                candidate_elems = [elem]

        candidate_elem = np.random.choice(candidate_elems)

        # Move candidate element to underfilled bin.
        of_bin.remove(candidate_elem)
        uf_bin.append(candidate_elem)

        curr_weights[uf_bin_ind] += candidate_weight
        curr_weights[of_bin_ind] -= candidate_weight

    return bins


def approximately_split_weighted(
    values: List[Any],
    split_percentages: List[float],
    weights: List[int] = None,
    balance: str = None,
    buffer: float = 0.01,
    num_iters: int = 100,
    seed: int = None,
) -> List[List[Any]]:
    """Approximately split values into different bins.

    This function should be used to split :obj:`values` into N disjoint sets.
    This is useful for applications in machine learning where dataset splits
    should be disjoint.

    In certain cases, values will not have uniform weights. For example, consider
    the case where there are frames extracted from 3 videos: video A- 10 frames,
    video B- 5 frames, video C- 2 frames. In this case, we would like to keep frames
    sampled from the same video within a specific set. For example, frames from video A
    cannot be in both bin 1 and bin 2. The :obj:`weights` attribute can be specified to
    keep the videos separate:

    >>> approximately_split(values=["videoA", "videoB", "videoC"], weights=[10, 5, 2], ...)

    In these cases, because of the randomness of splits, we may end up with a
    lop-sided split. Let's consider the example above, where the
    ``split_percentages=[0.3, 0.7]``. Assume after random shuffling values are in the order
    ``["videoA", "videoB", "videoC"]``. In this case, no value will be assigned to the first
    bin. Balancing will allow bins to randomly or greedily reconfigure into an appropriate.
    This method is not guaranteed to converge.

    In random balancing, entries between overfilled and underfilled bins are pseudo-randomly
    swapped. Bins are sequentially selected. If the current bin is overfilled, a candidate bin
    that is underfilled is selected at random. If the current bin is underfilled, a candidate
    bin that is overfilled is selected. Then an element from the overfilled bin with a weight
    greater than that of the other element is randomly selected and the two elements are
    swapped. This is repeated at most :obj:`iter` times or until bin is within buffer.

    In greedy balancing, the bin that is most underfilled is filled greedily
    iteratively from overfilled bins just enough so that it meets the buffer criteria.
    To fill the underfilled bin to as close to its size, set :obj:`buffer=0`.

    TODO: Write unit test.

    Args:
        values (:obj:`List[Any]`): A list of values that should be divided into different groups.
        split_percentages (:obj:`List[float]`): The approximate percent split between datasets.
            Must add up to 1.
        weights (:obj:`List[int]`, optional): Weights for each value in :obj:`values`.
            Use if certain values have a larger weight than others (i.e. more examples
            associated with value). Defaults to weight of 1 per each value in :obj:`values`.
        balance (:obj:`str`, optional): Balance weighting after partition.
            Options: `"greedy"`, `"random"`. Use with :obj:`weights`. Defaults to ``None``.
        buffer (:obj:`float`, optional): If bin percentages are within +/-:obj:`buffer`,
            the split is valid. Defaults to 0.01 (+/-1%).
        num_iters (:obj:`int`, optional): Number of iterations for random balancing.
            Defaults to ``100``.

    Returns:
        List[List[Any]]: A list of splits of values.
    """
    if not np.allclose(np.sum(split_percentages), 1):
        raise ValueError("split_percentages must sum to 1.")

    if len(set(values)) != len(values):
        raise ValueError("Values must be distinct")

    if balance not in [None, "greedy", "random"]:
        raise ValueError("Balance {} not supported".format(balance))

    values = np.asarray(values)
    num_values = len(values)

    use_weights = False
    if weights:
        if not isinstance(weights, (list, tuple, np.ndarray)):
            raise TypeError("weights must be a list, tuple, or np.ndarray")
        if len(weights) != num_values:
            raise ValueError(
                "{} weights found, but {} values provided".format(len(weights), num_values)
            )
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights must be 1D")
        use_weights = True
    else:
        weights = np.ones(num_values)

    total_weight = np.sum(weights)

    split_percentages = np.asarray(split_percentages)
    boundaries = list(np.cumsum(split_percentages * total_weight))
    boundaries[-1] = np.ceil(boundaries[-1])

    if seed is not None:
        rand_state = np.random.get_state()
        np.random.seed(seed)

    shuffled_inds = np.arange(num_values)
    np.random.shuffle(shuffled_inds)
    values = values[shuffled_inds]
    weights = weights[shuffled_inds]
    cum_weights = np.cumsum(weights)

    start_ind = 0

    outputs = []
    for bnd in boundaries:
        end_ind = np.max(np.argwhere(cum_weights - bnd <= 0)) + 1
        outputs.append(list(values[start_ind:end_ind]))
        start_ind = end_ind

    if use_weights and balance:
        weights_dict = {k: v for k, v in zip(values, weights)}
        if balance == "greedy":
            outputs = _greedy_balance(outputs, split_percentages, buffer, weights_dict)
        elif balance == "random":
            outputs = _random_balance(outputs, split_percentages, buffer, weights_dict, num_iters)
        else:
            assert False, "Should never reach here."

    assert np.sum([len(x) for x in outputs]) == num_values
    assert len(outputs) == len(split_percentages)
    assert all(
        len(set(outputs[i]) & set(outputs[j])) == 0
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    )

    for x in outputs:
        assert len(x) > 0

    if seed is not None:
        np.random.set_state(rand_state)

    return outputs


def approximately_split(values: List[Any], split_percentages: List[float]) -> List[List[Any]]:
    """Approximately split values into different bins.

    TODO: Deprecate once unit test is written for approximately_split_weighted.

    Args:
        values (:obj:`List[Any]`): A list of values that should be divided into different groups.
        split_percentages (:obj:`List[float]`): The approximate percent split between datasets.
            Must add up to 1.

    Returns:
        List[List[Any]]: A list of splits of values.
    """
    if not np.allclose(np.sum(split_percentages), 1):
        raise ValueError("split_percentages must sum to 1.")

    values = deepcopy(values)

    num_values = len(values)
    split_percentages = np.asarray(split_percentages)
    boundaries = list(np.cumsum(split_percentages * num_values))
    boundaries[0] = np.ceil(boundaries[0])
    boundaries[-1] = np.ceil(boundaries[-1])

    random.shuffle(values)
    start_ind = 0

    outputs = []
    for end_ind_f in boundaries:
        # Round such that .5 always rounds up.
        end_ind = int(np.floor(end_ind_f + 0.5))
        outputs.append(values[start_ind:end_ind])
        start_ind = end_ind

    assert np.sum([len(x) for x in outputs]) == num_values
    assert len(outputs) == len(split_percentages)

    for x, per in zip(outputs, split_percentages):
        if per > 0:
            assert len(x) > 0

    return outputs


def write_split(train_split, val_split, test_split, save_path):
    """Write files in given splits to directory.

    Args:
        train_split (:obj:`List[str]`): Training files.
        val_split (:obj:`List[str]`): Validation files.
        test_split (:obj:`List[str]`): Test Files.
        save_path (str): Directory to save partitions.
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    outputs = [train_split, val_split, test_split]

    # Write partition folders per dataset to text file.
    dataset_suffixes = ["train", "val", "test"]
    filenames = ["%s.txt" % x for x in dataset_suffixes]
    for i in range(len(outputs)):
        fpath = os.path.join(save_path, filenames[i])
        logging.info("Writing %d partitions to %s." % (len(outputs[i]), fpath))

        with open(fpath, "w+") as f:
            for partition_folder in outputs[i]:
                f.write("{}\n".format(partition_folder))
