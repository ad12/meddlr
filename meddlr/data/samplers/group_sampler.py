import logging
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler

from meddlr.utils import comm

_UNKNOWN_TOKEN = "<UNK>"

__all__ = ["GroupSampler", "AlternatingGroupSampler", "DistributedGroupSampler"]


class GroupSampler(Sampler):
    """
    TODO:
        - uniform group sampling (n groups per batch)
        - weighted group sampling (n groups per batch)
        - frequency sampling (duty cycle of each group)
    """

    def __init__(
        self,
        dataset,
        batch_by: Any = None,
        batch_size: int = None,
        as_batch_sampler: bool = False,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ):
        """
        Args:
            dataset: dataset to sample from.
                Must have attribute ``.examples``, which returns a list of
                dictionaries of examples with associated metadata.
            batch_by (Any, Sequence[Any]): Metadata field(s) to batch by.
            batch_size (int, optional): The batch size. Required if ``batched_by``
                specified or ``as_batch_sampler=True``.
            as_batch_sampler (bool, optional): If ``True``, this instance
                will behave like :cls:`torch.utils.data.BatchSampler.
                This is highly recommended if ``batched_by`` if specified.
            drop_last (bool, optional): If ``True``, drop the last batch for each group.
                If ``batch_by`` is specified, this will also drop the last batch for each group
                if it does not meet the specifications.
            shuffle (bool, optional): If ``True``, shuffles the data.
            seed (torch.Tensor): Random seed to use for initialization.
        """
        group_by = None

        super().__init__(dataset)
        logger = logging.getLogger(__name__)

        if batch_by and not batch_size:
            raise ValueError("`batch_size` must be specified if `batched=True`")
        if batch_by and not as_batch_sampler:
            logger.warning(
                "Using `batch_by` without batch sampling functionality. "
                "To use as a batch sampler, set `as_batch_sampler=True`."
            )

        # Configure batch by groups if they exist.
        group_unknown = False
        if batch_by:
            groups = _build_groups(dataset.examples, batch_by, group_unknown=group_unknown)
            if group_by:
                groups = {
                    grp: _build_groups(
                        dataset.examples, group_by, group_unknown=group_unknown, indices=idxs
                    )
                    for grp, idxs in groups.items()
                }
            else:
                groups = {k: {_UNKNOWN_TOKEN: v} for k, v in groups.items()}
        else:
            groups = {
                _UNKNOWN_TOKEN: _build_groups(
                    dataset.examples, group_by, group_unknown=group_unknown
                )
            }

        if seed is None:
            rng = torch.Generator().set_state(torch.random.get_rng_state())
        else:
            rng = torch.Generator().manual_seed(seed)

        self.group_by = group_by
        self.batch_by = batch_by
        self.batch_size = batch_size
        self.as_batch_sampler = as_batch_sampler
        self.drop_last = drop_last
        self.shuffle = shuffle

        self._rng = rng
        self._groups = groups
        batches = self._build_batches(shuffle=False)
        self._length = (
            len(batches) if self.as_batch_sampler else len(torch.cat(batches, dim=0).tolist())
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        batches: List[torch.Tensor] = self._build_batches()

        if self.as_batch_sampler:
            return iter(batches)
        else:
            return iter(torch.cat(batches, dim=0).tolist())

    def _build_batches(self, shuffle: bool = None):
        """Build batches of examples to sample."""
        # Shuffle indices (if applicable).
        shuffle = self.shuffle if shuffle is None else shuffle
        if shuffle:
            groups = {k: _shuffle_groups(v, self._rng) for k, v in self._groups.items()}
        else:
            groups = self._groups

        # Break into chunks of <= batch_size.
        batch_size = self.batch_size if self.batch_size else 1
        if batch_size:
            groups = [
                torch.split(torch.as_tensor(v), batch_size)
                for grp in groups.values()
                for v in grp.values()
            ]
            # Drop last
            if self.drop_last:
                groups = [v[:-1] if len(v[-1]) < batch_size else v for v in groups]

        batches = [batch for grp in groups for batch in grp]

        # Reorder groups if shuffle is enabled.
        if shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=self._rng)]

        return batches


class AlternatingGroupSampler(GroupSampler):
    """Alternate sampler between supervised/unsupervised examples within groups."""

    def __init__(
        self,
        dataset,
        T_s: int = 1,
        T_us: int = 1,
        batch_by: Any = None,
        batch_size: int = None,
        as_batch_sampler: bool = False,
        drop_last: bool = False,
        seed: int = None,
    ):
        self.T_s = T_s
        self.T_us = T_us

        example = dataset.examples[0]
        key = [k for k in ("is_unsupervised", "_is_unsupervised") if k in example]
        assert len(key) == 1
        key = key[0]

        if batch_by is None:
            batch_by = ()
        batch_by = tuple(batch_by) if isinstance(batch_by, (list, tuple)) else (batch_by,)
        batch_by += (key,)

        super().__init__(
            dataset,
            batch_by=batch_by,
            batch_size=batch_size if batch_size else 1,
            as_batch_sampler=as_batch_sampler,
            drop_last=False,
            shuffle=True,
            seed=seed,
        )

        # Groups should only have one subgroup.
        counts = {}
        counts_per_group = defaultdict(int)
        for grp_name, grp in self._groups.items():
            if len(grp) > 1:
                raise ValueError(
                    f"All groups should only have 1 subgroup. "
                    f"{grp_name} has {len(grp)} - {grp.keys()}"
                )
            num = len(list(grp.values())[0])
            counts_per_group[grp_name[:-1]] += num
            counts[grp_name] = num
        # Drop any groups that will be smaller than the batch size.
        if drop_last:
            drop_keys = [grp_name for grp_name, count in counts.items() if count < batch_size]
            for k in drop_keys:
                counts_per_group.pop(k)
                for gk in [k + (True,), k + (False,)]:
                    counts.pop(gk, None)
                    self._groups.pop(gk, None)
        self.counts_per_group = counts_per_group
        self.counts = counts
        self._samplers = self._build_samplers(seed=seed)

        # State variables
        self._iterators = {}
        self._counter = {grp_name: 0 for grp_name in self._samplers}
        self._pointer = 0
        self._excess_sampled = 0

    def __len__(self):
        """
        Note:
            This property should not be used for any real logic.
            It is purely to satisfy the requirement that samplers
            used as BatchSamplers should have a length.
        """
        return sum(v // self.batch_size for v in self.counts.values())

    def _batch_by_groups(self):
        return list(self.counts_per_group.keys())

    def _build_samplers(self, seed=None) -> Dict[Tuple[Hashable], SubsetRandomSampler]:
        groups = self._groups
        samplers = {}
        for grp_idx, (grp_name, grp) in enumerate(groups.items()):
            indices = list(grp.values())[0]
            g_seed = seed + grp_idx if seed is not None else None
            gen = torch.Generator()
            if g_seed is not None:
                gen = gen.manual_seed(g_seed)
            samplers[grp_name] = SubsetRandomSampler(indices, generator=gen)
        return samplers

    def _sample(self, group: Hashable, unsupervised: bool):
        def _reset_iterator(grp):
            self._iterators[grp] = iter(self._samplers[grp])

        group = group + (bool(unsupervised),)

        if group not in self._iterators:
            _reset_iterator(group)

        try:
            idx = next(self._iterators[group])
        except StopIteration:
            _reset_iterator(group)
            idx = next(self._iterators[group])

        self._counter[group] += 1
        return idx

    def _get_group(self, excess_sample, to_sample: List[bool]):
        """Return the group that should be sampled for this batch.

        Args:
            excess_sample (int): A value ``<0`` indicates that a surplus of supervised
                examples have previously been sampled. The algorithm will then choose
                a group with probability based on the remaining number of unsupervised examples.
                If value is ``>0``, then vice versa.
        """
        batch_by = self._batch_by_groups()
        sample_by_groups = (
            [excess_sample < 0] if excess_sample != 0 else sorted(np.unique(to_sample))
        )

        # Find the relative epoch for each iterator.
        num_examples_by_group = np.asarray(
            [
                [self.counts.get(batch_grp + (x,), 0) for x in sample_by_groups]
                for batch_grp in batch_by
            ]
        )
        counter = np.asarray(
            [
                [self._counter.get(batch_grp + (x,), 0) for x in sample_by_groups]
                for batch_grp in batch_by
            ]
        )

        epoch_num = counter // (num_examples_by_group + (num_examples_by_group == 0))
        epoch_num[num_examples_by_group == 0] = np.max(epoch_num) + 1
        remainder = num_examples_by_group - counter % (
            num_examples_by_group + (num_examples_by_group == 0)
        )
        remainder[epoch_num != np.min(epoch_num)] = 0
        weights = np.sum(remainder, axis=1)
        p = torch.as_tensor(weights / np.sum(weights))

        idx = torch.multinomial(p, 1, generator=self._rng).item()
        return batch_by[idx]

    def _next_batch(self) -> List[int]:
        batch_size: int = self.batch_size
        # pointer must be in range [0, self.T_s + self.T_us)
        pointer = self._pointer
        # False = supervised, True = unsupervised
        to_sample: List[bool] = []

        num_excess_sample = min(abs(self._excess_sampled), batch_size)
        to_sample.extend([self._excess_sampled < 0] * num_excess_sample)

        num_alt_sample = batch_size - len(to_sample)
        c_range = np.arange(pointer, pointer + num_alt_sample)
        to_sample.extend(c_range % (self.T_s + self.T_us) >= self.T_s)

        # Find group to sample.
        group = self._get_group(self._excess_sampled, to_sample)

        # If sampled group only has supervised or unsupervised examples (but not both),
        # the batch will be composed of only supervised or only unsupervised examples.
        # This violates the duty cycle of supervised to unsupervised samples.
        # We change the self._excess_sampled state variable to indicate that a surplus
        # of either supervised or unsupervised samples has been selected. In this case,
        # the next batch will attempt to "correct" this sampling by oversampling the
        # opposite.
        is_unsupervised_options = [False, True]
        has_supervised_unsupervised = [
            group + (is_unsup,) in self._groups for is_unsup in is_unsupervised_options
        ]
        # TODO: Check if to_sample is made up of all of same type.
        if not all(has_supervised_unsupervised):
            idx = np.where(has_supervised_unsupervised)[0][0]
            is_unsup = is_unsupervised_options[idx]
            to_sample = [is_unsup] * batch_size
            shift_pointer = 0
            while is_unsup == ((pointer + shift_pointer) % (self.T_s + self.T_us) >= self.T_s):
                shift_pointer += 1
            self._excess_sampled += ((-1) ** (is_unsup + 1)) * (batch_size - shift_pointer)
            self._pointer += shift_pointer
        else:
            if self._excess_sampled < 0:
                self._excess_sampled = self._excess_sampled + num_excess_sample
            else:
                self._excess_sampled = self._excess_sampled - num_excess_sample
            self._pointer = (self._pointer + num_alt_sample) % (self.T_s + self.T_us)

        samples = torch.as_tensor([self._sample(group, is_unsup) for is_unsup in to_sample])
        return samples

    def _build_batches(self, shuffle: bool = None):
        """Build batches of examples to sample."""
        # Hacky way of getting around GroupSampler.__init__ _length computation.
        if not hasattr(self, "counts"):
            return [torch.tensor([])]

        return [self._next_batch() for _ in range(len(self))]


class DistributedGroupSampler(Sampler):
    """Samples examples such that examples from the same group are on the same process.

    `dataset` must support the following attributes and methods
        * `groups(group_by) -> Dict`: Returns mapping from group id to indices

    dataset examples is a list of tuples (fname, instance),
    where fname is essentially the volume name (actually a filename).

    Note:
        Because the same groups will appear on the same process, this sampler is not
        recommended for training.
    """

    def __init__(
        self,
        dataset,
        group_by: str,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        pad: bool = False,
    ):
        self.dataset = dataset
        self.num_replicas = self._world_size() if num_replicas is None else num_replicas
        self.rank = self._rank() if rank is None else rank
        self.epoch = 0
        self.seed = seed
        self._group_by = group_by
        self.shuffle = shuffle
        # TODO: Determine what drop_last=True should mean for this sampler.
        self.drop_last = False
        self.pad = pad

        # All processes.
        all_groups: Dict[Hashable, List[int]] = dataset.groups(group_by)
        group_to_size = {k: len(v) for k, v in all_groups.items()}
        all_groups_split = _split_groups_greedy(group_to_size, self.num_replicas)
        max_size = max(
            sum(len(all_groups[x]) for x in process_split) for process_split in all_groups_split
        )

        self.all_groups_split: List[List[Hashable]] = all_groups_split
        self.max_size = max_size

        # This process.
        self.groups = self.all_groups_split[self.rank]
        self.indices = np.concatenate([all_groups[group_id] for group_id in self.groups])
        self.num_samples = len(self.indices)

    def _world_size(self):
        return comm.get_world_size()

    def _rank(self):
        return comm.get_rank()

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(self.num_samples, generator=g).tolist()
            indices = self.indices[ordering]
        else:
            indices = self.indices

        # Distributed sampler needs to pad so that
        if self.pad and len(indices) < self.max_size:
            pad_size = self.max_size - self.num_samples
            indices = np.concatenate([indices, indices[:pad_size]])

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def _shuffle_groups(groups, rng):
    if not isinstance(groups, Mapping):
        return torch.as_tensor(groups)[torch.randperm(len(groups), generator=rng)]

    return {k: _shuffle_groups(v, rng) for k, v in groups.items()}


def _build_groups(
    examples,
    group_by: Union[Any, Sequence[Any]] = None,
    group_unknown: bool = False,
    indices: Sequence[int] = None,
):
    if not indices:
        indices = range(len(examples))

    if group_by is None:
        return {_UNKNOWN_TOKEN: indices}
    elif isinstance(group_by, str):
        group_by = (group_by,)

    group_by_values = []
    for idx in indices:
        ex = examples[idx]
        values = []
        for gb in group_by:
            if gb in ex:
                values.append(ex[gb])
            elif gb in ex.get("_metadata", {}):
                values.append(ex["_metadata"][gb])
            else:
                values.append(_UNKNOWN_TOKEN)

        if group_unknown and _UNKNOWN_TOKEN in values:
            values = [_UNKNOWN_TOKEN]
        group_by_values.append(tuple(values))

    group_to_idxs = defaultdict(list)
    for idx, val in zip(indices, group_by_values):
        group_to_idxs[val].append(idx)

    return group_to_idxs


def _split_groups_greedy(group_sizes: Dict[Hashable, int], num_splits: int) -> List[List[Hashable]]:
    """Split array into approximately equal chunks.

    This process is greedy, so it may not result in the most optimal splits.

    Args:
        group_sizes: The group_name -> size mapping.
        num_splits: Number of splits.

    Returns:
        List of arrays.
    """
    split_groups = [[] for _ in range(num_splits)]
    split_sizes = np.zeros(num_splits)

    group_names = sorted(group_sizes.keys(), key=lambda x: group_sizes[x], reverse=True)
    for group_name in group_names:
        split_idx = np.argmin(split_sizes)
        split_groups[split_idx].append(group_name)
        split_sizes[split_idx] += group_sizes[group_name]

    return split_groups
