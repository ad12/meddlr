import logging
from collections import defaultdict
from typing import Any, List, Mapping, Sequence, Union

import torch
from torch.utils.data import Sampler

_UNKNOWN_TOKEN = "<UNK>"


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
            logger.warn(
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
