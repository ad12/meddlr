"""Implementation of different dataset samplers."""
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ["AlternatingSampler"]


class AlternatingSampler(Sampler):
    """Sampler that alternates between supervised and unsupervised examples.

    In many semi-supervised scenarios, we would like to switch between using
    supervised and unsupervised examples. This sampler facilitates returning
    samples in this alternating pattern.

    Note:
        Currently, periods :math:`T_s` and :math:`T_us` must be perfect divisors
        of the number of supervised_idxs and unsupervised_idxs.
    """

    def __init__(
        self, dataset, T_s: int, T_us: int, indices: Sequence[int] = None, seed: int = None
    ):
        """
        Args:
            dataset: dataset to sample from. Must have methods
                `get_supervised_idxs()` and `get_unsupervised_idxs()`
            T_s (int): Period for returning supervised examples
            T_us (int): Period for returning unsupervised examples
            indices (Sequence[int], optional): Indices in the dataset to consider.
                If specified, only these indices will be iterated over.
        """
        super().__init__(dataset)

        supervised_idxs = dataset.get_supervised_idxs()
        unsupervised_idxs = dataset.get_unsupervised_idxs()
        if indices is not None:
            supervised_idxs = [x for x in supervised_idxs if x in indices]
            unsupervised_idxs = [x for x in unsupervised_idxs if x in indices]
        self._supervised_idxs = torch.tensor(supervised_idxs)
        self._unsupervised_idxs = torch.tensor(unsupervised_idxs)

        if len(self._unsupervised_idxs) == 0 or len(self._supervised_idxs) == 0:
            raise ValueError(
                "AlternatingSampler can only be used for "
                "semi-supervised training. "
                "Dataset must have both supervised/unsupervised examples"
            )

        if len(self._supervised_idxs) % T_s != 0:
            raise ValueError(
                "Period T_s must be a perfect divisor of "
                "the number of supervised indices. "
                "Got {} supervised indices, and T_s={}".format(len(self._supervised_idxs), T_s)
            )
        if len(self._unsupervised_idxs) % T_s != 0:
            raise ValueError(
                "Period T_us must be a perfect divisor of "
                "the number of unsupervised indices. "
                "Got {} unsupervised indices, and T_us={}".format(
                    len(self._unsupervised_idxs), T_us
                )
            )

        self.T_s = T_s
        self.T_us = T_us

        if seed is None:
            rng = torch.Generator().set_state(torch.random.get_rng_state())
        else:
            rng = torch.Generator().manual_seed(seed)
        self._rng = rng

        # The least common multiple between the number of
        # supervised block and number of unsupervised blocks in single pass.
        # num_blocks = num_examples / num_period
        num_blocks_supervised = int(len(self._supervised_idxs) / T_s)
        num_blocks_unsupervised = int(len(self._unsupervised_idxs) / T_us)
        self._lcm = int(np.lcm(num_blocks_supervised, num_blocks_unsupervised))
        # The sampler is determined by how many blocks of each data type
        # (supervised/unsupervised) are returned.
        # Number of passes indicates how many times we have to go through
        # the dataset based on the least common multiple.
        num_passes_supervised = int(self._lcm / num_blocks_supervised)
        num_passes_unsupervised = int(self._lcm / num_blocks_unsupervised)
        self._num_samples_sup = num_passes_supervised * len(self._supervised_idxs)  # noqa
        self._num_samples_unsup = num_passes_unsupervised * len(self._unsupervised_idxs)  # noqa

    def get_indices(self, as_list=False):
        s_idxs = self._build_idx(self._supervised_idxs, self.T_s, self._num_samples_sup)
        us_idxs = self._build_idx(self._unsupervised_idxs, self.T_us, self._num_samples_unsup)

        # TODO: Add parameter to choose whether to start with supervised or
        # unsupervised data.
        idxs = torch.cat([s_idxs, us_idxs], dim=1).reshape(-1)
        if as_list:
            idxs = idxs.tolist()
        return idxs

    def __len__(self):
        return self._num_samples_sup + self._num_samples_unsup

    def __iter__(self):
        return iter(self.get_indices(as_list=True))

    def _build_idx(self, idxs: torch.Tensor, T: int, num_samples: int):
        assert num_samples % len(idxs) == 0
        num_passes = int(num_samples / len(idxs))

        perm_idxs = torch.cat(
            [torch.randperm(len(idxs), generator=self._rng) for _ in range(num_passes)]
        )

        idxs = idxs[perm_idxs].reshape(-1, T)
        return idxs
