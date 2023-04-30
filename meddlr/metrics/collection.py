from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import tabulate
from torchmetrics.collections import MetricCollection as _MetricCollection

from meddlr.metrics.metric import Metric

__all__ = ["MetricCollection"]


class MetricCollection(_MetricCollection):
    """The class that manages multiple metrics."""

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None
    ) -> None:
        super().__init__(metrics, *additional_metrics, prefix=prefix, postfix=postfix)
        self._is_data_stale = False

    def scans(self) -> List[str]:
        return list(self._scan_data.keys())

    def scan_summary(self, scan_id, delimiter: str = ", ") -> str:
        """Get summary of results for a scan.
        Args:
            scan_id: Scan id for which to summarize results.
            delimiter (`str`, optional): Delimiter between different metrics.
        Returns:
            str: A summary of metrics for the scan. Values are averaged across
                all categories.
        """
        scan_data = self._scan_data[scan_id]
        avg_data = scan_data.mean(axis=1)

        strs = ["{}: {:0.3f}".format(n, avg_data[n]) for n in avg_data.index.tolist()]

        return delimiter.join(strs)

    def to_pandas(self, sync_dist: bool = True) -> pd.DataFrame:
        frames = []
        metric: Metric
        for name, metric in self.items():
            df: pd.DataFrame = metric.to_pandas(sync_dist=sync_dist)
            df["Metric"] = name
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def to_dict(self, group_by="Metric", sync_dist: bool = True) -> Dict[str, Any]:
        df = self.to_pandas(sync_dist=sync_dist)
        df = df.melt(id_vars=["Metric", "id"], var_name="category", value_name="value")
        if len(np.unique(df["category"])) > 1:
            df["Metric"] = df["Metric"] + "/" + df["category"]
        df = df.drop(columns="category")
        values = df.groupby(by=group_by).mean(numeric_only=True)
        return values.to_dict()["value"]

    def summary(self, sync_dist: bool = True) -> str:
        """Get summary of results overall scans.
        Returns:
            str: Tabulated summary. Rows=metrics. Columns=classes.
        """
        df = self.to_pandas(sync_dist=sync_dist)
        if "id" in df:
            df = df.drop(columns="id")
        df = df.groupby(by="Metric")

        mean = df.mean().applymap(lambda x: "{:0.3f}".format(x))
        std = df.std().applymap(lambda x: "{:0.3f}".format(x))
        df = mean + " (" + std + ")"
        return tabulate.tabulate(df, headers=df.columns) + "\n"

    def ids(self, sync_dist=True) -> Set[str]:
        _ids = set()
        metric: Metric
        for _, metric in self.items():
            _ids |= set(metric.to_pandas(sync_dist=sync_dist)["id"].to_numpy())
        return _ids
