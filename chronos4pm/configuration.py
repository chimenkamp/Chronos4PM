"""Typed parameters shared by the small functional API."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pm4py.objects.log.obj import EventLog

Log: TypeAlias = EventLog | pd.DataFrame
Edge: TypeAlias = tuple[str, str]
Key: TypeAlias = str | tuple[str, ...] | tuple[Edge, int]
FloatArray: TypeAlias = NDArray[np.float64]
Series: TypeAlias = NDArray[np.generic]
Covariates: TypeAlias = Mapping[str, ArrayLike]
CovariateSet: TypeAlias = Literal["target_only", "selected", "global"]
Task: TypeAlias = Literal[
    "dfg",
    "edges",
    "start",
    "end",
    "activity",
    "variants",
    "footprints",
    "performance",
    "bottlenecks",
    "temporal",
    "entropy",
    "variety",
    "completeness",
    "next_activity",
    "next_time",
    "remaining_time",
]
ACTIVITY_KEY: str = "concept:name"
TIMESTAMP_KEY: str = "time:timestamp"
CASE_ID_KEY: str = "case:concept:name"
WINDOW_COUNT: int = 8
PRESENCE_THRESHOLD: float = 0.5
RELATIONS: tuple[str, ...] = ("none", "sequence", "parallel")


@dataclass(frozen=True)
class ModelConfig:
    """Identify one checkpoint and its inference device; no training occurs."""

    repository_id: str = "autogluon/chronos-2-small"
    revision: str = "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"
    device_map: str = "cpu"
    batch_size: int = 256

    def __post_init__(self) -> None:
        """Reject incomplete model configuration.

        :return: None.
        """
        if not self.repository_id or not self.revision or self.batch_size < 1:
            raise ValueError(
                "model repository, revision and positive batch_size are required"
            )


class ForecastOptions(TypedDict, total=False):
    """Additional keyword parameters accepted by aggregate functions."""

    n_windows: int
    forecast_origin: pd.Timestamp
    activity_key: str
    timestamp_key: str
    case_id_key: str
    attribute_allowlist: Sequence[str]
    past_covariates: Covariates
    future_covariates: Covariates


class PrefixOptions(TypedDict, total=False):
    """Additional keyword parameters accepted by prefix functions."""

    activity_key: str
    timestamp_key: str
    case_id_key: str
    attribute_allowlist: Sequence[str]
    past_covariates: Covariates
    future_covariates: Covariates


class Footprint(TypedDict):
    """The three PM4Py footprint fields forecast by the benchmark."""

    activities: set[str]
    sequence: set[Edge]
    parallel: set[Edge]
