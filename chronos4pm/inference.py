"""One cached Chronos checkpoint and validated one-step requests."""

from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from .configuration import Covariates, FloatArray, ModelConfig, Series

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline
    from torch import Tensor

PREDICTION_LENGTH: int = 1
QUANTILE_LEVELS: list[float] = [0.1, 0.5, 0.9]


class Request(TypedDict):
    """A Chronos history and aligned covariates."""

    target: FloatArray
    past_covariates: dict[str, Series]
    future_covariates: dict[str, Series]


def build_request(
    target: FloatArray,
    past: Covariates,
    future: Covariates,
) -> Request:
    """Validate target and covariate dimensions before loading a model.

    :param target: Scalar or jointly forecast target histories.
    :param past: One series per covariate, aligned to target history.
    :param future: Known length-one values with keys also present in past.
    :return: Chronos request containing independent covariate dictionaries.
    """
    if target.ndim not in (1, 2) or target.size == 0 or np.isinf(target).any():
        raise ValueError(
            "target must be a non-empty one- or two-dimensional history without infinity"
        )
    if not np.isfinite(np.atleast_2d(target)).any(axis=1).all():
        raise ValueError("each target channel needs at least one observed value")
    if not set(future).issubset(past):
        raise ValueError("future covariate keys must be a subset of past keys")
    past_arrays: dict[str, Series] = aligned_covariates(past, target.shape[-1])
    future_arrays: dict[str, Series] = aligned_covariates(future, PREDICTION_LENGTH)
    return {
        "target": target,
        "past_covariates": past_arrays,
        "future_covariates": future_arrays,
    }


def aligned_covariates(values: Covariates, length: int) -> dict[str, Series]:
    """Validate caller-provided numerical or categorical series.

    :param values: Named array-like covariates.
    :param length: Required alignment length.
    :return: Sorted array mapping.
    """
    arrays: dict[str, Series] = {}
    for name, raw in sorted(values.items()):
        array: Series = np.asarray(raw)
        if (
            not isinstance(name, str)
            or not name
            or array.ndim != 1
            or len(array) != length
        ):
            raise ValueError(f"covariate {name!r} must be aligned to length {length}")
        if array.dtype.kind not in "biufUSO":
            raise ValueError(f"covariate {name!r} must contain numbers or categories")
        if array.dtype.kind in "biuf" and np.isinf(array).any():
            raise ValueError(f"covariate {name!r} contains infinity")
        arrays[name] = array
    return arrays


@lru_cache(maxsize=1)
def load_model(config: ModelConfig) -> "Chronos2Pipeline":
    """Lazily load and reuse the configured checkpoint.

    :param config: Overridable package-level MODEL configuration.
    :return: Frozen Chronos-2 pipeline in evaluation mode.
    """
    from chronos import Chronos2Pipeline

    pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
        config.repository_id,
        revision=config.revision,
        device_map=config.device_map,
    )
    pipeline.model.eval()
    resolved: str | None = getattr(pipeline.model.config, "_commit_hash", None)
    if (
        len(config.revision) == 40
        and resolved is not None
        and resolved != config.revision
    ):
        raise RuntimeError("loaded model revision does not match MODEL.revision")
    return pipeline


def forecast(request: Request) -> FloatArray:
    """Forecast one step and return the native Chronos median for each channel.

    :param request: Validated scalar or multivariate request.
    :return: One finite median per target channel.
    """
    import torch

    import chronos4pm

    config: ModelConfig = chronos4pm.MODEL
    pipeline: Chronos2Pipeline = load_model(config)
    channels: int = np.atleast_2d(request["target"]).shape[0]
    group_size: int = channels + len(request["past_covariates"])
    points: list[Tensor]
    with torch.inference_mode():
        _, points = pipeline.predict_quantiles(
            [request],
            prediction_length=PREDICTION_LENGTH,
            quantile_levels=QUANTILE_LEVELS,
            batch_size=max(config.batch_size, group_size),
        )
    if len(points) != 1 or tuple(points[0].shape) != (channels, PREDICTION_LENGTH):
        raise RuntimeError("Chronos returned an unexpected forecast shape")
    result: FloatArray = cast(
        FloatArray, points[0][:, 0].detach().cpu().numpy().astype(float)
    )
    if not np.isfinite(result).all():
        raise ValueError("Chronos returned non-finite predictions")
    return result
