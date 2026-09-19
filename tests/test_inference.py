"""Model configuration and inference boundary checks using explicit test doubles."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import chronos4pm as cpm
from chronos4pm import inference
from chronos4pm.configuration import FloatArray


def test_model_override_and_joint_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the public model constant at call time and keep all target channels.

    :param monkeypatch: Scoped replacement helper.
    :return: None.
    """
    config: cpm.ModelConfig = replace(cpm.MODEL, batch_size=1)
    pipeline: Mock = Mock()
    pipeline.predict_quantiles.return_value = (
        [torch.ones((2, 1, 3))],
        [torch.tensor([[1.0], [2.0]])],
    )
    loader: Mock = Mock(return_value=pipeline)
    monkeypatch.setattr(cpm, "MODEL", config)
    monkeypatch.setattr(inference, "load_model", loader)
    request: inference.Request = inference.build_request(
        np.ones((2, 8)), {"x": np.ones(8)}, {}
    )
    np.testing.assert_array_equal(inference.forecast(request), [1.0, 2.0])
    loader.assert_called_once_with(config)
    assert pipeline.predict_quantiles.call_args.kwargs["batch_size"] == 3
    assert pipeline.predict_quantiles.call_args.kwargs["prediction_length"] == 1


@pytest.mark.parametrize(
    "points", [[], [torch.zeros(1, 2)], [torch.full((1, 1), float("nan"))]]
)
def test_invalid_model_outputs_fail(
    monkeypatch: pytest.MonkeyPatch, points: list[torch.Tensor]
) -> None:
    """Propagate malformed output instead of inventing a fallback prediction.

    :param monkeypatch: Scoped replacement helper.
    :param points: Invalid point forecasts.
    :return: None.
    """
    pipeline: Mock = Mock()
    pipeline.predict_quantiles.return_value = ([], points)
    monkeypatch.setattr(inference, "load_model", Mock(return_value=pipeline))
    with pytest.raises((ValueError, RuntimeError)):
        inference.forecast(inference.build_request(np.ones(8), {}, {}))


def test_model_failures_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never substitute a fallback model when loading fails.

    :param monkeypatch: Scoped replacement helper.
    :return: None.
    """
    monkeypatch.setattr(
        inference, "load_model", Mock(side_effect=OSError("unavailable"))
    )
    with pytest.raises(OSError, match="unavailable"):
        inference.forecast(inference.build_request(np.ones(8), {}, {}))


@pytest.mark.parametrize(
    "target", [np.asarray([]), np.asarray([np.inf]), np.asarray([[np.nan]])]
)
def test_invalid_histories_fail_before_loading(
    target: FloatArray,
) -> None:
    """Reject empty, infinite and entirely missing channels.

    :param target: Invalid target history.
    :return: None.
    """
    with pytest.raises(ValueError):
        inference.build_request(target, {}, {})
