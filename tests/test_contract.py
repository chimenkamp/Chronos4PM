"""Public behavior and native-library contracts; model decoding is isolated here."""

from typing import cast

import numpy as np
import pandas as pd
import pm4py
import pytest
from numpy.typing import NDArray
from pm4py.objects.log.obj import EventLog, Trace

import chronos4pm as cpm
from chronos4pm import inference
from chronos4pm.configuration import Edge, FloatArray, ForecastOptions, Key
from chronos4pm.logs import context_windows, normalize_log
from chronos4pm.tasks import history, project_simplex

CASE_COUNT: int = 16
ORIGIN: pd.Timestamp = pd.Timestamp("2025-01-01", tz="UTC")


def event_frame() -> pd.DataFrame:
    """Create a deterministic test log.

    :return: Sixteen completed three-event cases.
    """
    rows: list[dict[str, str | pd.Timestamp | float]] = [
        {
            "case:concept:name": f"{case:03d}",
            "concept:name": activity,
            "time:timestamp": ORIGIN + pd.Timedelta(days=case, seconds=event * 60),
            "cost": float(case + event),
            "customer": str(case % 2),
        }
        for case in range(CASE_COUNT)
        for event, activity in enumerate(("A", "B", "C"))
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[inference.Request]:
    """Capture input mappings and return final-history values for decoder tests.

    :param monkeypatch: Scoped replacement helper.
    :return: Captured requests; this fixture does not test model inference.
    """
    captured: list[inference.Request] = []

    def forecast(request: inference.Request) -> NDArray[np.float64]:
        """Record one request and expose its last target column.

        :param request: Encoded history.
        :return: Deterministic scores for unit tests only.
        """
        captured.append(request)
        return np.atleast_2d(request["target"])[:, -1].astype(float)

    monkeypatch.setattr(inference, "forecast", forecast)
    return captured


def test_dfg_and_distributions(requests: list[inference.Request]) -> None:
    """Preserve PM4Py key structures and benchmark probability targets.

    :param requests: Captured input mappings.
    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    graph: dict[Edge, float]
    starts: dict[str, float]
    ends: dict[str, float]
    graph, starts, ends = cpm.discover_dfg(frame)
    assert graph == {("A", "B"): 0.5, ("B", "C"): 0.5}
    assert starts == {"A": 1.0} and ends == {"C": 1.0}
    assert cpm.discover_dfg_edges(frame) == set(graph)
    assert cpm.get_event_attribute_values(frame) == pytest.approx(
        dict.fromkeys("ABC", 1 / 3)
    )
    assert cpm.get_variants_as_tuples(frame) == {("A", "B", "C"): 1.0}
    np.testing.assert_allclose(requests[0]["target"], np.full((2, 8), 0.5))


def test_native_performance_and_footprints(requests: list[inference.Request]) -> None:
    """Match native target constructors on a stationary log.

    :param requests: Captured input mappings.
    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    graph: dict[Edge, float]
    native: dict[Edge, float]
    graph, _, _ = cpm.discover_performance_dfg(frame)
    native, _, _ = pm4py.discover_performance_dfg(frame, perf_aggregation_key="mean")
    assert graph == native
    assert cpm.discover_temporal_profile(frame) == pm4py.discover_temporal_profile(
        frame
    )
    assert cpm.get_performance_bottlenecks(frame) == sorted(
        graph.items(), key=lambda item: (-item[1], item[0])
    )
    footprint: cpm.Footprint = cpm.discover_footprints(frame)
    assert footprint["activities"] == set("ABC")
    assert footprint["sequence"] == set(graph) and footprint["parallel"] == set()


def test_prefix_predictions(requests: list[inference.Request]) -> None:
    """Use only observed prefixes and correctly align time covariates.

    :param requests: Captured input mappings.
    :return: None.
    """
    context: EventLog = pm4py.convert_to_event_log(event_frame())
    prefix: Trace = Trace(context[-1][:2])
    history: EventLog = EventLog(context[:-1])
    assert cpm.predict_next_activity(history, prefix, covariates="selected") == "B"
    assert requests[-1]["future_covariates"] == {}
    assert cpm.predict_next_event_time(history, prefix, covariates="global") == 60.0
    request: inference.Request = requests[-1]
    np.testing.assert_array_equal(request["past_covariates"]["activity"], ["A"])
    np.testing.assert_array_equal(request["future_covariates"]["activity"], ["B"])
    assert cpm.predict_remaining_time(history, prefix, covariates="global") == 60.0
    assert requests[-1]["target"].shape == (CASE_COUNT - 1,)
    assert requests[-1]["future_covariates"]["activity_prefix"].tolist() == [
        '["A","B"]'
    ]


def test_explicit_covariates(requests: list[inference.Request]) -> None:
    """Forward aligned numeric and categorical caller covariates.

    :param requests: Captured input mappings.
    :return: None.
    """
    cpm.get_start_activities(
        event_frame(),
        past_covariates={"workload": list(range(8))},
        future_covariates={"workload": [9]},
    )
    np.testing.assert_array_equal(requests[-1]["future_covariates"]["workload"], [9])
    with pytest.raises(ValueError, match="aligned"):
        cpm.get_start_activities(event_frame(), past_covariates={"wrong": [1]})
    with pytest.raises(ValueError, match="subset"):
        cpm.get_start_activities(event_frame(), future_covariates={"unknown": [1]})


def test_input_copy_custom_keys_and_windows() -> None:
    """Accept both PM4Py inputs without mutating them or losing remainder cases.

    :return: None.
    """
    frame: pd.DataFrame = event_frame().rename(
        columns={
            "case:concept:name": "case",
            "concept:name": "activity",
            "time:timestamp": "time",
        }
    )
    before: pd.DataFrame = frame.copy(deep=True)
    log: EventLog = normalize_log(
        frame, activity_key="activity", timestamp_key="time", case_id_key="case"
    )
    pd.testing.assert_frame_equal(frame, before)
    assert len(log) == CASE_COUNT
    assert [len(window) for window in context_windows(log, 8)] == [2] * 8
    with pytest.raises(ValueError, match="divisible"):
        context_windows(EventLog(log[:-1]), 8)
    with pytest.raises(ValueError):
        context_windows(log, 0)


def test_reject_future_context_and_short_prefix(
    requests: list[inference.Request],
) -> None:
    """Reject context leakage and an unobservable inter-event history.

    :param requests: Captured input mappings.
    :return: None.
    """
    log: EventLog = pm4py.convert_to_event_log(event_frame())
    with pytest.raises(ValueError, match="strictly"):
        cpm.predict_remaining_time(log, Trace(log[-1][:2]))
    with pytest.raises(ValueError, match="two"):
        cpm.predict_next_event_time(EventLog(log[:-1]), Trace(log[-1][:1]))
    with pytest.raises(ValueError):
        cpm.get_start_activities(
            event_frame(), covariates=cast(cpm.CovariateSet, "typo")
        )


def test_ebi_statistics(requests: list[inference.Request]) -> None:
    """Exercise real EBI scalar extraction, including rational results.

    :param requests: Captured input mappings.
    :return: None.
    """
    assert cpm.analyse_entropy(event_frame()) == pytest.approx(0.0)
    assert np.isfinite(cpm.analyse_variety(event_frame()))
    assert np.isfinite(cpm.analyse_completeness(event_frame()))


def test_aggregate_covariates(requests: list[inference.Request]) -> None:
    """Require a known origin for automatic future calendar information.

    :param requests: Captured input mappings.
    :return: None.
    """
    options: ForecastOptions = {
        "forecast_origin": ORIGIN + pd.Timedelta(days=CASE_COUNT)
    }
    cpm.get_start_activities(event_frame(), covariates="global", **options)
    request: inference.Request = requests[-1]
    assert "case_attribute::customer" in request["past_covariates"]
    assert "event_attribute::cost" in request["past_covariates"]
    assert len(request["future_covariates"]) == 6
    assert all(value.shape == (8,) for value in request["past_covariates"].values())
    with pytest.raises(ValueError, match="forecast_origin"):
        cpm.get_start_activities(event_frame(), covariates="selected")


def test_footprint_channel_order_matches_benchmark() -> None:
    """Preserve none/sequence/parallel channel order, including tie-breaking.

    :return: None.
    """
    keys: list[Key]
    keys, _ = history(context_windows(normalize_log(event_frame()), 8), "footprints")
    assert keys[:3] == [
        ("A", "A", "none"),
        ("A", "A", "sequence"),
        ("A", "A", "parallel"),
    ]


def test_simplex_is_euclidean_projection() -> None:
    """Retain the scientific projection instead of clipping and renormalizing.

    :return: None.
    """
    np.testing.assert_allclose(
        project_simplex(np.asarray([0.8, 0.4, -0.2])), [0.7, 0.3, 0.0]
    )
    np.testing.assert_allclose(project_simplex(np.asarray([-2.0, -2.0])), [0.5, 0.5])
    with pytest.raises(ValueError):
        project_simplex(np.asarray([np.nan]))


def test_durations_are_not_silently_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve raw benchmark medians even when predictions are negative.

    :param monkeypatch: Scoped replacement helper.
    :return: None.
    """

    def negative_forecast(request: inference.Request) -> NDArray[np.float64]:
        """Return negative decoder-test scores.

        :param request: Target history.
        :return: One negative value per channel.
        """
        return np.full(np.atleast_2d(request["target"]).shape[0], -1.0)

    monkeypatch.setattr(inference, "forecast", negative_forecast)
    profile: dict[tuple[str, str], tuple[float, float]] = cpm.discover_temporal_profile(
        event_frame()
    )
    assert all(value == (-1.0, -1.0) for value in profile.values())


def test_missing_edges_remain_nan_for_duration_histories() -> None:
    """Represent absent duration channels as missing rather than zero time.

    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    frame.loc[frame["case:concept:name"] < "008", "concept:name"] = frame.loc[
        frame["case:concept:name"] < "008", "concept:name"
    ].replace({"B": "D"})
    keys: list[Key]
    values: FloatArray
    keys, values = history(context_windows(normalize_log(frame), 8), "performance")
    row: NDArray[np.float64] = values[keys.index(("A", "B"))]
    assert np.isnan(row[:4]).all()
    np.testing.assert_array_equal(row[4:], [60.0] * 4)


def test_no_future_next_activity_and_no_covariate_overwrites(
    requests: list[inference.Request],
) -> None:
    """Reject unknown successor values and accidental overrides.

    :param requests: Captured input mappings.
    :return: None.
    """
    log: EventLog = normalize_log(event_frame())
    with pytest.raises(ValueError, match="no known future"):
        cpm.predict_next_activity(
            EventLog(log[:-1]),
            Trace(log[-1][:2]),
            past_covariates={"x": [1, 2]},
            future_covariates={"x": [3]},
        )
    with pytest.raises(ValueError, match="duplicate"):
        cpm.get_start_activities(
            event_frame(),
            covariates="selected",
            forecast_origin=ORIGIN + pd.Timedelta(days=CASE_COUNT),
            past_covariates={"mean_prefix_length": [3] * 8},
        )


def test_bad_logs_and_exhausted_prefix_support(
    requests: list[inference.Request],
) -> None:
    """Reject malformed input and absent fixed-position history.

    :param requests: Captured input mappings.
    :return: None.
    """
    with pytest.raises(ValueError):
        normalize_log(EventLog())
    with pytest.raises(ValueError):
        normalize_log(event_frame().drop(columns=["time:timestamp"]))
    log: EventLog = normalize_log(event_frame())
    with pytest.raises(ValueError, match="reaches"):
        cpm.predict_remaining_time(EventLog([Trace(log[0][:1])]), Trace(log[-1][:2]))


def test_unsupported_options_are_rejected(requests: list[inference.Request]) -> None:
    """Do not silently accept a different native aggregation method.

    :param requests: Captured input mappings.
    :return: None.
    """
    wrong_options: ForecastOptions = cast(
        ForecastOptions, {"perf_aggregation_key": "median"}
    )
    with pytest.raises(TypeError, match="unsupported"):
        cpm.discover_performance_dfg(event_frame(), **wrong_options)


def test_ebi_receives_only_benchmark_xes_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip unrelated attributes before the EBI boundary as the benchmark does.

    :param monkeypatch: Scoped replacement helper.
    :return: None.
    """
    import ebi

    from chronos4pm.tasks import ebi_statistic

    def inspect_log(log: EventLog) -> float:
        """Verify the exact EBI input contract independently of model inference.

        :param log: Adapter output.
        :return: Test-only scalar.
        """
        assert all(
            set(event) == {"concept:name", "time:timestamp"}
            for trace in log
            for event in trace
        )
        assert all(set(trace.attributes) == {"concept:name"} for trace in log)
        return 0.0

    monkeypatch.setattr(ebi, "analyse_entropy", inspect_log)
    assert ebi_statistic(normalize_log(event_frame()), "entropy") == 0.0


def test_calendar_preserves_supplied_timezone(
    requests: list[inference.Request],
) -> None:
    """Preserve the benchmark's local-hour interpretation of observed timestamps.

    :param requests: Captured input mappings.
    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    frame["time:timestamp"] = frame["time:timestamp"].dt.tz_convert("Europe/Berlin")
    origin: pd.Timestamp = (ORIGIN + pd.Timedelta(days=CASE_COUNT)).tz_convert(
        "Europe/Berlin"
    )
    cpm.get_start_activities(frame, covariates="selected", forecast_origin=origin)
    expected: float = float(np.sin(2 * np.pi / 24))
    np.testing.assert_allclose(
        cast(FloatArray, requests[-1]["past_covariates"]["calendar_hour_sin"]), expected
    )
    np.testing.assert_allclose(
        cast(FloatArray, requests[-1]["future_covariates"]["calendar_hour_sin"]),
        expected,
    )


def test_dataframe_case_columns_stay_event_observed(
    requests: list[inference.Request],
) -> None:
    """Keep event-carried case attributes without promoting future values to metadata.

    :param requests: Captured input mappings.
    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    frame["case:customer"] = frame["customer"]
    frame.loc[frame.index % 3 == 0, "case:customer"] = None
    log: EventLog = normalize_log(frame)
    assert "case:customer" in log[0][1]
    cpm.predict_next_activity(
        EventLog(log[:-1]),
        Trace(log[-1][:2]),
        covariates="global",
        attribute_allowlist=["case:customer"],
    )
    np.testing.assert_array_equal(
        requests[-1]["past_covariates"]["case_attribute::case:customer"],
        ["__missing__", "1"],
    )
