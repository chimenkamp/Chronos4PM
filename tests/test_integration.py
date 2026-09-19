"""Real inference checks, enabled explicitly because checkpoints can be large."""

import os

import numpy as np
import pandas as pd
import pm4py
import pytest
from pm4py.objects.log.obj import EventLog, Trace
from test_contract import CASE_COUNT, ORIGIN, event_frame

import chronos4pm as cpm
from chronos4pm.configuration import Edge, ForecastOptions


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_CHRONOS_INTEGRATION") != "1",
    reason="set RUN_CHRONOS_INTEGRATION=1 to load the real checkpoint",
)
@pytest.mark.parametrize("selection", ["target_only", "selected", "global"])
def test_all_tasks_with_real_checkpoint(selection: cpm.CovariateSet) -> None:
    """Exercise all sixteen tasks and all three covariate arms on the real model.

    :param selection: Benchmark covariate arm.
    :return: None.
    """
    frame: pd.DataFrame = event_frame()
    options: ForecastOptions = {
        "forecast_origin": ORIGIN + pd.Timedelta(days=CASE_COUNT)
    }
    graph: dict[Edge, float]
    starts: dict[str, float]
    ends: dict[str, float]
    graph, starts, ends = cpm.discover_dfg(frame, covariates=selection, **options)
    assert sum(graph.values()) == pytest.approx(1.0)
    assert sum(starts.values()) == pytest.approx(1.0)
    assert sum(ends.values()) == pytest.approx(1.0)
    assert cpm.discover_dfg_edges(frame, covariates=selection, **options) <= set(graph)
    assert sum(
        cpm.get_event_attribute_values(frame, covariates=selection, **options).values()
    ) == pytest.approx(1.0)
    assert sum(
        cpm.get_variants_as_tuples(frame, covariates=selection, **options).values()
    ) == pytest.approx(1.0)
    assert cpm.discover_footprints(frame, covariates=selection, **options)[
        "activities"
    ] == set("ABC")
    performance: dict[Edge, float]
    performance, _, _ = cpm.discover_performance_dfg(
        frame, covariates=selection, **options
    )
    assert np.isfinite(list(performance.values())).all()
    assert len(
        cpm.get_performance_bottlenecks(frame, covariates=selection, **options)
    ) == len(graph)
    assert np.isfinite(
        list(
            cpm.discover_temporal_profile(
                frame, covariates=selection, **options
            ).values()
        )
    ).all()
    assert np.isfinite(cpm.analyse_entropy(frame, covariates=selection, **options))
    assert np.isfinite(cpm.analyse_variety(frame, covariates=selection, **options))
    assert np.isfinite(cpm.analyse_completeness(frame, covariates=selection, **options))
    log: EventLog = pm4py.convert_to_event_log(frame)
    context: EventLog = EventLog(log[:-1])
    prefix: Trace = Trace(log[-1][:2])
    assert cpm.predict_next_activity(context, prefix, covariates=selection) in set(
        "ABC"
    )
    assert np.isfinite(
        cpm.predict_next_event_time(context, prefix, covariates=selection)
    )
    assert np.isfinite(
        cpm.predict_remaining_time(context, prefix, covariates=selection)
    )
