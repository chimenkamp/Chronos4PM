"""Native target histories and the benchmark's forecast decoders."""

import re
from collections.abc import Callable
from fractions import Fraction
from itertools import product
from typing import cast

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.discovery.footprints import algorithm as footprints
from pm4py.objects.log.obj import EventLog

from . import inference
from .configuration import (
    ACTIVITY_KEY,
    CASE_ID_KEY,
    RELATIONS,
    TIMESTAMP_KEY,
    WINDOW_COUNT,
    CovariateSet,
    Edge,
    FloatArray,
    Footprint,
    ForecastOptions,
    Key,
    Log,
    Series,
    Task,
)
from .covariates import Attribute, aggregate_covariates, attribute_schema, families
from .logs import context_windows, ebi_log, normalize_log, require_prior_context

APPROXIMATION: re.Pattern[str] = re.compile(
    r"Approximately\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
DISTRIBUTION_TASKS: frozenset[Task] = frozenset(
    {"dfg", "start", "end", "activity", "variants"}
)
PERFORMANCE_TASKS: frozenset[Task] = frozenset(
    {"performance", "bottlenecks", "temporal"}
)
SCALAR_TASKS: frozenset[Task] = frozenset({"entropy", "variety", "completeness"})


def ebi_statistic(log: EventLog, task: Task) -> float:
    """Call the pinned EBI estimator and parse its exact or approximate result.

    :param log: Canonical PM4Py log accepted by EBI.
    :param task: Entropy, variety or completeness.
    :return: Finite scalar in native EBI units.
    """
    import ebi

    estimators: dict[Task, Callable[[EventLog], str | float | list[float]]] = {
        "entropy": ebi.analyse_entropy,
        "variety": ebi.analyse_variety,
        "completeness": ebi.analyse_completeness,
    }
    result: str | float | list[float] = estimators[task](ebi_log(log))
    raw: str = str(result[0] if isinstance(result, list) else result)
    match: re.Match[str] | None = APPROXIMATION.search(raw)
    value: float = float(match.group(1)) if match else float(Fraction(raw.strip()))
    if not np.isfinite(value):
        raise ValueError(f"EBI returned a non-finite {task} value")
    return value


def extract(log: EventLog, task: Task, activities: list[str]) -> dict[Key, float]:
    """Construct one window's target with the benchmark's native library call.

    :param log: One completed-case window.
    :param task: Descriptor or statistic to extract.
    :param activities: Context-only activity support for footprint channels.
    :return: Unnormalized native values on process-element keys.
    """
    if task in SCALAR_TASKS:
        return {"value": ebi_statistic(log, task)}
    if task == "temporal":
        profile: dict[Edge, tuple[float, float]] = pm4py.discover_temporal_profile(log)
        return {
            (edge, component): float(values[component])
            for edge, values in profile.items()
            for component in (0, 1)
        }
    if task == "footprints":
        footprint: Footprint = cast(
            Footprint,
            footprints.apply(log, variant=footprints.Variants.ENTIRE_EVENT_LOG),
        )
        return {
            (source, target, relation): float(
                relation == relation_for(footprint, (source, target))
            )
            for source, target in product(activities, repeat=2)
            for relation in RELATIONS
        }
    counts: dict[Key, float]
    if task in ("dfg", "edges"):
        counts = pm4py.discover_dfg(log)[0]
    elif task in ("performance", "bottlenecks"):
        counts = pm4py.discover_performance_dfg(log, perf_aggregation_key="mean")[0]
    elif task == "start":
        counts = pm4py.get_start_activities(log)
    elif task == "end":
        counts = pm4py.get_end_activities(log)
    elif task == "activity":
        counts = pm4py.get_event_attribute_values(log, ACTIVITY_KEY)
    elif task == "variants":
        variants: dict[tuple[str, ...], list[object]] = pm4py.get_variants_as_tuples(
            log
        )
        counts = {variant: float(len(traces)) for variant, traces in variants.items()}
    else:
        raise ValueError(f"unsupported aggregate task: {task}")
    return {key: float(value) for key, value in counts.items()}


def relation_for(footprint: Footprint, pair: Edge) -> str:
    """Apply the benchmark's parallel-before-sequence footprint precedence.

    :param footprint: Native footprint.
    :param pair: Ordered activity pair.
    :return: One of none, sequence, parallel.
    """
    return (
        "parallel"
        if pair in footprint["parallel"]
        else "sequence"
        if pair in footprint["sequence"]
        else "none"
    )


def history(windows: list[EventLog], task: Task) -> tuple[list[Key], FloatArray]:
    """Build scalar, frequency, presence or duration target channels.

    :param windows: Equal-size context windows in chronological order.
    :param task: Benchmark target mapping.
    :return: Ordered process-element keys and their target matrix.
    """
    activities: list[str] = sorted(
        {
            str(event[ACTIVITY_KEY])
            for window in windows
            for trace in window
            for event in trace
        }
    )
    rows: list[dict[Key, float]] = [
        extract(window, task, activities) for window in windows
    ]
    keys: list[Key] = (
        list(rows[0])
        if task == "footprints"
        else sorted({key for row in rows for key in row})
    )
    if not keys:
        raise ValueError(f"context windows contain no {task} support")
    missing: float = np.nan if task in PERFORMANCE_TASKS else 0.0
    values: FloatArray = np.asarray(
        [[row.get(key, missing) for row in rows] for key in keys], dtype=float
    )
    if task in DISTRIBUTION_TASKS:
        totals: FloatArray = values.sum(axis=0)
        if np.any(totals <= 0):
            raise ValueError(
                "each context window must contain positive distribution mass"
            )
        values = values / totals
    if task == "edges":
        values = (values > 0).astype(float)
    return keys, values


def project_simplex(values: FloatArray) -> FloatArray:
    """Apply the benchmark's Euclidean projection onto the unit simplex.

    :param values: Finite, non-empty category scores.
    :return: Nonnegative probabilities that sum to one.
    """
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("simplex projection requires a finite non-empty vector")
    ordered: FloatArray = np.sort(values)[::-1]
    offsets: FloatArray = (np.cumsum(ordered) - 1.0) / np.arange(1, values.size + 1)
    threshold: float = float(offsets[np.flatnonzero(ordered > offsets)[-1]])
    projected: FloatArray = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def merge_covariates(
    automatic: dict[str, Series], explicit: dict[str, Series]
) -> dict[str, Series]:
    """Reject ambiguous duplicate names when adding caller-provided covariates.

    :param automatic: Automatically derived covariates.
    :param explicit: Validated caller-provided covariates.
    :return: Combined series.
    """
    if automatic.keys() & explicit.keys():
        raise ValueError(
            "explicit covariates must not duplicate automatic covariate names"
        )
    return automatic | explicit


def aggregate(
    log: Log, task: Task, covariates: CovariateSet, options: ForecastOptions
) -> dict[Key, float]:
    """Orchestrate native history extraction, covariates and one-step decoding.

    :param log: Completed historical cases.
    :param task: Aggregate benchmark task.
    :param covariates: Automatic covariate selection.
    :param options: Window, key and explicit-covariate parameters.
    :return: Forecast per retained process element.
    """
    unknown: set[str] = options.keys() - ForecastOptions.__annotations__.keys()
    if unknown:
        raise TypeError(f"unsupported forecast options: {sorted(unknown)}")
    active: frozenset[str] = families(task, covariates)
    context: EventLog = normalize_log(
        log,
        activity_key=options.get("activity_key", ACTIVITY_KEY),
        timestamp_key=options.get("timestamp_key", TIMESTAMP_KEY),
        case_id_key=options.get("case_id_key", CASE_ID_KEY),
    )
    windows: list[EventLog] = context_windows(
        context, options.get("n_windows", WINDOW_COUNT)
    )
    origin: pd.Timestamp | None = (
        pd.Timestamp(options["forecast_origin"])
        if "forecast_origin" in options
        else None
    )
    if origin is not None:
        require_prior_context(context, origin)
    schema: list[Attribute] = (
        attribute_schema(context, options.get("attribute_allowlist"))
        if {"case_attributes", "event_attributes"} & active
        else []
    )
    keys: list[Key]
    target: FloatArray
    keys, target = history(windows, task)
    past: dict[str, Series]
    future: dict[str, Series]
    past, future = aggregate_covariates(windows, task, covariates, origin, schema)
    past = merge_covariates(
        past,
        inference.aligned_covariates(options.get("past_covariates", {}), len(windows)),
    )
    future = merge_covariates(
        future,
        inference.aligned_covariates(
            options.get("future_covariates", {}), inference.PREDICTION_LENGTH
        ),
    )
    request: inference.Request = inference.build_request(
        target[0] if task in SCALAR_TASKS else target, past, future
    )
    scores: FloatArray = inference.forecast(request)
    if task in DISTRIBUTION_TASKS:
        scores = project_simplex(scores)
    return {key: float(value) for key, value in zip(keys, scores, strict=True)}
