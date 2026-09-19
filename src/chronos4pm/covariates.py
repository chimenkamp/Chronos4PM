"""Benchmark covariates built only from completed cases and observed events."""

import json
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import cast

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace

from .configuration import (
    ACTIVITY_KEY,
    CASE_ID_KEY,
    TIMESTAMP_KEY,
    CovariateSet,
    FloatArray,
    Series,
    Task,
)

MISSING_CATEGORY: str = "__missing__"
SECONDS_PER_HOUR: float = 3600.0
MINUTES_PER_HOUR: float = 60.0
HOURS_PER_DAY: float = 24.0
DAYS_PER_WEEK: float = 7.0
DAYS_PER_YEAR: float = 365.25
TWO_PI: float = 2 * np.pi
AGGREGATE_FAMILIES: tuple[str, ...] = (
    "prefix_length",
    "calendar_features",
    "task_specific_time_series",
)
TIMING_FAMILIES: tuple[str, ...] = ("elapsed_case_time", "time_since_previous_event")
ATTRIBUTE_FAMILIES: tuple[str, ...] = ("case_attributes", "event_attributes")
PERFORMANCE_TASKS: frozenset[Task] = frozenset(
    {"performance", "bottlenecks", "temporal"}
)
PREFIX_SELECTED: dict[Task, tuple[str, ...]] = {
    "next_activity": ("prefix_length", *TIMING_FAMILIES, "calendar_features"),
    "next_time": (
        "prefix_length",
        "activity_prefix",
        "last_activity",
        "elapsed_case_time",
        "calendar_features",
    ),
    "remaining_time": (
        "prefix_length",
        "last_activity",
        *TIMING_FAMILIES,
        "calendar_features",
    ),
}
TASK_SERIES: tuple[str, ...] = (
    "event_count",
    "unique_activity_count",
    "trace_variant_count",
    "dfg_edge_count",
)
SELECTED_SERIES: dict[Task, tuple[str, ...]] = {
    "dfg": TASK_SERIES[:2],
    "edges": (TASK_SERIES[1], TASK_SERIES[3]),
    "start": (TASK_SERIES[1],),
    "end": (TASK_SERIES[1],),
    "activity": (TASK_SERIES[0],),
    "variants": (TASK_SERIES[1], TASK_SERIES[2]),
    "footprints": (TASK_SERIES[1], TASK_SERIES[3]),
    "performance": (TASK_SERIES[0], TASK_SERIES[3]),
    "bottlenecks": (TASK_SERIES[0], TASK_SERIES[3]),
    "temporal": (TASK_SERIES[0], TASK_SERIES[3]),
    "entropy": (TASK_SERIES[0], TASK_SERIES[2]),
    "variety": (TASK_SERIES[0], TASK_SERIES[2]),
    "completeness": (TASK_SERIES[0], TASK_SERIES[2]),
}


@dataclass(frozen=True)
class Attribute:
    """A strict-context event attribute's inferred role and encoding."""

    key: str
    numeric: bool
    case_level: bool


def families(task: Task, selection: CovariateSet) -> frozenset[str]:
    """Resolve the benchmark's task-dependent covariate sets.

    :param task: Forecast task.
    :param selection: Target-only, selected, or global arm.
    :return: Applicable family names.
    """
    if selection not in ("target_only", "selected", "global"):
        raise ValueError("covariates must be 'target_only', 'selected', or 'global'")
    if selection == "target_only":
        return frozenset()
    if task in PREFIX_SELECTED:
        selected: tuple[str, ...] = PREFIX_SELECTED[task]
        extra: tuple[str, ...] = ATTRIBUTE_FAMILIES
        if task == "next_time":
            extra += ("time_since_previous_event",)
        if task == "remaining_time":
            extra += ("activity_prefix",)
    else:
        selected = (
            (*TIMING_FAMILIES, *AGGREGATE_FAMILIES[1:])
            if task in PERFORMANCE_TASKS
            else AGGREGATE_FAMILIES
        )
        extra = (*AGGREGATE_FAMILIES, *TIMING_FAMILIES, *ATTRIBUTE_FAMILIES)
    return frozenset(selected if selection == "selected" else selected + extra)


def calendar(timestamps: Sequence[pd.Timestamp]) -> dict[str, Series]:
    """Encode the benchmark's hour, weekday and day-of-year cycles.

    :param timestamps: Observed or known-origin timestamps.
    :return: Six numeric sine/cosine series.
    """
    phases: dict[str, FloatArray] = {
        "hour": np.asarray(
            [
                (t.hour + t.minute / MINUTES_PER_HOUR + t.second / SECONDS_PER_HOUR)
                / HOURS_PER_DAY
                for t in timestamps
            ]
        ),
        "weekday": np.asarray([t.dayofweek / DAYS_PER_WEEK for t in timestamps]),
        "day_of_year": np.asarray(
            [(t.dayofyear - 1) / DAYS_PER_YEAR for t in timestamps]
        ),
    }
    return {
        f"calendar_{name}_{suffix}": transform(TWO_PI * values)
        for name, values in phases.items()
        for suffix, transform in (("sin", np.sin), ("cos", np.cos))
    }


def missing(value: object) -> bool:
    """Handle missing heterogeneous PM4Py attribute values.

    :param value: Native event attribute, possibly nested.
    :return: Whether the scalar is missing.
    """
    result: Series = np.asarray(pd.isna(np.asarray(value, dtype=object)), dtype=bool)
    return bool(result) if result.ndim == 0 else False


def hashable(value: object) -> Hashable:
    """Preserve native equality for scalar attributes and represent nested ones.

    :param value: Native event attribute.
    :return: Comparable attribute identity.
    """
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def attribute_schema(log: EventLog, allowlist: Sequence[str] | None) -> list[Attribute]:
    """Infer attribute roles exclusively from strict context events.

    :param log: Completed context cases.
    :param allowlist: Optional prediction-time-available attribute names.
    :return: Useful case/event attributes and their numeric flags.
    """
    keys: set[str] = {str(key) for trace in log for event in trace for key in event} - {
        ACTIVITY_KEY,
        TIMESTAMP_KEY,
        CASE_ID_KEY,
    }
    if allowlist is not None:
        keys.intersection_update(allowlist)
    schema: list[Attribute] = []
    for key in sorted(keys):
        values: list[object] = [
            event[key]
            for trace in log
            for event in trace
            if key in event and not missing(event[key])
        ]
        if len(values) > 1:
            numeric: bool = all(
                isinstance(value, Number) and not isinstance(value, (bool, np.bool_))
                for value in values
            )
            stable: bool = all(
                len(
                    {
                        hashable(event[key])
                        for event in trace
                        if key in event and not missing(event[key])
                    }
                )
                <= 1
                for trace in log
            )
            schema.append(Attribute(key, numeric, stable))
    return schema


def observed_attribute(trace: Trace, attribute: Attribute) -> list[object]:
    """Forward-fill case attributes only after their first observed value.

    :param trace: Observed events only; trace metadata is excluded.
    :param attribute: Attribute role.
    :return: Event-aligned values.
    """
    latest: object = None
    values: list[object] = []
    for event in trace:
        value: object = event.get(attribute.key)
        if not attribute.case_level or not missing(value):
            latest = value
        values.append(latest)
    return values


def encode(values: Sequence[object], numeric: bool) -> Series:
    """Convert native values into numerical or categorical Chronos inputs.

    :param values: Heterogeneous native attributes.
    :param numeric: Strict-context numeric flag.
    :return: Numerical values with NaN, or strings with a missing category.
    """
    if numeric:
        return np.asarray(
            [
                np.nan if missing(value) else float(cast(float, value))
                for value in values
            ],
            dtype=float,
        )
    return np.asarray(
        [MISSING_CATEGORY if missing(value) else str(value) for value in values]
    )


def attribute_name(attribute: Attribute) -> str:
    """Name one attribute exactly as in the benchmark.

    :param attribute: Inferred attribute.
    :return: Namespaced covariate name.
    """
    return f"{'case' if attribute.case_level else 'event'}_attribute::{attribute.key}"


def active_attributes(
    schema: list[Attribute], active: frozenset[str]
) -> list[Attribute]:
    """Select only enabled attribute families.

    :param schema: Strict-context attributes.
    :param active: Enabled covariate families.
    :return: Selected attributes.
    """
    return [
        attribute
        for attribute in schema
        if ("case_attributes" if attribute.case_level else "event_attributes") in active
    ]


def event_covariates(
    trace: Trace, active: frozenset[str], schema: list[Attribute]
) -> dict[str, Series]:
    """Build event-aligned covariates from a fully observed prefix.

    :param trace: Events observed so far.
    :param active: Enabled covariate families.
    :param schema: Strict-context attribute roles.
    :return: One value per observed event for each enabled covariate.
    """
    timestamps: list[pd.Timestamp] = [event[TIMESTAMP_KEY] for event in trace]
    elapsed: Series = np.asarray(
        [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]
    )
    result: dict[str, Series] = {}
    if "prefix_length" in active:
        result["prefix_length"] = np.arange(1, len(trace) + 1, dtype=float)
    if {"activity_prefix", "last_activity"} & active:
        result["activity"] = np.asarray([event[ACTIVITY_KEY] for event in trace])
    if "elapsed_case_time" in active:
        result["elapsed_case_seconds"] = elapsed
    if "time_since_previous_event" in active:
        result["time_since_previous_event_seconds"] = np.diff(elapsed, prepend=0.0)
    if "calendar_features" in active:
        result.update(calendar(timestamps))
    for attribute in active_attributes(schema, active):
        result[attribute_name(attribute)] = encode(
            observed_attribute(trace, attribute), attribute.numeric
        )
    return result


def case_covariates(
    prefixes: Sequence[Trace], active: frozenset[str], schema: list[Attribute]
) -> dict[str, Series]:
    """Align remaining-time covariates across completed prior case prefixes.

    :param prefixes: Equal-length observed prefixes from separate cases.
    :param active: Enabled covariate families.
    :param schema: Strict-context attribute roles.
    :return: One covariate value per case.
    """
    rows: list[dict[str, Series]] = [
        event_covariates(prefix, active, schema) for prefix in prefixes
    ]
    result: dict[str, Series] = {
        name: np.asarray([row[name][-1] for row in rows]) for name in rows[0]
    }
    if "activity_prefix" in active:
        result["activity_prefix"] = np.asarray(
            [
                json.dumps(
                    [event[ACTIVITY_KEY] for event in prefix],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for prefix in prefixes
            ]
        )
    if "last_activity" not in active:
        result.pop("activity", None)
    return result


def summarize_attribute(window: EventLog, attribute: Attribute) -> object:
    """Compute the benchmark's mean or deterministic categorical mode.

    :param window: Completed cases in one window.
    :param attribute: Inferred role and numeric flag.
    :return: Window summary with one vote per case for stable attributes.
    """
    values: list[object] = []
    for trace in window:
        observed: list[object] = [
            event[attribute.key]
            for event in trace
            if attribute.key in event and not missing(event[attribute.key])
        ]
        values.extend(observed[-1:] if attribute.case_level else observed)
    if not values:
        return np.nan if attribute.numeric else MISSING_CATEGORY
    if attribute.numeric:
        return float(np.mean(np.asarray(values, dtype=float)))
    counts: Counter[str] = Counter(str(value) for value in values)
    return min(counts, key=lambda value: (-counts[value], value))


def window_series(window: EventLog) -> dict[str, float]:
    """Extract shared structural and timing covariates from one window.

    :param window: Completed cases.
    :return: Benchmark aggregate summaries.
    """
    variants: list[tuple[str, ...]] = [
        tuple(event[ACTIVITY_KEY] for event in trace) for trace in window
    ]
    intervals: list[float] = [
        (b[TIMESTAMP_KEY] - a[TIMESTAMP_KEY]).total_seconds()
        for trace in window
        for a, b in zip(trace, trace[1:])
    ]
    return {
        "mean_prefix_length": float(np.mean([len(trace) for trace in window])),
        "mean_case_elapsed_seconds": float(
            np.mean(
                [
                    (trace[-1][TIMESTAMP_KEY] - trace[0][TIMESTAMP_KEY]).total_seconds()
                    for trace in window
                ]
            )
        ),
        "mean_inter_event_seconds": float(np.mean(intervals)) if intervals else 0.0,
        "event_count": float(sum(map(len, window))),
        "unique_activity_count": float(
            len({activity for variant in variants for activity in variant})
        ),
        "trace_variant_count": float(len(set(variants))),
        "dfg_edge_count": float(
            len({pair for variant in variants for pair in zip(variant, variant[1:])})
        ),
    }


def aggregate_covariates(
    windows: list[EventLog],
    task: Task,
    selection: CovariateSet,
    origin: pd.Timestamp | None,
    schema: list[Attribute],
) -> tuple[dict[str, Series], dict[str, Series]]:
    """Build benchmark covariates for equally sized completed-case windows.

    :param windows: Chronological context windows.
    :param task: Forecast task.
    :param selection: Covariate arm.
    :param origin: Known forecast timestamp, required for calendar covariates.
    :param schema: Strict-context attribute roles.
    :return: Past and known-origin future series.
    """
    active: frozenset[str] = families(task, selection)
    past: dict[str, Series] = {}
    future: dict[str, Series] = {}
    if not active:
        return past, future
    rows: list[dict[str, float]] = [window_series(window) for window in windows]
    for family, name in (
        ("prefix_length", "mean_prefix_length"),
        ("elapsed_case_time", "mean_case_elapsed_seconds"),
        ("time_since_previous_event", "mean_inter_event_seconds"),
    ):
        if family in active:
            past[name] = np.asarray([row[name] for row in rows])
    if "task_specific_time_series" in active:
        names: tuple[str, ...] = (
            SELECTED_SERIES[task] if selection == "selected" else TASK_SERIES
        )
        past.update(
            {f"task::{name}": np.asarray([row[name] for row in rows]) for name in names}
        )
    if "calendar_features" in active:
        if origin is None:
            raise ValueError(
                "forecast_origin is required for automatic calendar covariates"
            )
        past.update(calendar([window[0][0][TIMESTAMP_KEY] for window in windows]))
        future.update(calendar([origin]))
    for attribute in active_attributes(schema, active):
        past[attribute_name(attribute)] = encode(
            [summarize_attribute(window, attribute) for window in windows],
            attribute.numeric,
        )
    return past, future
