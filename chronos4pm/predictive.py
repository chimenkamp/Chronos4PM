"""Single-prefix prediction with the original benchmark target mappings."""

import numpy as np
from pm4py.algo.transformation.log_to_target import algorithm as log_to_target
from pm4py.objects.log.obj import EventLog, Trace

from . import inference
from .configuration import (
    ACTIVITY_KEY,
    CASE_ID_KEY,
    TIMESTAMP_KEY,
    CovariateSet,
    FloatArray,
    Log,
    PrefixOptions,
    Series,
    Task,
)
from .covariates import (
    Attribute,
    attribute_schema,
    case_covariates,
    event_covariates,
    families,
)
from .logs import normalize_log, require_prior_context
from .tasks import merge_covariates


def prefix_inputs(
    log: Log, prefix: Trace | Log, options: PrefixOptions
) -> tuple[EventLog, Trace]:
    """Validate a completed context and one observed case prefix.

    :param log: Completed historical cases.
    :param prefix: One trace, one-case event log, or one-case DataFrame.
    :param options: PM4Py attribute keys.
    :return: Canonical copies with a strict case-start leakage boundary.
    """
    keys: dict[str, str] = {
        "activity_key": options.get("activity_key", ACTIVITY_KEY),
        "timestamp_key": options.get("timestamp_key", TIMESTAMP_KEY),
        "case_id_key": options.get("case_id_key", CASE_ID_KEY),
    }
    context: EventLog = normalize_log(log, **keys)
    observed_log: EventLog = normalize_log(
        EventLog([prefix]) if isinstance(prefix, Trace) else prefix, **keys
    )
    if len(observed_log) != 1:
        raise ValueError("prefix must contain exactly one observed case")
    observed: Trace = observed_log[0]
    require_prior_context(context, observed[0][TIMESTAMP_KEY])
    return context, observed


def prefix_target(
    context: EventLog, prefix: Trace, task: Task
) -> tuple[FloatArray, list[Trace], list[str]]:
    """Encode within-case activity/time or fixed-position remaining-time history.

    :param context: Strictly prior completed cases.
    :param prefix: Observed current case prefix.
    :param task: One predictive task.
    :return: Target history, aligned past prefixes, and activity channel labels.
    """
    if task == "next_activity":
        activities: list[str] = sorted(
            {event[ACTIVITY_KEY] for trace in context for event in trace}
        )
        target: FloatArray = np.asarray(
            [
                [float(event[ACTIVITY_KEY] == activity) for event in prefix]
                for activity in activities
            ]
        )
        return target, [], activities
    if task == "next_time":
        if len(prefix) < 2:
            raise ValueError("next-event time requires at least two observed events")
        rows: list[list[float]]
        rows, _ = log_to_target.apply(
            EventLog([prefix]), variant=log_to_target.Variants.NEXT_TIME
        )
        return np.asarray(rows[0][: len(prefix) - 1], dtype=float), [], []
    eligible: EventLog = EventLog(
        [trace for trace in context if len(trace) >= len(prefix)]
    )
    if not eligible:
        raise ValueError("no completed context case reaches this prefix position")
    remaining: list[list[float]]
    remaining, _ = log_to_target.apply(
        eligible, variant=log_to_target.Variants.REMAINING_TIME
    )
    return (
        np.asarray([row[len(prefix) - 1] for row in remaining], dtype=float),
        [Trace(trace[: len(prefix)]) for trace in eligible],
        [],
    )


def prefix_covariates(
    prefix: Trace,
    histories: list[Trace],
    task: Task,
    active: frozenset[str],
    schema: list[Attribute],
) -> tuple[dict[str, Series], dict[str, Series]]:
    """Align covariates without reading a successor event or timestamp.

    :param prefix: Observed current prefix.
    :param histories: Fixed-position context prefixes for remaining time.
    :param task: Predictive task.
    :param active: Enabled covariate families.
    :param schema: Strict-context attribute roles.
    :return: Past and known-origin future covariates.
    """
    if task == "remaining_time":
        return case_covariates(histories, active, schema), case_covariates(
            [prefix], active, schema
        )
    values: dict[str, Series] = event_covariates(prefix, active, schema)
    if task == "next_activity":
        return values, {}
    return {name: values[:-1] for name, values in values.items()}, {
        name: values[-1:] for name, values in values.items()
    }


def predict(
    log: Log,
    prefix: Trace | Log,
    task: Task,
    covariates: CovariateSet,
    options: PrefixOptions,
) -> str | float:
    """Prepare one prefix and return its decoded Chronos prediction.

    :param log: Strictly prior completed context.
    :param prefix: Events observed so far for one case.
    :param task: Predictive benchmark mapping.
    :param covariates: Automatic covariate selection.
    :param options: Attribute keys and explicit aligned series.
    :return: Next activity label or duration in seconds.
    """
    unknown: set[str] = options.keys() - PrefixOptions.__annotations__.keys()
    if unknown:
        raise TypeError(f"unsupported prefix options: {sorted(unknown)}")
    active: frozenset[str] = families(task, covariates)
    context: EventLog
    observed: Trace
    context, observed = prefix_inputs(log, prefix, options)
    schema: list[Attribute] = (
        attribute_schema(context, options.get("attribute_allowlist"))
        if {"case_attributes", "event_attributes"} & active
        else []
    )
    target: FloatArray
    histories: list[Trace]
    activities: list[str]
    target, histories, activities = prefix_target(context, observed, task)
    past: dict[str, Series]
    future: dict[str, Series]
    past, future = prefix_covariates(observed, histories, task, active, schema)
    past = merge_covariates(
        past,
        inference.aligned_covariates(
            options.get("past_covariates", {}), target.shape[-1]
        ),
    )
    future = merge_covariates(
        future,
        inference.aligned_covariates(
            options.get("future_covariates", {}), inference.PREDICTION_LENGTH
        ),
    )
    if task == "next_activity" and future:
        raise ValueError(
            "next-activity prediction has no known future event covariates"
        )
    request: inference.Request = inference.build_request(target, past, future)
    scores: FloatArray = inference.forecast(request)
    return (
        activities[int(np.argmax(scores))]
        if task == "next_activity"
        else float(scores[0])
    )
