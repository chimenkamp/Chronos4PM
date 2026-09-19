"""PM4Py-style functions returning forecasts over context-supported elements."""

from itertools import product
from typing import Unpack, cast

import numpy as np
from pm4py.objects.log.obj import Trace

from .configuration import (
    ACTIVITY_KEY,
    PRESENCE_THRESHOLD,
    RELATIONS,
    CovariateSet,
    Edge,
    Footprint,
    ForecastOptions,
    Key,
    Log,
    PrefixOptions,
)
from .predictive import predict
from .tasks import aggregate


def discover_dfg(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> tuple[dict[Edge, float], dict[str, float], dict[str, float]]:
    """Forecast directly-follows, start and end distributions for the next window.

    :param log: Completed historical cases (EventLog or DataFrame).
    :param covariates: Automatic target_only, selected or global covariate set.
    :param options: Window count, origin, PM4Py keys and past/future covariates.
    :return: PM4Py-style (dfg, start_activities, end_activities), with probabilities.
    """
    return (
        cast(dict[Edge, float], aggregate(log, "dfg", covariates, options)),
        get_start_activities(log, covariates=covariates, **options),
        get_end_activities(log, covariates=covariates, **options),
    )


def discover_dfg_edges(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> set[Edge]:
    """Forecast edge presence from binary histories using the benchmark threshold.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Set of ordered activity pairs with scores at least 0.5.
    """
    scores: dict[Key, float] = aggregate(log, "edges", covariates, options)
    return {
        cast(Edge, edge)
        for edge, score in scores.items()
        if score >= PRESENCE_THRESHOLD
    }


def get_start_activities(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> dict[str, float]:
    """Forecast the next-window start-activity distribution.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Activity-to-probability dictionary.
    """
    return cast(dict[str, float], aggregate(log, "start", covariates, options))


def get_end_activities(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> dict[str, float]:
    """Forecast the next-window end-activity distribution.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Activity-to-probability dictionary.
    """
    return cast(dict[str, float], aggregate(log, "end", covariates, options))


def get_event_attribute_values(
    log: Log,
    attribute: str = ACTIVITY_KEY,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> dict[str, float]:
    """Forecast event mass by activity (or another categorical event attribute).

    :param log: Completed historical cases.
    :param attribute: Attribute whose values form the target support.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Attribute-value-to-probability dictionary with string keys.
    """
    options["activity_key"] = attribute
    return cast(dict[str, float], aggregate(log, "activity", covariates, options))


def get_variants_as_tuples(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> dict[tuple[str, ...], float]:
    """Forecast probabilities of complete trace variants observed in context.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Activity-tuple-to-probability dictionary.
    """
    return cast(
        dict[tuple[str, ...], float], aggregate(log, "variants", covariates, options)
    )


def discover_footprints(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> Footprint:
    """Forecast the benchmark's activities, sequence and parallel footprint fields.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: PM4Py footprint dictionary containing the three forecast fields.
    """
    scores: dict[Key, float] = aggregate(log, "footprints", covariates, options)
    activities: set[str] = {cast(tuple[str, str, str], key)[0] for key in scores}
    result: Footprint = {"activities": activities, "sequence": set(), "parallel": set()}
    for source, target in product(sorted(activities), repeat=2):
        relation: str = RELATIONS[
            int(
                np.argmax(
                    [scores[(source, target, relation)] for relation in RELATIONS]
                )
            )
        ]
        if relation == "sequence":
            result["sequence"].add((source, target))
        elif relation == "parallel":
            result["parallel"].add((source, target))
    return result


def discover_performance_dfg(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> tuple[dict[Edge, float], dict[str, float], dict[str, float]]:
    """Forecast mean directly-follows durations and start/end distributions.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: PM4Py-style tuple; edge values are raw median forecasts in seconds.
    """
    return (
        cast(dict[Edge, float], aggregate(log, "performance", covariates, options)),
        get_start_activities(log, covariates=covariates, **options),
        get_end_activities(log, covariates=covariates, **options),
    )


def get_performance_bottlenecks(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> list[tuple[Edge, float]]:
    """Rank forecast mean edge durations from longest to shortest.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: (Edge, seconds) pairs with lexical tie-breaking.
    """
    scores: dict[Edge, float] = cast(
        dict[Edge, float], aggregate(log, "bottlenecks", covariates, options)
    )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def discover_temporal_profile(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> dict[Edge, tuple[float, float]]:
    """Forecast each temporal relation's mean and standard deviation.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: PM4Py pair-to-(mean, stdev) dictionary in seconds.
    """
    scores: dict[Key, float] = aggregate(log, "temporal", covariates, options)
    pairs: list[Edge] = [
        cast(tuple[Edge, int], key)[0]
        for key in scores
        if cast(tuple[Edge, int], key)[1] == 0
    ]
    return {pair: (scores[(pair, 0)], scores[(pair, 1)]) for pair in pairs}


def analyse_entropy(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> float:
    """Forecast the EBI trace-distribution entropy in bits.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Raw next-window scalar forecast.
    """
    return aggregate(log, "entropy", covariates, options)["value"]


def analyse_variety(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> float:
    """Forecast EBI's trace-variety statistic.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Raw next-window scalar forecast in EBI units.
    """
    return aggregate(log, "variety", covariates, options)["value"]


def analyse_completeness(
    log: Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[ForecastOptions],
) -> float:
    """Forecast EBI's event-log completeness statistic.

    :param log: Completed historical cases.
    :param covariates: Automatic covariate set.
    :param options: Shared aggregate forecast parameters.
    :return: Raw next-window scalar forecast in EBI units.
    """
    return aggregate(log, "completeness", covariates, options)["value"]


def predict_next_activity(
    log: Log,
    prefix: Trace | Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[PrefixOptions],
) -> str:
    """Predict the next activity from the observed within-case one-hot history.

    :param log: Completed cases ending before the current case starts.
    :param prefix: One observed case prefix, with at least one event.
    :param covariates: Automatic covariate set.
    :param options: PM4Py keys, attribute allowlist and aligned past covariates.
    :return: Activity label from context support; no end-of-case class is added.
    """
    return cast(str, predict(log, prefix, "next_activity", covariates, options))


def predict_next_event_time(
    log: Log,
    prefix: Trace | Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[PrefixOptions],
) -> float:
    """Predict the duration from the current observed event to its successor.

    :param log: Completed cases ending before the current case starts.
    :param prefix: One observed case prefix, with at least two events.
    :param covariates: Automatic covariate set.
    :param options: PM4Py keys, attribute allowlist and aligned past/future series.
    :return: Raw median duration in seconds, not an absolute timestamp.
    """
    return cast(float, predict(log, prefix, "next_time", covariates, options))


def predict_remaining_time(
    log: Log,
    prefix: Trace | Log,
    *,
    covariates: CovariateSet = "target_only",
    **options: Unpack[PrefixOptions],
) -> float:
    """Predict remaining cycle time from prior cases at the same prefix position.

    :param log: Completed cases ending before the current case starts.
    :param prefix: One observed nonterminal case prefix.
    :param covariates: Automatic covariate set.
    :param options: PM4Py keys, attribute allowlist and aligned past/future series.
    :return: Raw median remaining duration in seconds.
    """
    return cast(float, predict(log, prefix, "remaining_time", covariates, options))
