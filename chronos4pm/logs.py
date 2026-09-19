"""Copy, validate and partition native PM4Py inputs."""

import pandas as pd
from pm4py.objects.log.obj import Event, EventLog, Trace

from .configuration import ACTIVITY_KEY, CASE_ID_KEY, TIMESTAMP_KEY, Log


def normalize_log(
    log: Log,
    *,
    activity_key: str = ACTIVITY_KEY,
    timestamp_key: str = TIMESTAMP_KEY,
    case_id_key: str = CASE_ID_KEY,
) -> EventLog:
    """Copy an event log into chronological cases using canonical XES keys.

    :param log: PM4Py event log or event DataFrame.
    :param activity_key: Activity attribute or column.
    :param timestamp_key: Timestamp attribute or column.
    :param case_id_key: Case identifier column for DataFrames.
    :return: Validated copy ordered by case start and case identifier.
    """
    if isinstance(log, pd.DataFrame):
        required: list[str] = [case_id_key, activity_key, timestamp_key]
        if not set(required).issubset(log.columns) or log[required].isna().any().any():
            raise ValueError(
                "event DataFrame requires non-null case, activity and timestamp columns"
            )
        frame: pd.DataFrame = log.copy()
        frame[timestamp_key] = pd.to_datetime(frame[timestamp_key])
        frame = frame.sort_values(timestamp_key, kind="stable")
        log = dataframe_log(frame, case_id_key)
    if not isinstance(log, EventLog):
        raise TypeError("log must be a pandas DataFrame or PM4Py EventLog")
    traces: list[Trace] = [
        normalize_trace(trace, activity_key, timestamp_key) for trace in log
    ]
    if not traces or any(not trace for trace in traces):
        raise ValueError("log and its traces must be non-empty")
    traces.sort(
        key=lambda trace: (
            trace[0][TIMESTAMP_KEY],
            str(trace.attributes.get(ACTIVITY_KEY, "")),
        )
    )
    return EventLog(traces)


def dataframe_log(frame: pd.DataFrame, case_id_key: str) -> EventLog:
    """Retain event-carried case attributes at their observed rows.

    :param frame: Validated event table.
    :param case_id_key: Column identifying traces.
    :return: PM4Py log without promoting case-prefixed columns into trace metadata.
    """
    return EventLog(
        [
            Trace(
                [
                    Event(row)
                    for row in case.drop(columns=[case_id_key]).to_dict("records")
                ],
                attributes={ACTIVITY_KEY: str(case_id)},
            )
            for case_id, case in frame.groupby(case_id_key, sort=False)
        ]
    )


def normalize_trace(trace: Trace, activity_key: str, timestamp_key: str) -> Trace:
    """Copy one trace and normalize its event keys without filling attributes.

    :param trace: Observed events only.
    :param activity_key: Input activity attribute.
    :param timestamp_key: Input timestamp attribute.
    :return: Copied trace in timestamp order.
    """
    events: list[Event] = []
    for original in trace:
        if activity_key not in original or timestamp_key not in original:
            raise ValueError("events require activity and timestamp attributes")
        if pd.isna(original[activity_key]) or pd.isna(original[timestamp_key]):
            raise ValueError("activity and timestamp must not be missing")
        event: Event = Event(dict(original))
        event[ACTIVITY_KEY] = str(original[activity_key])
        event[TIMESTAMP_KEY] = pd.Timestamp(original[timestamp_key])
        if activity_key != ACTIVITY_KEY:
            del event[activity_key]
        if timestamp_key != TIMESTAMP_KEY:
            del event[timestamp_key]
        events.append(event)
    return Trace(
        sorted(events, key=lambda event: event[TIMESTAMP_KEY]),
        attributes=dict(trace.attributes),
    )


def context_windows(log: EventLog, n_windows: int) -> list[EventLog]:
    """Partition whole cases into equal-size, chronological windows.

    :param log: Chronologically ordered completed cases.
    :param n_windows: Number of equally sized historical observations.
    :return: All input cases partitioned exactly once.
    """
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 1:
        raise ValueError("n_windows must be a positive integer")
    if len(log) < n_windows or len(log) % n_windows:
        raise ValueError(
            "case count must be at least n_windows and divisible by n_windows"
        )
    size: int = len(log) // n_windows
    return [EventLog(log[start : start + size]) for start in range(0, len(log), size)]


def require_prior_context(log: EventLog, origin: pd.Timestamp) -> None:
    """Enforce completion of every context case before the forecast boundary.

    :param log: Completed historical cases.
    :param origin: Known forecast boundary with compatible timezone awareness.
    :return: None.
    """
    if pd.isna(origin) or any(trace[-1][TIMESTAMP_KEY] >= origin for trace in log):
        raise ValueError("context cases must end strictly before the forecast origin")


def ebi_log(log: EventLog) -> EventLog:
    """Keep exactly the case, activity and timestamp fields consumed by EBI.

    :param log: Canonical context window.
    :return: Minimal XES event log matching the benchmark's EBI adapter.
    """
    return EventLog(
        [
            Trace(
                [
                    Event(
                        {
                            ACTIVITY_KEY: event[ACTIVITY_KEY],
                            TIMESTAMP_KEY: event[TIMESTAMP_KEY],
                        }
                    )
                    for event in trace
                ],
                attributes={
                    ACTIVITY_KEY: trace.attributes.get(ACTIVITY_KEY, str(index))
                },
            )
            for index, trace in enumerate(log)
        ]
    )
