"""An illustrative log for trying the real pretrained model, not paper evidence."""

import pandas as pd

import chronos4pm as cpm

CASE_COUNT: int = 16
ORIGIN: pd.Timestamp = pd.Timestamp("2025-01-01", tz="UTC")
ACTIVITIES: tuple[str, ...] = ("Register", "Review", "Complete")


def example_log() -> pd.DataFrame:
    """Create a small, explicitly illustrative PM4Py event table.

    :return: Sixteen completed cases ordered by day.
    """
    rows: list[dict[str, str | pd.Timestamp]] = [
        {
            "case:concept:name": f"{case:03d}",
            "concept:name": activity,
            "time:timestamp": ORIGIN + pd.Timedelta(days=case, hours=position),
        }
        for case in range(CASE_COUNT)
        for position, activity in enumerate(ACTIVITIES)
    ]
    return pd.DataFrame(rows)


def run_example() -> None:
    """Run genuine aggregate and prefix inference on the illustrative log.

    :return: None.
    """
    log: pd.DataFrame = example_log()
    print("Forecast DFG probabilities:", cpm.discover_dfg(log)[0])
    print("Forecast temporal profile (seconds):", cpm.discover_temporal_profile(log))
    current_case: str = f"{CASE_COUNT - 1:03d}"
    context: pd.DataFrame = log[log["case:concept:name"] != current_case]
    prefix: pd.DataFrame = log[log["case:concept:name"] == current_case].iloc[:2]
    print(
        "Next activity:",
        cpm.predict_next_activity(context, prefix, covariates="selected"),
    )
    print(
        "Next-event duration (seconds):",
        cpm.predict_next_event_time(context, prefix, covariates="selected"),
    )
