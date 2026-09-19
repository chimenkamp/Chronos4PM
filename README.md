# Chronos4PM

Forecast process behavior with a frozen Chronos-2 model, using familiar
[PM4Py](https://processintelligence.solutions/pm4py) event logs, pandas DataFrames,
activity names, and directly-follows graphs.

This standalone package extracts the 16 task mappings from
[Chronos-2 meets Process Mining](https://github.com/chimenkamp/chronos-2-meets-process-mining).
It does not import or modify the benchmark repository. It contains no training,
fine-tuning, benchmark runner, paper reporting, or FEEED integration.

**These functions forecast the next window or event.** PM4Py normally describes
an already observed log. Dictionary keys and tuple structures are familiar, but
distribution values here are probabilities, not observed integer counts. This
is a forecast API, not a drop-in replacement for every PM4Py function or option.

## Install

Use Python 3.11 or newer in a separate environment. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python main.py
```

`main.py` runs a small illustrative event log through the real model. The first
forecast downloads the checkpoint from Hugging Face unless it is already cached.
Importing `chronos4pm` does not load a model. The default device is CPU.

The distribution name and import name are both `chronos4pm`. This repository does
not publish anything to PyPI. Until a release is uploaded, install from a checkout
or a supplied wheel:

```bash
python -m pip install /path/to/Chronos4PM
# Or, after building:
python -m pip install dist/chronos4pm-0.1.0-py3-none-any.whl
```

The package metadata pins `pm4py==2.7.23.4`, `ebi-pm==0.3.12`, and
`chronos-forecasting==2.1.0`. EBI's Python import is `ebi`. Other transitive
dependencies are resolved by pip; the package is not a complete lock of the
paper's experimental environment.

## Load a log and forecast

```python
import pm4py
import chronos4pm as cpm

log = pm4py.read_xes("completed_cases.xes")  # DataFrame or legacy EventLog
dfg, start_activities, end_activities = cpm.discover_dfg(log, n_windows=8)

print(dfg)  # {("Register", "Review"): probability, ...}
print(start_activities)  # {"Register": probability, ...}
print(cpm.discover_temporal_profile(log, n_windows=8))
# {("Register", "Review"): (mean_seconds, stdev_seconds), ...}
```

For tabular input, use columns `case:concept:name`, `concept:name`, and
`time:timestamp`. For example:

```python
import pandas as pd
import chronos4pm as cpm

log = pd.read_csv("completed_cases.csv")
log["time:timestamp"] = pd.to_datetime(log["time:timestamp"], utc=True)
probabilities = cpm.get_start_activities(log, n_windows=8)

# Custom column names work too:
dfg, starts, ends = cpm.discover_dfg(
    custom_log,
    activity_key="activity",
    timestamp_key="timestamp",
    case_id_key="case_id",
    n_windows=8,
)
```

`custom_log` above is your DataFrame with the three named columns. Inputs are
copied; the package does not mutate them. Events are sorted by timestamp and
cases by first timestamp, then case identifier. Supplied timezones are preserved
because calendar covariates use the observed local hour. Keep timestamps and the
forecast origin consistently timezone-aware (or consistently naive); normalize
to UTC explicitly, as above, when working with mixed input offsets. Use the same
time basis as the benchmark when comparing calendar covariates.

## Windows and forecast boundaries

Aggregate methods divide completed cases into **eight equal-size chronological
windows** by default and predict one further window of that same case count.
`n_windows` changes the number of history points. Case count must be at least
`n_windows` and divisible by it; the package rejects an uneven split rather than
discarding cases. A window is measured in cases, not hours or days.

Supply only completed historical cases. When `forecast_origin` is supplied,
every context case must end strictly before it. Automatic selected/global
covariates require this known timestamp to construct the benchmark's future
calendar covariates:

```python
probabilities = cpm.get_start_activities(
    log,
    n_windows=8,
    covariates="selected",
    forecast_origin=pd.Timestamp("2026-01-01T00:00:00Z"),
)
```

Choose an origin after **all** events in your completed-case context. Without
`forecast_origin`, target-only calls cannot check that external boundary; the
caller is responsible for supplying genuinely historical data. The package does
not construct the benchmark's held-out blocks or rolling evaluation origins.

## API

Every function takes `covariates="target_only"`, `"selected"`, or `"global"`.
Aggregate functions accept the shared keyword parameters listed below.

| Function | Forecast result |
| --- | --- |
| `discover_dfg(log)` | `(edge_probabilities, start_probabilities, end_probabilities)` |
| `discover_dfg_edges(log)` | `set[tuple[str, str]]`, from binary-presence scores ≥ 0.5 |
| `get_start_activities(log)` | `dict[str, float]` of start probabilities |
| `get_end_activities(log)` | `dict[str, float]` of end probabilities |
| `get_event_attribute_values(log, attribute="concept:name")` | `dict[str, float]` of event-mass probabilities |
| `get_variants_as_tuples(log)` | `dict[tuple[str, ...], float]` of variant probabilities |
| `discover_footprints(log)` | Dictionary of `activities`, `sequence`, and `parallel` sets |
| `discover_performance_dfg(log)` | `(edge_mean_seconds, start_probabilities, end_probabilities)` |
| `get_performance_bottlenecks(log)` | List of `(edge, seconds)` pairs, longest duration first |
| `discover_temporal_profile(log)` | `dict[tuple[str, str], tuple[float, float]]` of mean/stdev seconds |
| `analyse_entropy(log)` | EBI trace-distribution entropy forecast, in bits |
| `analyse_variety(log)` | EBI trace-variety forecast |
| `analyse_completeness(log)` | EBI event-log completeness forecast |
| `predict_next_activity(log, prefix)` | Next activity label (`str`) |
| `predict_next_event_time(log, prefix)` | Duration until the next event, in seconds (`float`) |
| `predict_remaining_time(log, prefix)` | Remaining case duration, in seconds (`float`) |

The three `analyse_*` names follow EBI; the prefix prediction and bottleneck/edge
helpers have no equivalent PM4Py prediction API. `discover_footprints` exposes
only the three fields modeled by the benchmark. `discover_performance_dfg` uses
PM4Py's **mean** edge duration target, not its default collection of aggregations.
DFG tuple methods also run start/end distribution forecasts; these are three
separate task requests sharing the cached model.

| Shared keyword | Default | Meaning |
| --- | --- | --- |
| `n_windows` | `8` | Equal-size aggregate history windows; aggregate methods only |
| `forecast_origin` | omitted | Known next-window origin; aggregate methods only |
| `activity_key` | `"concept:name"` | Activity column/event key |
| `timestamp_key` | `"time:timestamp"` | Timestamp column/event key |
| `case_id_key` | `"case:concept:name"` | Case column for DataFrames |
| `attribute_allowlist` | omitted | Event-carried attributes eligible for automatic global covariates |
| `past_covariates` | `{}` | Additional named, aligned numeric or categorical histories |
| `future_covariates` | `{}` | Additional known length-one values; names must occur in past covariates |

For `get_event_attribute_values`, use `attribute` to choose the activity/target
column; it takes precedence over `activity_key`.

## Predict from a case prefix

`log` contains completed prior cases. `prefix` contains **only events observed
so far** in one current case. It may be a PM4Py `Trace`, a one-case `EventLog`,
or a one-case DataFrame. Every context case must end before the first prefix
event; overlapping context is rejected, not silently removed.

```python
import pandas as pd
import chronos4pm as cpm

history = pd.read_csv("completed_cases.csv")
prefix = pd.DataFrame(
    [
        {
            "case:concept:name": "open-case",
            "concept:name": "Register",
            "time:timestamp": pd.Timestamp("2026-01-01T09:00:00Z"),
        },
        {
            "case:concept:name": "open-case",
            "concept:name": "Review",
            "time:timestamp": pd.Timestamp("2026-01-01T10:00:00Z"),
        },
    ]
)

activity = cpm.predict_next_activity(history, prefix, covariates="selected")
next_seconds = cpm.predict_next_event_time(history, prefix, covariates="selected")
remaining_seconds = cpm.predict_remaining_time(history, prefix, covariates="global")
```

The history must be earlier than this example's case start. Next activity uses
one-hot observed activity histories on context-supported labels. It needs one
observed event and does not predict an end-of-case class. Next-event time needs
two events to observe at least one interval. Remaining time uses PM4Py's remaining
durations from completed prior cases **at the same prefix position**, including
terminal zero durations when present in history. It rejects positions with no
context support. Pass a prefix whose case is still running; the package cannot
infer whether the last supplied event is actually terminal.

## Covariates as parameters

`target_only` adds no automatic covariates. `selected` and `global` use the
benchmark's task-dependent selections:

- Aggregate selected sets use progress/calendar and task-specific event volume
  or activity, variant and DFG support. Performance tasks select timing instead
  of mean prefix length. Global sets add all applicable aggregate families.
- Prefix sets use observed position, activity (where applicable), elapsed time,
  previous intervals, and cyclic calendar features. Global remaining-time inputs
  include collision-safe JSON encodings of complete observed activity prefixes.
- Global attributes are discovered only from context events. Case attributes
  are forward-filled after their first observation; event attributes are aligned
  directly. Trace metadata is not promoted into event covariates. Use
  `attribute_allowlist=["customer_type", "org:resource"]` to restrict fields to
  those known at prediction time. Availability cannot be inferred from a name.

You can add your own series to any set. For an aggregate call with eight windows:

```python
probabilities = cpm.get_start_activities(
    log,
    n_windows=8,
    past_covariates={"staff_on_shift": [4, 4, 5, 5, 6, 5, 4, 5]},
    future_covariates={"staff_on_shift": [6]},
)
```

Numerical lists/arrays and categorical string lists/arrays are supported. Each
past covariate is one-dimensional and has exactly the target history length;
each future covariate has length one and the same name as a past series. Names
must not collide with automatic covariates. Arrays are shared across target
channels, not extra jointly forecast targets.

| Task | Alignment of caller-provided past covariates |
| --- | --- |
| Aggregate | One value per chronological context window |
| Next activity | One value per observed prefix event; future covariates are rejected |
| Next-event time | One value per observed interval, aligned to its **starting** event; future values describe the current observed event |
| Remaining time | One value per chronologically ordered context case reaching the prefix position; future values describe the current observed prefix |

Only supply values available at the forecast origin. Shape validation cannot
detect a future-derived attribute. Full histories are passed to Chronos so its
categorical encoding occurs before its internal context truncation; this avoids
changing the long-history remaining-time mapping. Preparing very large histories
can therefore use substantial host memory.

## Override the model constant

Set `chronos4pm.MODEL` once before calling the functions. It is deliberately not
a parameter repeated on every task method.

```python
from dataclasses import replace
import chronos4pm as cpm

cpm.MODEL = replace(cpm.MODEL, device_map="mps")  # Apple Silicon; "cuda" for CUDA

# Or choose a different checkpoint:
cpm.MODEL = cpm.ModelConfig(
    repository_id="amazon/chronos-2",
    revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
    device_map="cpu",
)
```

| Benchmark checkpoint | Revision |
| --- | --- |
| `autogluon/chronos-2-small` (default) | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` |
| `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| `autogluon/chronos-2-synth` | `3607918a9fd027d5c465d8213e46b98e2c041cea` |

The last configuration's pipeline is cached. Changing the constant causes the
next call to load that configuration. Do not change the global constant while
other threads are forecasting. `ModelConfig.batch_size` controls the internal
series batch (default 256); a multivariate request may require a larger minimum.
The package performs one request at a time and propagates model/download/memory
errors. It does not substitute another model or shrink the process-element
support. Chronos-1/Bolt checkpoints are not supported by this Chronos-2 API.

## Method boundaries

The package retains native PM4Py/EBI target construction, one-step Chronos-2
median forecasts, Euclidean probability-simplex projection, binary edge
thresholding, and none/sequence/parallel argmax decoding. Missing performance
observations remain NaN in history. Distributions cannot introduce unseen
activities, edges, or variants. Unknown activities in an observed prefix have
zero entries in the context-supported one-hot channels.

Duration and EBI scalar forecasts are returned raw, as in the benchmark. They
can be negative or outside a statistic's natural bounds; temporal standard
deviations are not silently clipped. Use these methods for experimentation and
evaluate them on your own held-out data. A successful forecast does not establish
accuracy. Remaining-time prediction was implemented in the benchmark but excluded
from the current paper's 15-task evaluation.

The package does not reproduce the paper's experiment orchestration, metrics,
quantile reports, data selection, or runtime measurements. Use the original
repository for those. No benchmark artifacts or results are bundled here.

## Development and building a release

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests examples main.py
RUN_CHRONOS_INTEGRATION=1 python -m pytest -q
python -m build
python -m twine check dist/*
```

Default tests isolate decoding with explicit test doubles and call real PM4Py
and EBI. The opt-in integration tests execute all 16 tasks in all three covariate
arms with the real default checkpoint. Set `HF_HUB_OFFLINE=1` to use only already
cached Hugging Face files. Tests use illustrative logs, not the paper's datasets.
Wheel and source archives are created in `dist/`; building does not publish them.

The implementation is divided by responsibility: public functions in `api.py`,
native histories/decoders in `tasks.py`, prefix mappings in `predictive.py`,
covariate construction in `covariates.py`, input validation in `logs.py`, and
lazy model inference in `inference.py`. `main.py` only starts the example.
