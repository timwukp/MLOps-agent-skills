# Temporal Validation and Leakage Prevention

Use this when rows have a time order: forecasting, failure or event prediction, churn,
fraud, sensor and industrial IoT data, logs and transactions. It applies to any framework
(pandas, sklearn, PyTorch, Spark, or a packaged trainer). It is a set of questions and checks
to settle before a score can be trusted, not a required pipeline.

A temporal model is only as honest as its answer to one question: **what could have
been known at the moment the prediction would have been made?** Random K-fold, global
scalers, centered rolling windows and "latest" table snapshots all quietly answer
"everything", which inflates offline scores and then fails in production.

## 1. Define the prediction before splitting

Write these down; they determine every later choice.

| Decision | Question | Example |
|---|---|---|
| Prediction time `t` | When is the model invoked, and with what data cut? | Hourly at :00, using data ingested by then |
| Sample grain | One row per what? | One device per hour |
| Target | What exactly is predicted, relative to `t`? | Any fault in `(t, t+24h]` |
| Horizon / lead time | How far ahead, and how much warning is useful? | ≥ 2h before the fault |
| Entities | Which keys separate independent series? | `device_id`, `site_id` |
| Deployment population | Same entities as training, new entities, or both? | New devices are installed monthly |

Natural-language horizons are ambiguous. "Predict failure one hour ahead" can mean:

- **State at `t+1h`**: is the device in a failed state at exactly `t+1h`?
- **Event within the window**: does a failure start anywhere in `(t, t+1h]`?
- **Minimum warning time**: alert at least 1h before failure onset, so the label window is `(t+1h, t+W]`.

These produce different labels, base rates and scores. Compute what you can from the data
yourself (columns, time ranges, event frequency), and ask the user only about meanings the
data cannot settle. Do not silently choose one.

## 2. Feature availability (point-in-time correctness)

A feature value is usable at prediction time `t` only if it was **observable** by `t`.
For every source, distinguish:

- **Event time**: when the measured thing happened.
- **Availability time**: when the value reached the system the model reads from. This is
  event time plus ingestion/reporting delay, batch schedules, manual entry lag and so on.
- **Revisions**: values corrected or backfilled later. A table queried today shows revised
  values that did not exist at `t`; use as-of snapshots or bitemporal history when they matter.

Rules for constructing features:

- **Use the past only, grouped by entity.** Sort by `(entity, time)` and compute lags and
  windows within each entity; ungrouped `shift`/`rolling` leaks across series boundaries.
- **Shift by the availability delay.** With step size `s` and delay `d`, a lag `k` is legal only if
  `k * s >= d`. Rolling windows should end at the most recent available observation,
  i.e. shift by `ceil(d / s)` steps before rolling.
- **Never include the target's own current or future value.** A rolling mean of the target
  computed without a shift contains the label. For a direct `h`-step forecast whose row is
  indexed at the target time, target lags must be at least `h`.
- **No centered windows, `bfill`, forward-looking interpolation or full-series normalization**
  in features. Forward fill is fine only when the filled value was itself available.
- **Cross-entity aggregates** (fleet mean, rank among devices) must use only entities' data
  available by `t`, not the whole day or the whole dataset.
- **"Known future" covariates** (calendar, published schedules, planned setpoints) are legal
  only if they really are known at `t`. Weather *forecasts* are; observed weather is not.
  A numeric column says nothing about its availability; record the evidence.
- **Fitted transforms** (scalers, imputers, encoders, target encoding, feature selection,
  PCA) are fit on the training portion of each fold only. Put them in the per-fold pipeline.

As-of joins attach the latest *available* value of a slowly changing source:

```python
import pandas as pd

# events: one row per (entity, prediction_time); readings: (entity, available_at, value)
features = pd.merge_asof(
    events.sort_values("prediction_time"),
    readings.sort_values("available_at"),
    left_on="prediction_time",
    right_on="available_at",
    by="entity",
    direction="backward",          # only values available at or before t
    allow_exact_matches=True,      # set False if "available at t" is not usable at t
)
```

Leak-safe lag and rolling features on a regular grid (one row per entity per step):

```python
import math

def past_rolling(df, entity, time, col, window, delay_steps):
    """Mean of the `window` most recent values available `delay_steps` before each row.

    Use delay_steps >= 1 for the target itself; ceil(delay / step) for other sources.
    """
    df = df.sort_values([entity, time])
    return df.groupby(entity, sort=False)[col].transform(
        lambda s: s.shift(delay_steps).rolling(window, min_periods=window).mean()
    )

delay_steps = math.ceil(pd.Timedelta("90min") / pd.Timedelta("1h"))  # -> 2
```

Rows with insufficient history become missing values. Decide explicitly whether to drop
them or keep them with a missing-value strategy, and report how many were affected.

Leakage smells that warrant investigation before reporting a score:

- A single feature almost perfectly separates the classes, or the model is far better than
  a naive baseline on a problem known to be hard.
- Top importances go to IDs, row numbers, timestamps, or fields written by the process that
  handles the outcome (repair ticket ID, `resolved_at`, maintenance flag, final status).
- Validation is much better than the same model on a later, untouched period.

## 3. Labels: windows, censoring and maturity

- **Window semantics.** For "event in `(t, t+W]`", the label at `t` is positive if an event
  occurs after `t` and at or before `t+W`. Pick and document the open and closed ends.
- **Right censoring.** Near the end of the data, `t+W` extends past the last observation.
  Those labels are **unknown**, not negative. Drop them, or model censoring explicitly.
- **Reporting delay and maturity.** A label for `t` becomes known at
  `t + W + reporting_delay`. A training row is usable at cutoff `c` only if
  `t + W + reporting_delay <= c`. Using immature labels trains on outcomes that would not
  yet be visible, and also mislabels recent positives as negatives.
- **Label provenance.** If the label comes from a downstream process (alarms that only fire
  when monitored, tickets that exist only after inspection), the label reflects that process
  as much as the physical event. Record that and check whether features encode it.
- **Overlapping windows.** Adjacent rows share most of their label window, so samples are
  strongly correlated. Effective sample size is much smaller than the row count; account for
  this when reading confidence intervals or comparing close scores.

## 4. Choosing a split

| Strategy | Use when | Leaks if misused | Notes |
|---|---|---|---|
| Random / stratified K-fold | Rows are exchangeable: no time order affects deployment, no repeated entities | Any temporal dependence, repeated entities, time-derived labels | Check for hidden time first (`created_at`, monotonically increasing IDs) |
| Group K-fold (by entity) | Deployment targets new entities | Shared time periods let models learn period effects | Combine with a time cutoff when both matter |
| Single time holdout | Simple final check; lots of data | Selecting models on it repeatedly | Keep it untouched until the end |
| Expanding / rolling-origin backtest | Model selection on temporal data | Too few origins → noisy; origins in an unusual period | Several origins across seasons and regimes |
| Purged backtest with embargo | Labels or features span time (windows, lags) | Without purging, training rows near the cutoff carry labels from the validation period | Purge: drop training rows whose label is not known by the cutoff. Embargo: extra gap for autocorrelated features |
| Entity × time | New entities *and* future periods | — | Hold out both a time range and a set of entities |

Practical rules:

- **Cut by timestamp, not by row index.** For panels with many entities, use one common cutoff
  time for all entities. sklearn's `TimeSeriesSplit` splits by row position; it is suitable
  for a single regularly spaced series sorted by time (its `gap` parameter can implement an
  embargo). For panels, define folds from timestamps.
- **Training rows for a fold**: rows with `time < cutoff` whose labels have matured by the
  cutoff (section 3). This maturity filter is the purge: it removes every training row whose
  label window or reporting delay reaches past the cutoff. Add an embargo gap when features
  are strongly autocorrelated across the cutoff.
- **Validation rows**: the period starting at the cutoff, keeping only labels that mature
  before the final test period starts.
- **Model selection data must also respect the final test.** Data used to tune hyperparameters,
  thresholds or features must have been observable before the test period starts.
- **Refit before the test** uses only data whose labels matured before the test start.
- **Hyperparameter search** runs inside the backtest; each fold refits every fitted transform.

A minimal, framework-agnostic fold generator for panel data:

```python
import pandas as pd

def time_folds(df, time_col, cutoffs, valid_span, label_known_col=None, gap="0h"):
    """Yield (cutoff, train, valid) using one common cutoff time across entities.

    Time columns must be timezone-aware UTC (cutoffs are parsed as UTC).
    label_known_col: per-row time at which the label becomes known
                     (t + window + reporting_delay); None for labels known at t.
    gap: optional embargo before the cutoff.
    """
    for cutoff in pd.to_datetime(cutoffs, utc=True):
        train_end = cutoff - pd.Timedelta(gap)
        train = df[df[time_col] < train_end]
        if label_known_col:
            train = train[train[label_known_col] <= cutoff]
        valid = df[(df[time_col] >= cutoff)
                   & (df[time_col] < cutoff + pd.Timedelta(valid_span))]
        yield cutoff, train, valid
```

## 5. Forecasting specifics

- **Origin and horizon.** State the forecast origin (the last observed time, or the first
  predicted time; name which) and the horizons evaluated. Report metrics per horizon;
  averaging hides degradation at long horizons.
- **Strategy.** *Direct* models train one model per horizon on features available at the
  origin. *Recursive* models feed their own predictions back as lags, so evaluation must use
  predictions, never the true future target, for those lags. *Multi-output* predicts all
  horizons at once. Each has different leakage risks; check the one you implement.
- **Exogenous inputs over the horizon.** For every step up to the maximum horizon, an
  exogenous input must be known-future or forecast, never the observed value.
- **Baselines.** Always compare with naive (last value) and seasonal naive (same time last
  period) on the same samples. A model that cannot beat them is not ready, whatever its
  absolute error.
- **Scale-free metrics** (MASE, sMAPE, WAPE) make comparisons across entities meaningful;
  decide whether entities are pooled or weighted equally and say which.

## 6. Event and failure prediction specifics

- **Base rate per fold.** Report positive rates per fold and period. Temporal shifts in base
  rate change PR-AUC and precision substantially.
- **Thresholds** come from validation data only, ideally chosen from the user's alert budget
  or false-positive/false-negative costs. Accuracy and default 0.5 thresholds are rarely the
  operational choice; state when you are using a default.
- **Event-level evaluation.** Many positive rows belong to one event. Also report events
  detected, lead time before the event, and false alarms per entity per period.
- **Post-event rows.** Decide whether rows during and right after a failure (repair,
  restart) are excluded; they often contain trivially predictable or mislabeled samples.

## 7. Holdout discipline

Keep a final test period that does not influence any choice: features, thresholds, models,
or stopping. If it was looked at during development, results on it are development
results. Report that plainly instead of presenting them as an unbiased estimate.
When the history of test usage is unknown, say so.

## 8. Verify, don't assume

Write checks against the frames you actually trained and evaluated on, not against the
intended design. `scripts/check_split_leakage.py` in this skill runs the common checks on
saved split files (CSV/Parquet, pandas only) and writes a JSON report with input hashes:
time order and gaps, label maturity, entity overlap, duplicate keys, rows copied across
splits, and single features that nearly determine the target. Checks it cannot run
without settings are reported as skipped, not passed. It does not replace the
point-in-time replay below. Useful assertions in your own code:

```python
def check_temporal_split(train, valid, time_col, cutoff, entity=None,
                         label_known_col=None, require_new_entities=False):
    cutoff = pd.Timestamp(cutoff)
    assert train[time_col].max() < cutoff, "training rows at/after cutoff"
    assert valid[time_col].min() >= cutoff, "validation rows before cutoff"
    if label_known_col:
        assert (train[label_known_col] <= cutoff).all(), "immature training labels"
    if require_new_entities:
        assert not set(train[entity]) & set(valid[entity]), "entity leakage"
```

Stronger checks worth the extra effort:

- **Point-in-time replay.** Pick a few random `(entity, t)` samples, recompute their features
  from raw data filtered to what was available by `t`, and compare with the training table.
  Any mismatch means a feature used information from after `t`.
- **Extra-lag probe.** Lag a suspect feature by one or two additional steps and retrain. A
  sharp score drop from a small extra lag suggests its original alignment used information
  from at or after `t`.
- **Later-period check.** Score the final model on a period after all development data.
  A large gap from backtest scores points to leakage or drift.
- **Baseline comparison** on identical samples (section 5).

Report which checks were executed and which were only planned.

## 9. What to report

State the prediction time, sample grain, target definition and window, the availability
assumptions per feature source, the split strategy with cutoffs and gaps, how labels near the
end were treated, which checks were executed, metrics per fold/horizon with baselines, and
whether the test period was used for any decision. Separate assumptions confirmed by the user
or documents from defaults chosen by the agent.
