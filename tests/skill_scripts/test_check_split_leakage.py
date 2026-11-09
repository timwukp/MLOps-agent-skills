"""Behavior of model-training/scripts/check_split_leakage.py on known-good and leaky splits."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/mlops/model-training/scripts/check_split_leakage.py"
)
spec = importlib.util.spec_from_file_location("check_split_leakage", SCRIPT)
leak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak)

H = pd.Timedelta("1h")
START = pd.Timestamp("2025-01-01", tz="UTC")
LABEL_DELAY = pd.Timedelta("24h") + pd.Timedelta("2h")  # window + reporting delay


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    times = pd.date_range(START, periods=24 * 12, freq="h")
    return pd.concat(
        [
            pd.DataFrame({
                "timestamp": times,
                "device": device,
                "temperature": rng.normal(size=len(times)),
                "fault": (rng.random(len(times)) < 0.1).astype(int),
            })
            for device in ["a", "b", "c"]
        ],
        ignore_index=True,
    )


def by_check(report):
    return {c["check"]: c for c in report["checks"]}


def temporal(splits, **kwargs):
    options = dict(time_column="timestamp", entity_columns=["device"], target="fault")
    return leak.check_splits(splits, **{**options, **kwargs})


def clean_splits(panel):
    valid_start, test_start = START + 6 * 24 * H, START + 9 * 24 * H
    t = panel["timestamp"]
    return {
        "train": panel[t + LABEL_DELAY <= valid_start],
        "valid": panel[(t >= valid_start) & (t + LABEL_DELAY <= test_start)],
        "test": panel[t >= test_start],
    }


def test_purged_temporal_split_passes(panel):
    report = temporal(clean_splits(panel), label_window="24h", reporting_delay="2h")
    checks = by_check(report)
    assert report["status"] == "pass", report
    assert report["skipped"] == []
    for name in ["time_order", "label_maturity", "duplicate_keys", "entity_overlap",
                 "cross_split_duplicates", "target_proxy"]:
        assert checks[name]["status"] == "pass", checks[name]


def test_per_entity_cutoffs_fail_time_order(panel):
    # Each device's last 30% as test: devices with later data leak into earlier test.
    staggered = panel[~((panel["device"] == "a") & (panel["timestamp"] > START + 5 * 24 * H))]
    train, test = [], []
    for _, group in staggered.groupby("device"):
        cut = int(len(group) * 0.7)
        train.append(group.iloc[:cut])
        test.append(group.iloc[cut:])
    report = temporal({"train": pd.concat(train), "test": pd.concat(test)})
    check = by_check(report)["time_order"]
    assert report["status"] == "fail"
    assert check["status"] == "fail" and check["violations"][0]["rows"] > 0


def test_unpurged_labels_fail_maturity(panel):
    cutoff = START + 6 * 24 * H
    splits = {"train": panel[panel["timestamp"] < cutoff],
              "valid": panel[panel["timestamp"] >= cutoff]}
    report = temporal(splits, label_window="24h", reporting_delay="2h")
    checks = by_check(report)
    assert checks["time_order"]["status"] == "pass"
    assert checks["label_maturity"]["status"] == "fail"
    # Labels known at t+26h: rows from cutoff-25h to cutoff-1h are immature (25 x 3 devices);
    # the row at exactly cutoff-26h becomes known at the cutoff and is allowed.
    assert checks["label_maturity"]["violations"][0]["rows"] == 25 * 3


def test_label_known_column_and_unknown_labels(panel):
    cutoff = START + 6 * 24 * H
    data = panel.assign(label_known=panel["timestamp"] + 3 * H)
    train = data[data["timestamp"] + 3 * H <= cutoff].copy()
    splits = {"train": train, "valid": data[data["timestamp"] >= cutoff]}
    assert by_check(temporal(splits, label_known_column="label_known"))[
        "label_maturity"]["status"] == "pass"
    train.loc[train.index[:4], "label_known"] = pd.NaT  # label never observed
    check = by_check(temporal(splits | {"train": train}, label_known_column="label_known"))[
        "label_maturity"]
    assert check["status"] == "fail" and check["violations"][0]["rows"] == 4


def test_missing_label_timing_is_skipped_not_passed(panel):
    report = temporal(clean_splits(panel))
    assert by_check(report)["label_maturity"]["status"] == "skipped"
    assert "label_maturity" in report["skipped"]


def test_gap_is_enforced(panel):
    cutoff = START + 6 * 24 * H
    splits = {"train": panel[panel["timestamp"] < cutoff],
              "valid": panel[panel["timestamp"] >= cutoff]}
    assert by_check(temporal(splits, gap="6h"))["time_order"]["status"] == "fail"
    splits["train"] = panel[panel["timestamp"] < cutoff - 6 * H]
    assert by_check(temporal(splits, gap="6h"))["time_order"]["status"] == "pass"


def test_duplicate_entity_time_keys_warn(panel):
    splits = clean_splits(panel)
    splits["train"] = pd.concat([splits["train"], splits["train"].iloc[:5]])
    check = by_check(temporal(splits))["duplicate_keys"]
    assert check["status"] == "warn" and check["duplicates"] == {"train": 5}


def test_mixed_naive_and_aware_times_fail(panel):
    splits = clean_splits(panel)
    splits["test"] = splits["test"].assign(
        timestamp=splits["test"]["timestamp"].dt.tz_localize(None)
    )
    report = temporal(splits)
    assert report["status"] == "fail"
    assert by_check(report)["time_parse"]["status"] == "fail"


@pytest.fixture
def tabular():
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.4).astype(int)
    frame = pd.DataFrame({
        "customer": np.arange(n) % 150,
        "noise": rng.normal(size=n),
        "leaky": y + rng.normal(scale=0.01, size=n),
        "y": y,
    })
    return frame.iloc[:300], frame.iloc[300:]


def test_tabular_proxy_overlap_and_duplicates(tabular):
    train, test = tabular
    test = pd.concat([test, train.iloc[:7]])
    report = leak.check_splits(
        {"train": train, "test": test},
        target="y", entity_columns=["customer"], disjoint_entities=True,
    )
    checks = by_check(report)
    assert report["status"] == "fail"
    assert checks["entity_overlap"]["status"] == "fail"
    assert checks["cross_split_duplicates"]["status"] == "warn"
    assert checks["cross_split_duplicates"]["duplicates"][0]["rows"] == 7
    assert [f["column"] for f in checks["target_proxy"]["features"]] == ["leaky"]
    assert {"time_order", "label_maturity"} <= set(report["skipped"])


def test_regression_proxy_and_exclusion(tabular):
    train, test = tabular
    train = train.assign(y=train["noise"] * 3 + 1)
    report = leak.check_splits({"train": train, "test": test}, target="y")
    assert [f["column"] for f in by_check(report)["target_proxy"]["features"]] == ["noise"]
    report = leak.check_splits({"train": train, "test": test}, target="y",
                               exclude_columns=["noise"])
    assert by_check(report)["target_proxy"]["status"] == "pass"


def test_invalid_settings_raise(tabular):
    splits = dict(zip(["train", "test"], tabular))
    with pytest.raises(ValueError, match="require time_column"):
        leak.check_splits(splits, label_window="24h")
    with pytest.raises(ValueError, match="at least two"):
        leak.check_splits({"train": tabular[0]})
    with pytest.raises(ValueError, match="lacks columns"):
        leak.check_splits(splits, target="missing")


def test_cli_files_hashes_and_exit_codes(panel, tmp_path, capsys):
    splits = clean_splits(panel)
    paths = {}
    for name, frame in splits.items():
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    base = [f"--split={name}={path}" for name, path in paths.items()]
    base += ["--time-column", "timestamp", "--entity-column", "device", "--target", "fault",
             "--label-window", "24h", "--reporting-delay", "2h"]

    output = tmp_path / "report.json"
    assert leak.main([*base, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "pass"
    assert report["files"][str(paths["train"])] == hashlib.sha256(
        paths["train"].read_bytes()).hexdigest()

    leaky = splits["train"].assign(copy=splits["train"]["fault"] * 1.0)
    leaky.to_csv(paths["train"], index=False)
    for name in ["valid", "test"]:
        splits[name].assign(copy=0.0).to_csv(paths[name], index=False)
    assert leak.main(base) == 0  # target proxy only warns
    assert json.loads(capsys.readouterr().out)["status"] == "warn"
    assert leak.main([*base, "--strict"]) == 1


def test_cli_single_file_with_split_column(panel, tmp_path, capsys):
    cutoff = START + 6 * 24 * H
    data = panel.assign(part=np.where(panel["timestamp"] < cutoff, "train", "test"))
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    args = ["--input", str(path), "--split-column", "part", "--order", "train,test",
            "--time-column", "timestamp", "--label-window", "24h"]
    assert leak.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert by_check(report)["label_maturity"]["status"] == "fail"
    assert report["splits"]["train"]["rows"] == int((data["part"] == "train").sum())

    with pytest.raises(SystemExit) as exc:
        leak.main([*args[:5], "train", "--time-column", "timestamp"])
    assert exc.value.code == 2
