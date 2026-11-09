"""Behavior of model-training/scripts/execution_record.py."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/mlops/model-training/scripts/execution_record.py"
)
spec = importlib.util.spec_from_file_location("execution_record", SCRIPT)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

PY = sys.executable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.csv").write_text("x\n1\n2\n")
    (tmp_path / "cfg.json").write_text('{"k": 1}')
    (tmp_path / "requirements.txt").write_text("pandas==2.3.3\n")
    (tmp_path / "work.py").write_text(
        "import sys, pathlib\n"
        "rows = pathlib.Path('in.csv').read_text().splitlines()\n"
        "pathlib.Path('out.txt').write_text(str(len(rows)))\n"
        "d = pathlib.Path('outdir'); d.mkdir(exist_ok=True)\n"
        "(d / 'a.txt').write_text('a')\n"
        "print('hello from work')\n"
        "print('warn', file=sys.stderr)\n"
        "sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)\n"
    )
    return tmp_path


def run_args(*extra, record="rec/execution_record.json"):
    return ["run", "--record", record, "--role", "evaluation", "--script", "work.py",
            "--config", "cfg.json", "--input", "in.csv", "--output", "out.txt",
            "--output", "outdir", "--lock", "requirements.txt", *extra]


def load(path="rec/execution_record.json"):
    return json.loads(Path(path).read_text())


def test_packages_only_from_explicit_interpreter(workspace):
    # The command's first word is not trusted: wrappers run code in another environment.
    er.main(["run", "--record", "a.json", "--role", "other", "--", PY, "-c", "pass"])
    environment = load("a.json")["environment"]
    assert environment["packages"] is None
    assert "--python" in environment["packages_from"]
    er.main(["run", "--record", "b.json", "--role", "other", "--python", PY, "--",
             PY, "-c", "pass"])
    assert "pytest" in load("b.json")["environment"]["packages"]


def test_run_records_inputs_outputs_scripts_and_logs(workspace, capsys):
    assert er.main([*run_args("--python", PY), "--", PY, "work.py"]) == 0
    assert "hello from work" in capsys.readouterr().out  # output still reaches the caller
    record = load()
    assert record["execution_status"] == "completed" and record["exit_code"] == 0
    assert record["validation_status"] == "unverified"
    assert record["command"] == [PY, "work.py"]
    assert record["inputs"][0]["sha256"] == sha("in.csv")
    assert record["locks"][0]["sha256"] == sha("requirements.txt")
    outputs = {o["path"]: o for o in record["outputs"]}
    assert outputs["out.txt"]["sha256"] == sha("out.txt")
    assert outputs["outdir"]["kind"] == "directory" and outputs["outdir"]["files"] == 1
    copy = Path("rec") / record["scripts"][0]["copy"]
    assert copy.read_text() == Path("work.py").read_text()
    assert (Path("rec") / record["configs"][0]["copy"]).read_text() == '{"k": 1}'
    assert "hello from work" in (Path("rec") / record["logs"]["stdout"]).read_text()
    assert "warn" in (Path("rec") / record["logs"]["stderr"]).read_text()
    assert "pytest" in record["environment"]["packages"]
    assert "inputs_changed_during_run" not in record


def test_failed_and_unstartable_commands(workspace):
    assert er.main([*run_args(), "--", PY, "work.py", "3"]) == 3
    assert load()["execution_status"] == "failed" and load()["exit_code"] == 3
    assert er.main([*run_args("--force"), "--", "no-such-command-xyz"]) == 127
    record = load()
    assert record["execution_status"] == "failed" and record["exit_code"] is None
    assert "could not start" in (Path("rec") / record["logs"]["stderr"]).read_text()


def test_existing_record_is_not_overwritten(workspace):
    assert er.main([*run_args(), "--", PY, "work.py"]) == 0
    with pytest.raises(SystemExit) as exc:
        er.main([*run_args(), "--", PY, "work.py"])
    assert exc.value.code == 2


def test_plan_is_not_run(workspace):
    assert er.main(["plan", "--record", "plan.json", "--role", "core", "--script",
                    "work.py", "--output", "out.txt", "--", PY, "work.py"]) == 0
    record = load("plan.json")
    assert record["execution_status"] == "not_run" and record["exit_code"] is None
    assert record["outputs"] == [{"path": "out.txt", "planned": True}]
    assert "logs" not in record and not Path("out.txt").exists()
    assert er.main(["verify", "--record", "plan.json"]) == 0


def test_checks_set_validation_status(workspace):
    er.main([*run_args(), "--", PY, "work.py"])
    record_path = "rec/execution_record.json"
    assert er.main(["check", "--record", record_path, "--name", "rows ok", "--",
                    PY, "-c", "print('ok')"]) == 0
    assert load()["validation_status"] == "passed"
    assert er.main(["check", "--record", record_path, "--name", "bad", "--",
                    PY, "-c", "import sys; sys.exit(4)"]) == 4
    record = load()
    assert record["validation_status"] == "failed"
    assert [c["status"] for c in record["checks"]] == ["passed", "failed"]
    assert (Path("rec") / record["checks"][0]["logs"]["stdout"]).read_text().strip() == "ok"


def test_passing_check_on_failed_run_stays_unverified(workspace):
    er.main([*run_args(), "--", PY, "work.py", "1"])
    er.main(["check", "--record", "rec/execution_record.json", "--name", "x", "--",
             PY, "-c", "pass"])
    assert load()["validation_status"] == "unverified"


def test_verify_detects_changes(workspace, capsys):
    er.main([*run_args(), "--", PY, "work.py"])
    capsys.readouterr()
    verify = ["verify", "--record", "rec/execution_record.json"]
    assert er.main(verify) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unchanged"

    Path("work.py").write_text(Path("work.py").read_text() + "# edited later\n")
    assert er.main(verify) == 0  # executed copy is intact; edit is only noted
    report = json.loads(capsys.readouterr().out)
    assert report["problems"] == [] and "work.py differs" in report["notes"][0]

    Path("outdir/a.txt").write_text("tampered")
    Path("in.csv").unlink()
    copy = Path("rec") / load()["scripts"][0]["copy"]
    copy.write_text("tampered")
    assert er.main(verify) == 1
    issues = {(p["section"], p["issue"]) for p in json.loads(capsys.readouterr().out)["problems"]}
    assert issues == {("outputs", "changed"), ("inputs", "missing"),
                      ("scripts", "frozen copy changed")}


def test_verify_with_moved_workspace(workspace, tmp_path, capsys):
    er.main([*run_args(), "--", PY, "work.py"])
    capsys.readouterr()
    moved = tmp_path.parent / (tmp_path.name + "-moved")
    tmp_path.rename(moved)
    record = moved / "rec/execution_record.json"
    assert er.main(["verify", "--record", str(record)]) == 1  # recorded cwd is gone
    capsys.readouterr()
    assert er.main(["verify", "--record", str(record), "--base", str(moved)]) == 0


def test_inputs_modified_by_the_run_are_flagged(workspace):
    Path("mutate.py").write_text("open('in.csv', 'a').write('3\\n')\n")
    er.main(["run", "--record", "r.json", "--role", "preparation", "--input", "in.csv",
             "--", PY, "mutate.py"])
    assert load("r.json")["inputs_changed_during_run"] == ["in.csv"]


def test_redaction():
    argv = ["train", "--token", "abc", "--api-key=xyz", "--tokenizer", "bert",
            "--db-password=hunter2", "https://user:pw@host/path", "--sort-key", "id"]
    assert er.redact(argv) == [
        "train", "--token", "<redacted>", "--api-key=<redacted>", "--tokenizer", "bert",
        "--db-password=<redacted>", "https://<redacted>@host/path", "--sort-key", "<redacted>",
    ]


def test_cli_entrypoint_and_separator(tmp_path):
    result = subprocess.run(
        [PY, str(SCRIPT), "run", "--record", "r.json", "--role", "other", "--",
         PY, "-c", "print(42)"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42"
    record = json.loads((tmp_path / "r.json").read_text())
    assert record["command"] == [PY, "-c", "print(42)"]
