#!/usr/bin/env python3
"""
Record and verify what was actually executed. Python standard library only.

Subcommands:
  plan    write a record for work that has not run yet (execution_status: not_run)
  run     run a command, tee its output to log files, and record what it used and produced
  check   run a validation command and attach its result to an existing record
  verify  re-hash the recorded files and report anything that changed since recording

A record keeps execution status (not_run / completed / failed) separate from validation
status (unverified / passed / failed). Validation becomes "passed" only when the run
completed and every attached check command exited 0; it is never set by assertion.
Scripts and configs are copied next to the record so later edits cannot change what the
record says was executed. Arguments that look like secrets are redacted, but do not pass
secrets on the command line; environment variables are never recorded.

Usage:
    # Run an evaluation and record it
    python execution_record.py run --record runs/eval/execution_record.json \\
        --role evaluation --script evaluate.py --config eval.yaml \\
        --input data/test.parquet --output runs/eval/metrics.json \\
        --lock uv.lock --python .venv/bin/python -- .venv/bin/python evaluate.py

    # Attach a validation check (any command; exit code 0 means passed)
    python execution_record.py check --record runs/eval/execution_record.json \\
        --name split-leakage -- python check_split_leakage.py --split ...

    # Later: confirm recorded inputs, outputs, scripts and locks are unchanged
    python execution_record.py verify --record runs/eval/execution_record.json

Exit codes: run/check return the command's exit code; verify returns 1 if a recorded file
changed or disappeared; 2 means invalid usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "execution-record-v1"
ROLES = ["preparation", "evaluation", "core", "other"]
SECRET_WORDS = {"token", "password", "passwd", "secret", "apikey", "key", "credential",
                "credentials", "auth"}
USERINFO = re.compile(r"(://)[^/@\s]+@")
PACKAGES_CODE = (
    "import importlib.metadata as m, json; "
    "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() "
    "if d.metadata['Name']}))"
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    """SHA-256 of a file, or of a directory's sorted (relative path, file hash) list."""
    path = Path(path)
    if path.is_file():
        return {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.is_dir():
        digest, count, size = hashlib.sha256(), 0, 0
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
        for item in files:
            digest.update(f"{item.relative_to(path).as_posix()}\0{sha256_file(item)}\n".encode())
            count += 1
            size += item.stat().st_size
        return {"sha256": digest.hexdigest(), "bytes": size, "files": count,
                "kind": "directory"}
    return {"missing": True}


def redact(argv):
    """Hide values of secret-looking flags and URL credentials."""
    result, hide_next = [], False
    for arg in argv:
        if hide_next:
            result.append("<redacted>")
            hide_next = False
            continue
        if arg.startswith("-"):
            name, sep, _ = arg.partition("=")
            words = set(re.split(r"[-_]+", name.lstrip("-").lower()))
            if words & SECRET_WORDS or "api-key" in name.lower() or "api_key" in name.lower():
                result.append(f"{name}=<redacted>" if sep else arg)
                hide_next = not sep
                continue
        result.append(USERINFO.sub(r"\1<redacted>@", arg))
    return result


def packages(python):
    # No guessing from the command: `python run.py`, `uv run` or `conda run` execute code
    # in a different environment than their first word, and a wrong package list is worse
    # than none.
    if not python:
        return None, ("not recorded; pass --python with the interpreter that runs the "
                      "code, or identify the environment with --lock")
    try:
        result = subprocess.run([python, "-c", PACKAGES_CODE], capture_output=True,
                                text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"could not query {python}: {exc}"
    if result.returncode:
        return None, f"could not query {python}: {result.stderr.strip()[-300:]}"
    found = json.loads(result.stdout)
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower())), python


def git_state(cwd):
    if not shutil.which("git"):
        return None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                          text=True)
    if head.returncode:
        return None
    status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                            text=True)
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def run_logged(command, cwd, stdout_log, stderr_log):
    """Run command, streaming output to the terminal and to log files."""
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    def pump(source, sink, echo):
        for chunk in iter(lambda: source.read1(65536), b""):
            sink.write(chunk)
            target = getattr(echo, "buffer", None)
            if target is not None:
                target.write(chunk)
            else:
                echo.write(chunk.decode(errors="replace"))
            echo.flush()

    with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        except OSError as exc:
            err.write(f"could not start command: {exc}\n".encode())
            return None
        threads = [
            threading.Thread(target=pump, args=(process.stdout, out, sys.stdout)),
            threading.Thread(target=pump, args=(process.stderr, err, sys.stderr)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return process.wait()


def load(path):
    record = json.loads(Path(path).read_text())
    if record.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} file")
    return record


def save(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + "\n")


def files_dir(record_path):
    return record_path.parent / f"{record_path.stem}_files"


def relative(path, start):
    try:
        return Path(path).relative_to(start).as_posix()
    except ValueError:
        return str(path)


def freeze(paths, cwd, record_path, kind):
    """Copy files beside the record so later edits cannot change what was executed."""
    frozen = []
    for index, name in enumerate(paths):
        source = cwd / name
        if not source.is_file():
            raise ValueError(f"--{kind} {name} is not a file")
        target = files_dir(record_path) / kind / f"{index}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        frozen.append({"path": name, "sha256": sha256_file(source),
                       "copy": relative(target, record_path.parent)})
    return frozen


def new_record(args, record_path, cwd):
    if record_path.exists() and not args.force:
        raise ValueError(f"{record_path} exists; use --force to replace it")
    command = list(args.command)
    installed, source = packages(args.python)
    return {
        "format": FORMAT,
        "role": args.role,
        "note": args.note,
        "command": redact(command),
        "cwd": str(cwd),
        "created_at": now(),
        "execution_status": "not_run",
        "exit_code": None,
        "validation_status": "unverified",
        "checks": [],
        "scripts": freeze(args.script, cwd, record_path, "scripts"),
        "configs": freeze(args.config, cwd, record_path, "configs"),
        "inputs": [{"path": p, **fingerprint(cwd / p)} for p in args.input],
        "outputs": [{"path": p, "planned": True} for p in args.output],
        "locks": [{"path": p, **fingerprint(cwd / p)} for p in args.lock],
        "model_uri": args.model_uri,
        "git": git_state(cwd),
        "environment": {
            "recorder_python": platform.python_version(),
            "platform": platform.platform(),
            "packages_from": source,
            "packages": installed,
        },
    }


def command_plan(args):
    record_path, cwd = Path(args.record).resolve(), Path.cwd()
    record = new_record(args, record_path, cwd)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    save(record_path, record)
    return 0


def command_run(args):
    if not args.command:
        raise ValueError("give the command to run after --")
    record_path, cwd = Path(args.record).resolve(), Path.cwd()
    record = new_record(args, record_path, cwd)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    logs = files_dir(record_path) / "logs"
    record["started_at"] = now()
    start = time.monotonic()
    code = run_logged(args.command, cwd, logs / "stdout.log", logs / "stderr.log")
    record.update(
        finished_at=now(),
        duration_seconds=round(time.monotonic() - start, 3),
        exit_code=code,
        execution_status="completed" if code == 0 else "failed",
        logs={"stdout": relative(logs / "stdout.log", record_path.parent),
              "stderr": relative(logs / "stderr.log", record_path.parent)},
        outputs=[{"path": p, **fingerprint(cwd / p)} for p in args.output],
    )
    changed = [e["path"] for e in record["inputs"]
               if fingerprint(cwd / e["path"]).get("sha256") != e.get("sha256")]
    if changed:
        record["inputs_changed_during_run"] = changed
    save(record_path, record)
    return 127 if code is None else code


def command_check(args):
    if not args.command:
        raise ValueError("give the check command after --")
    record_path = Path(args.record).resolve()
    record = load(record_path)
    index = len(record["checks"])
    logs = files_dir(record_path) / "checks"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name)
    stdout_log = logs / f"{index}_{safe}.stdout.log"
    stderr_log = logs / f"{index}_{safe}.stderr.log"
    started = now()
    code = run_logged(args.command, Path.cwd(), stdout_log, stderr_log)
    record["checks"].append({
        "name": args.name,
        "command": redact(list(args.command)),
        "cwd": str(Path.cwd()),
        "started_at": started,
        "finished_at": now(),
        "exit_code": code,
        "status": "passed" if code == 0 else "failed",
        "logs": {"stdout": relative(stdout_log, record_path.parent),
                 "stderr": relative(stderr_log, record_path.parent)},
    })
    statuses = {check["status"] for check in record["checks"]}
    if "failed" in statuses:
        record["validation_status"] = "failed"
    elif record["execution_status"] == "completed":
        record["validation_status"] = "passed"
    else:
        record["validation_status"] = "unverified"
    save(record_path, record)
    return 127 if code is None else code


def command_verify(args):
    record_path = Path(args.record).resolve()
    record = load(record_path)
    base = Path(args.base).resolve() if args.base else Path(record["cwd"])
    problems, notes = [], []

    def compare(section, entry, path):
        if entry.get("planned") or entry.get("missing"):
            return
        now_print = fingerprint(path)
        if now_print.get("missing"):
            problems.append({"section": section, "path": entry["path"], "issue": "missing"})
        elif now_print["sha256"] != entry["sha256"]:
            problems.append({"section": section, "path": entry["path"], "issue": "changed"})

    for section in ["inputs", "outputs", "locks"]:
        for entry in record.get(section, []):
            compare(section, entry, base / entry["path"])
    for section in ["scripts", "configs"]:
        for entry in record.get(section, []):
            copy = record_path.parent / entry["copy"]
            if not copy.is_file():
                problems.append({"section": section, "path": entry["copy"],
                                 "issue": "frozen copy missing"})
            elif sha256_file(copy) != entry["sha256"]:
                problems.append({"section": section, "path": entry["copy"],
                                 "issue": "frozen copy changed"})
            original = base / entry["path"]
            if not original.is_file() or sha256_file(original) != entry["sha256"]:
                notes.append(f"{entry['path']} differs from the executed version; "
                             f"the executed copy is {entry['copy']}")
    report = {
        "record": str(record_path),
        "status": "changed" if problems else "unchanged",
        "execution_status": record["execution_status"],
        "validation_status": record["validation_status"],
        "problems": problems,
        "notes": notes,
        "scope": "file integrity only; verify does not re-run or re-validate the work",
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for name, helptext in [("plan", "record planned work without running it"),
                           ("run", "run a command and record it")]:
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("--record", default="execution_record.json")
        sub.add_argument("--role", choices=ROLES, required=True,
                         help="preparation, evaluation, core (implementation) or other")
        sub.add_argument("--note", help="short description of the work")
        sub.add_argument("--script", action="append", default=[],
                         help="code file executed; copied beside the record (repeatable)")
        sub.add_argument("--config", action="append", default=[],
                         help="config file used; copied beside the record (repeatable)")
        sub.add_argument("--input", action="append", default=[],
                         help="input file or directory to hash (repeatable)")
        sub.add_argument("--output", action="append", default=[],
                         help="output file or directory to hash after the run (repeatable)")
        sub.add_argument("--lock", action="append", default=[],
                         help="dependency lock or requirements file to hash (repeatable)")
        sub.add_argument("--python",
                         help="interpreter that actually runs the code; its installed "
                              "packages are recorded (not inferred from the command)")
        sub.add_argument("--model-uri", help="model URI used or produced")
        sub.add_argument("--force", action="store_true", help="replace an existing record")
        sub.add_argument("command", nargs=argparse.REMAINDER,
                         help="-- followed by the command")
    check = commands.add_parser("check", help="run a validation command and attach it")
    check.add_argument("--record", default="execution_record.json")
    check.add_argument("--name", required=True, help="short name of the check")
    check.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by the command")
    verify = commands.add_parser("verify", help="re-hash recorded files")
    verify.add_argument("--record", default="execution_record.json")
    verify.add_argument("--base", help="directory to resolve recorded paths against "
                                       "(default: the recorded working directory)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    handlers = {"plan": command_plan, "run": command_run, "check": command_check,
                "verify": command_verify}
    try:
        return handlers[args.action](args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
