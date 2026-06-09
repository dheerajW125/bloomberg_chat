#!/usr/bin/env python3
"""Run the complete replay-to-order-lifecycle pipeline in real time."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any


PROJECT_ROOT = Path(__file__).resolve().parent
STAGES_DIR = PROJECT_ROOT / "outputs"


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def stage_command(script: str, *arguments: Any) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(STAGES_DIR / script),
        *(str(argument) for argument in arguments),
    ]


def relay_output(name: str, stream: IO[str] | None) -> None:
    if stream is None:
        return
    for line in stream:
        print(f"[{name}] {line.rstrip()}", flush=True)


def start_stage(
    name: str,
    command: list[str],
    processes: list[tuple[str, subprocess.Popen[str]]],
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    processes.append((name, process))
    thread = threading.Thread(
        target=relay_output,
        args=(name, process.stdout),
        daemon=True,
    )
    thread.start()
    return process


def stop_processes(
    processes: list[tuple[str, subprocess.Popen[str]]],
) -> None:
    for _, process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for _, process in reversed(processes):
        if process.poll() is not None:
            continue
        timeout = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
    for _, process in reversed(processes):
        if process.poll() is None:
            process.wait()


def check_stage_health(
    processes: list[tuple[str, subprocess.Popen[str]]],
) -> None:
    failures = [
        (name, process.returncode)
        for name, process in processes
        if process.poll() is not None and process.returncode not in {0, None}
    ]
    if failures:
        detail = ", ".join(f"{name} exited {code}" for name, code in failures)
        raise RuntimeError(detail)


def wait_for_stage_readiness(
    processes: list[tuple[str, subprocess.Popen[str]]],
    expected_files: tuple[Path, ...],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        check_stage_health(processes)
        if all(path.exists() for path in expected_files):
            return
        time.sleep(0.1)
    missing = ", ".join(str(path) for path in expected_files if not path.exists())
    raise RuntimeError(f"pipeline startup timed out; missing readiness files: {missing}")


def clear_run_outputs(run_dir: Path) -> None:
    files = (
        "raw_chat_events.jsonl",
        "client_messages.jsonl",
        "trader_messages.jsonl",
        "automated_messages.jsonl",
        "trader_classification_review.jsonl",
        "nlp_trade_intent_messages.jsonl",
        "nlp_trade_review.jsonl",
        "trader_order_events.jsonl",
        "trader_event_mapping_review.jsonl",
        "order_lifecycle_snapshots.jsonl",
        "trader_registry.sqlite3",
        "nlp_chat_sessions.sqlite3",
        "orders.sqlite3",
    )
    directories = (
        "client_sessions_by_client",
        "client_intents_by_client",
        "client_trader_events",
        "client_order_lifecycle",
    )
    for name in files:
        path = run_dir / name
        if path.exists():
            path.unlink()
    for name in directories:
        path = run_dir / name
        if path.exists():
            shutil.rmtree(path)


def prepare_paths(run_dir: Path, rebuild: bool) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if rebuild:
        clear_run_outputs(run_dir)
    paths = {
        "raw": run_dir / "raw_chat_events.jsonl",
        "sessions": run_dir / "client_sessions_by_client",
        "intents": run_dir / "client_intents_by_client",
        "trader_events": run_dir / "client_trader_events",
        "lifecycle": run_dir / "client_order_lifecycle",
        "trader_messages": run_dir / "trader_messages.jsonl",
        "trader_event_jsonl": run_dir / "trader_order_events.jsonl",
        "mapping_review": run_dir / "trader_event_mapping_review.jsonl",
        "snapshots": run_dir / "order_lifecycle_snapshots.jsonl",
        "registry_db": run_dir / "trader_registry.sqlite3",
        "session_db": run_dir / "nlp_chat_sessions.sqlite3",
        "orders_db": run_dir / "orders.sqlite3",
    }
    for key in ("sessions", "intents", "trader_events", "lifecycle"):
        paths[key].mkdir(parents=True, exist_ok=True)
    for key in ("raw", "trader_messages"):
        paths[key].touch(exist_ok=True)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay chat rows and run role classification, client intent NLP, "
            "trader event NLP, and order lifecycle correlation together."
        )
    )
    parser.add_argument(
        "--chat-file",
        required=True,
        type=existing_file,
        help="Chat replay source: .xlsx, .xls, or .csv",
    )
    parser.add_argument(
        "--symbol-file",
        required=True,
        type=existing_file,
        help="Ticker master: .xlsx, .xls, .xlsm, or .csv",
    )
    parser.add_argument(
        "--run-dir",
        type=output_directory,
        default=PROJECT_ROOT / "outputs" / "realtime_pipeline",
        help="Directory for all generated pipeline data",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--drain-seconds", type=float, default=8.0)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--date-col", default="A")
    parser.add_argument("--timestamp-col", default="B")
    parser.add_argument("--sender-col", default="C")
    parser.add_argument("--chat-id-col", default="D")
    parser.add_argument("--participant-id-col", default="E")
    parser.add_argument("--message-col", default="F")
    parser.add_argument(
        "--actor-id-field",
        default="participant_id",
        help="Replay event field used as the sender/client/trader identifier",
    )
    parser.add_argument("--symbol-sheet", default=0)
    parser.add_argument("--symbol-col", default="Symbol")
    parser.add_argument("--symbol-skip-rows", type=int, default=0)
    parser.add_argument("--additional-client-ids", default="")
    parser.add_argument("--max-match-minutes", type=float, default=120.0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete this run directory's generated pipeline outputs first",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.interval < 0
        or args.poll_interval <= 0
        or args.startup_timeout <= 0
        or args.drain_seconds < 0
    ):
        raise SystemExit("Intervals must be non-negative; poll interval must be positive")

    paths = prepare_paths(args.run_dir, args.rebuild)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    capture_command = stage_command(
        "capture_trader_pipeline.py",
        "--input",
        paths["raw"],
        "--actor-id-field",
        args.actor_id_field,
        "--follow",
        "--poll-interval",
        args.poll_interval,
        "--additional-client-ids",
        args.additional_client_ids,
        "--output-dir",
        args.run_dir,
        "--client-session-dir",
        paths["sessions"],
        "--registry-db",
        paths["registry_db"],
    )
    client_nlp_command = stage_command(
        "nlp_trade_intent_layer.py",
        "--input",
        paths["sessions"],
        "--symbol-file",
        args.symbol_file,
        "--symbol-sheet",
        args.symbol_sheet,
        "--symbol-col",
        args.symbol_col,
        "--symbol-skip-rows",
        args.symbol_skip_rows,
        "--follow",
        "--poll-interval",
        args.poll_interval,
        "--output-dir",
        args.run_dir,
        "--client-output-dir",
        paths["intents"],
        "--session-db",
        paths["session_db"],
    )
    trader_nlp_command = stage_command(
        "trader_event_nlp.py",
        "--input",
        paths["trader_messages"],
        "--symbol-file",
        args.symbol_file,
        "--symbol-column",
        args.symbol_col,
        "--symbol-sheet",
        args.symbol_sheet,
        "--symbol-skip-rows",
        args.symbol_skip_rows,
        "--client-intents",
        paths["intents"],
        "--client-output-dir",
        paths["trader_events"],
        "--mapping-review",
        paths["mapping_review"],
        "--max-match-minutes",
        args.max_match_minutes,
        "--follow",
        "--poll-interval",
        args.poll_interval,
        "--output",
        paths["trader_event_jsonl"],
    )
    lifecycle_command = stage_command(
        "order_lifecycle_correlator.py",
        "--client-intents",
        paths["intents"],
        "--trader-events",
        paths["trader_events"],
        "--client-sessions-dir",
        paths["sessions"],
        "--client-output-dir",
        paths["lifecycle"],
        "--follow",
        "--poll-interval",
        args.poll_interval,
        "--database",
        paths["orders_db"],
        "--output",
        paths["snapshots"],
    )
    replay_command = stage_command(
        "replay_excel_chat.py",
        "--file",
        args.chat_file,
        "--sheet",
        args.sheet,
        "--interval",
        args.interval,
        "--skip-rows",
        args.skip_rows,
        "--start-row",
        args.start_row,
        "--limit",
        args.limit,
        "--date-col",
        args.date_col,
        "--timestamp-col",
        args.timestamp_col,
        "--sender-col",
        args.sender_col,
        "--chat-id-col",
        args.chat_id_col,
        "--participant-id-col",
        args.participant_id_col,
        "--message-col",
        args.message_col,
    )

    print(f"Pipeline output: {args.run_dir}", flush=True)
    print("Starting real-time stages...", flush=True)
    try:
        start_stage("capture", capture_command, processes)
        start_stage("client-nlp", client_nlp_command, processes)
        start_stage("trader-nlp", trader_nlp_command, processes)
        start_stage("lifecycle", lifecycle_command, processes)
        wait_for_stage_readiness(
            processes,
            (
                args.run_dir / "nlp_trade_intent_messages.jsonl",
                args.run_dir / "nlp_trade_review.jsonl",
                paths["trader_event_jsonl"],
                paths["snapshots"],
            ),
            args.startup_timeout,
        )

        print(f"Replaying chat: {args.chat_file}", flush=True)
        replay = subprocess.Popen(
            replay_command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        emitted = 0
        with paths["raw"].open("a", encoding="utf-8") as raw_output:
            if replay.stdout is not None:
                for line in replay.stdout:
                    text = line.strip()
                    if not text:
                        continue
                    event = json.loads(text)
                    raw_output.write(json.dumps(event, ensure_ascii=False) + "\n")
                    raw_output.flush()
                    emitted += 1
                    print(
                        f'[replay] {event.get("source_timestamp", "")} '
                        f'actor={event.get(args.actor_id_field, "")} '
                        f'message={event.get("message", "")}',
                        flush=True,
                    )
                    check_stage_health(processes)
        replay_stderr = replay.stderr.read() if replay.stderr is not None else ""
        replay_code = replay.wait()
        if replay_code:
            raise RuntimeError(
                f"replay exited {replay_code}: {replay_stderr.strip()}"
            )

        print(
            f"Replay complete: {emitted} messages. "
            f"Draining pipeline for {args.drain_seconds:g} seconds...",
            flush=True,
        )
        deadline = time.monotonic() + args.drain_seconds
        while time.monotonic() < deadline:
            check_stage_health(processes)
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("Stopping pipeline...", flush=True)
        return 130
    except (json.JSONDecodeError, OSError, RuntimeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_processes(processes)

    print("Pipeline complete.", flush=True)
    print(f"Per-client lifecycle JSON: {paths['lifecycle']}", flush=True)
    print(f"Automated messages: {args.run_dir / 'automated_messages.jsonl'}", flush=True)
    print(
        f"Classification review: "
        f"{args.run_dir / 'trader_classification_review.jsonl'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
