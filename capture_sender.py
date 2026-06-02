#!/usr/bin/env python3
"""
Realtime capture pipeline for simulated Bloomberg-style chat events.

Run this first, then point replay_excel_chat.py at:
  http://localhost:8000/ingest

It captures client sender IDs, messages, and timestamps into JSONL and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_CLIENT_IDS = {
    "15409927",
    "19020",
    "8233522",
    "7659808",
    "32879265",
    "29268290",
    "15793230",
    "29337542",
    "29705945",
    "7792265",
    "23586289",
}


class CaptureStore:
    def __init__(self, output_dir: Path, client_ids: set[str], capture_all: bool) -> None:
        self.output_dir = output_dir
        self.client_ids = client_ids
        self.capture_all = capture_all
        self.jsonl_path = output_dir / "realtime_chat_capture.jsonl"
        self.csv_path = output_dir / "realtime_chat_capture.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fields())
            writer.writeheader()

    @staticmethod
    def csv_fields() -> list[str]:
        return [
            "captured_at",
            "source_timestamp",
            "sender_id",
            "sender_name",
            "chat_id",
            "message",
            "source_row_num",
            "event_id",
        ]

    def should_capture(self, sender_id: str) -> bool:
        return self.capture_all or sender_id in self.client_ids

    def normalize_event(self, event: dict[str, Any]) -> dict[str, str]:
        sender_id = str(
            event.get("sender_id")
            or event.get("participant_id")
            or event.get("client_id")
            or ""
        ).strip()

        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_timestamp": str(event.get("source_timestamp") or ""),
            "sender_id": sender_id,
            "sender_name": str(event.get("sender") or event.get("sender_name") or ""),
            "chat_id": str(event.get("chat_id") or ""),
            "message": str(event.get("message") or ""),
            "source_row_num": str(event.get("row_num") or ""),
            "event_id": str(event.get("event_id") or ""),
        }

    def append(self, row: dict[str, str]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fields())
            writer.writerow(row)


def make_handler(store: CaptureStore) -> type[BaseHTTPRequestHandler]:
    class CaptureHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Use POST /ingest"})

        def do_POST(self) -> None:
            if self.path != "/ingest":
                self._send_json(404, {"error": "Use POST /ingest"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                event = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            row = store.normalize_event(event)
            if not row["message"]:
                self._send_json(202, {"captured": False, "reason": "empty message"})
                return

            if not store.should_capture(row["sender_id"]):
                self._send_json(
                    202,
                    {
                        "captured": False,
                        "reason": "sender_id not in client list",
                        "sender_id": row["sender_id"],
                    },
                )
                return

            store.append(row)
            print(
                f'{row["captured_at"]} sender_id={row["sender_id"]} '
                f'time={row["source_timestamp"]} message={row["message"]}',
                flush=True,
            )
            self._send_json(200, {"captured": True, "sender_id": row["sender_id"]})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return CaptureHandler


def parse_client_ids(value: str) -> set[str]:
    if not value.strip():
        return set(DEFAULT_CLIENT_IDS)
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Folder where JSONL and CSV capture files are saved",
    )
    parser.add_argument(
        "--client-ids",
        default="",
        help="Comma-separated sender IDs to capture; blank uses the default client list",
    )
    parser.add_argument(
        "--capture-all",
        action="store_true",
        help="Capture every event, even if sender_id is not in the client list",
    )
    args = parser.parse_args()

    client_ids = parse_client_ids(args.client_ids)
    store = CaptureStore(Path(args.output_dir), client_ids, args.capture_all)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))

    print(f"Capture pipeline listening on http://{args.host}:{args.port}/ingest")
    print(f"Capturing IDs: {'ALL' if args.capture_all else ', '.join(sorted(client_ids))}")
    print(f"JSONL output: {store.jsonl_path}")
    print(f"CSV output:   {store.csv_path}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
