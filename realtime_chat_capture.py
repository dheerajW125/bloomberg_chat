#!/usr/bin/env python3
"""
Realtime raw chat capture service.

Captures every participant message into one stream:
  POST http://127.0.0.1:8000/ingest

Outputs:
  raw_chat.jsonl
  raw_chat.csv

Role filtering must happen downstream. Do not filter clients/traders here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RawChatStore:
    FIELDS = [
        "event_id",
        "captured_at",
        "source_timestamp",
        "room_id",
        "room_name",
        "sender_id",
        "sender_name",
        "participant_id",
        "message",
        "source_row_num",
    ]

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "raw_chat.jsonl"
        self.csv_path = output_dir / "raw_chat.csv"
        self.db_path = output_dir / "raw_chat_capture.sqlite3"
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS captured_events (
                event_id TEXT PRIMARY KEY,
                captured_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDS).writeheader()

    def normalize(self, event: dict[str, Any]) -> dict[str, str]:
        room_id = clean(
            event.get("room_id")
            or event.get("chat_id")
            or event.get("conversation_id")
        )
        sender_id = clean(
            event.get("sender_id")
            or event.get("message_sender_id")
            or event.get("actor_id")
        )
        participant_id = clean(
            event.get("participant_id")
            or event.get("client_id")
        )
        message = clean(event.get("message") or event.get("text"))
        source_timestamp = clean(
            event.get("source_timestamp")
            or event.get("timestamp")
            or event.get("time")
        )
        source_row_num = clean(event.get("source_row_num") or event.get("row_num"))
        event_id = clean(event.get("event_id"))

        if not event_id:
            material = "|".join(
                [room_id, sender_id, participant_id, source_timestamp, source_row_num, message]
            )
            event_id = hashlib.sha256(material.encode("utf-8")).hexdigest()

        return {
            "event_id": event_id,
            "captured_at": utc_now(),
            "source_timestamp": source_timestamp,
            "room_id": room_id,
            "room_name": clean(event.get("room_name") or event.get("room")),
            "sender_id": sender_id,
            "sender_name": clean(event.get("sender_name") or event.get("sender")),
            "participant_id": participant_id,
            "message": message,
            "source_row_num": source_row_num,
        }

    def append(self, row: dict[str, str]) -> bool:
        with self.lock:
            existing = self.connection.execute(
                "SELECT 1 FROM captured_events WHERE event_id = ?",
                (row["event_id"],),
            ).fetchone()
            if existing:
                return False

            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDS).writerow(row)

            self.connection.execute(
                "INSERT INTO captured_events (event_id, captured_at) VALUES (?, ?)",
                (row["event_id"], row["captured_at"]),
            )
            self.connection.commit()
            return True

    def count(self) -> int:
        with self.lock:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM captured_events"
            ).fetchone()
            return int(row[0])


def make_handler(store: RawChatStore) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(
                    200,
                    {
                        "status": "ok",
                        "captured_events": store.count(),
                        "jsonl": str(store.jsonl_path),
                    },
                )
                return
            self.send_json(404, {"error": "Use POST /ingest or GET /health"})

        def do_POST(self) -> None:
            if self.path != "/ingest":
                self.send_json(404, {"error": "Use POST /ingest"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
            except Exception as exc:
                self.send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            if not isinstance(payload, dict):
                self.send_json(400, {"error": "JSON body must be an object"})
                return

            row = store.normalize(payload)
            if not row["message"]:
                self.send_json(202, {"captured": False, "reason": "empty message"})
                return
            if not row["room_id"]:
                self.send_json(422, {"captured": False, "reason": "missing room_id/chat_id"})
                return
            if not row["sender_id"]:
                self.send_json(422, {"captured": False, "reason": "missing sender_id"})
                return

            inserted = store.append(row)
            print(
                f'{row["source_timestamp"]} room={row["room_id"]} '
                f'sender={row["sender_id"]} message={row["message"]}',
                flush=True,
            )
            self.send_json(
                200,
                {
                    "captured": inserted,
                    "duplicate": not inserted,
                    "event_id": row["event_id"],
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ChatHandler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    store = RawChatStore(Path(args.output_dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Raw chat capture listening on http://{args.host}:{args.port}/ingest")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"JSONL output: {store.jsonl_path}")
    print(f"CSV output:   {store.csv_path}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
