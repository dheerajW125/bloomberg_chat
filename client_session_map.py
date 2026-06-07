#!/usr/bin/env python3
"""Map real-time chat messages into client-centered sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
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

URL_RE = re.compile(r"(?:https?|ftp|bloomberg)://|www\.|mailto:|<GO>|{GO}", re.I)
UPPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9.\-]{1,9}\b")
SIDE_TICKER_RE = re.compile(
    r"\b(?:b|s|buy|sell|bought|sold)\s+"
    r"(?:\d+(?:[.,]\d+)?\s*(?:k|m|mm|mn|pcs?|shares|shs)?\s+)?"
    r"(?P<ticker>[A-Za-z][A-Za-z0-9.\-]{1,9})\b",
    re.I,
)
STOP_TOKENS = {
    "BUY",
    "SELL",
    "BOUGHT",
    "SOLD",
    "WANT",
    "TICKER",
    "LIMIT",
    "ORDER",
    "CLIENT",
    "TRADER",
    "HTTP",
    "HTTPS",
    "WWW",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def jsonl_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".jsonl":
        raise argparse.ArgumentTypeError("path must be a .jsonl file")
    return path


def actor_id(event: dict[str, Any], field: str) -> str:
    if field:
        return clean(event.get(field))
    return clean(
        event.get("sender_id")
        or event.get("message_sender_id")
        or event.get("actor_id")
        or event.get("sender")
    )


def room_id(event: dict[str, Any]) -> str:
    return clean(
        event.get("room_id")
        or event.get("chat_id")
        or event.get("room_name")
        or event.get("room")
        or "UNKNOWN_ROOM"
    )


def message_text(event: dict[str, Any]) -> str:
    return clean(event.get("message") or event.get("raw_message") or event.get("text"))


def timestamp(event: dict[str, Any]) -> str:
    return clean(
        event.get("source_timestamp")
        or event.get("captured_at")
        or event.get("created_at")
        or event.get("timestamp")
    )


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def event_key(event: dict[str, Any], author_id: str, message: str) -> str:
    explicit = clean(event.get("event_id") or event.get("source_event_id"))
    if explicit:
        return explicit
    material = "|".join(
        [
            room_id(event),
            author_id,
            timestamp(event),
            clean(event.get("source_row_num") or event.get("row_num")),
            message,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def extract_order_tickers(event: dict[str, Any]) -> set[str]:
    tickers: set[str] = set()
    raw_orders = event.get("orders_json")
    if raw_orders:
        try:
            orders = json.loads(raw_orders)
        except json.JSONDecodeError:
            orders = []
        if isinstance(orders, list):
            for order in orders:
                if isinstance(order, dict) and clean(order.get("ticker")):
                    tickers.add(clean(order.get("ticker")).upper())
    for field in ("candidate_tickers", "tickers", "matched_tickers"):
        raw = event.get(field)
        if isinstance(raw, list):
            tickers.update(clean(item).upper() for item in raw if clean(item))
        elif isinstance(raw, str):
            for token in re.split(r"[|,\s]+", raw):
                token = clean(token).upper()
                if token and token not in STOP_TOKENS:
                    tickers.add(token)
    return tickers


def extract_message_tickers(message: str) -> set[str]:
    tickers = {
        match.group("ticker").upper()
        for match in SIDE_TICKER_RE.finditer(message)
        if match.group("ticker").upper() not in STOP_TOKENS
    }
    tickers.update(
        token.upper()
        for token in UPPER_TOKEN_RE.findall(message)
        if token.upper() not in STOP_TOKENS
    )
    return tickers


def message_record(
    event: dict[str, Any],
    author_id: str,
    message_class: str,
    id_field: str,
) -> dict[str, Any]:
    message = message_text(event)
    return {
        "event_id": event_key(event, author_id, message),
        "timestamp": timestamp(event),
        "room_id": room_id(event),
        "sender_id": author_id,
        "sender_name": clean(event.get("sender_name") or event.get("sender")),
        "message_class": message_class,
        "message": message,
        "tickers": sorted(extract_order_tickers(event) | extract_message_tickers(message)),
        "raw": event,
    }


def read_new_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"Expected JSON object at {path}:{line_number}")
            events.append(value)
        return events, handle.tell()


class ClientSessionMapper:
    def __init__(
        self,
        client_ids: set[str],
        id_field: str,
        output: Path,
        active_window_minutes: float,
        emit_updates: bool,
    ) -> None:
        self.client_ids = client_ids
        self.id_field = id_field
        self.output = output
        self.active_window_seconds = active_window_minutes * 60
        self.emit_updates = emit_updates
        self.sessions: dict[str, dict[str, Any]] = {}
        self.active_sessions_by_room: dict[str, list[str]] = {}
        self.session_counter = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.touch(exist_ok=True)

    def process(self, event: dict[str, Any]) -> bool:
        author_id = actor_id(event, self.id_field)
        message = message_text(event)
        if not author_id or not message:
            return False
        if URL_RE.search(message):
            return False
        if author_id in self.client_ids:
            session = self._start_session(event, author_id)
            self._write_update(session, "client_message")
            return True

        session = self._select_session(event)
        if not session:
            return False
        session["trader_messages"].append(
            message_record(event, author_id, "trader", self.id_field)
        )
        session["updated_at"] = timestamp(event) or utc_now()
        session["message_count"] = (
            len(session["client_messages"]) + len(session["trader_messages"])
        )
        session["tickers"] = sorted(
            set(session["tickers"]) | extract_message_tickers(message)
        )
        self._write_update(session, "trader_message")
        return True

    def _start_session(self, event: dict[str, Any], client_id: str) -> dict[str, Any]:
        self.session_counter += 1
        room = room_id(event)
        opened_at = timestamp(event) or utc_now()
        record = message_record(event, client_id, "client", self.id_field)
        session_material = "|".join([room, client_id, opened_at, record["event_id"]])
        session_id = "CS-" + hashlib.sha256(
            session_material.encode("utf-8")
        ).hexdigest()[:16]
        session = {
            "record_type": "client_chat_session",
            "session_id": session_id,
            "session_sequence": self.session_counter,
            "room_id": room,
            "client_id": client_id,
            "opened_at": opened_at,
            "updated_at": opened_at,
            "status": "active",
            "tickers": record["tickers"],
            "message_count": 1,
            "client_messages": [record],
            "trader_messages": [],
        }
        self.sessions[session_id] = session
        self.active_sessions_by_room.setdefault(room, []).append(session_id)
        return session

    def _select_session(self, event: dict[str, Any]) -> dict[str, Any] | None:
        room = room_id(event)
        session_ids = self.active_sessions_by_room.get(room, [])
        if not session_ids:
            return None
        message = message_text(event)
        message_tickers = extract_message_tickers(message)
        event_time = parse_time(timestamp(event))

        active_sessions = [
            self.sessions[session_id]
            for session_id in session_ids
            if self._is_active(self.sessions[session_id], event_time)
        ]
        if not active_sessions:
            return None
        if message_tickers:
            for session in reversed(active_sessions):
                if message_tickers & set(session["tickers"]):
                    return session
        return active_sessions[-1]

    def _is_active(self, session: dict[str, Any], event_time: datetime | None) -> bool:
        if self.active_window_seconds <= 0 or event_time is None:
            return True
        updated_at = parse_time(clean(session.get("updated_at")))
        if updated_at is None:
            return True
        return (event_time - updated_at).total_seconds() <= self.active_window_seconds

    def _write_update(self, session: dict[str, Any], update_type: str) -> None:
        if not self.emit_updates:
            return
        self._write(session, update_type)

    def _write(self, session: dict[str, Any], update_type: str) -> None:
        output = {
            **session,
            "update_type": update_type,
            "generated_at": utc_now(),
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    def write_final_sessions(self) -> None:
        for session in sorted(
            self.sessions.values(),
            key=lambda item: (clean(item.get("opened_at")), item["session_sequence"]),
        ):
            self._write(session, "final_session")


def parse_client_ids(value: str) -> set[str]:
    ids = set(DEFAULT_CLIENT_IDS)
    ids.update(item.strip() for item in value.split(",") if item.strip())
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=jsonl_path)
    parser.add_argument(
        "--output",
        type=jsonl_path,
        default=Path(__file__).resolve().parent / "client_chat_sessions.jsonl",
    )
    parser.add_argument(
        "--client-ids",
        default="",
        help="Comma-separated client sender IDs. Built-in client IDs are included.",
    )
    parser.add_argument(
        "--actor-id-field",
        default="sender_id",
        help="Field containing the message sender ID",
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--active-window-minutes",
        type=float,
        default=480.0,
        help="Attach trader replies to active client sessions within this window. Use 0 for no expiry.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output instead of replacing it on batch runs.",
    )
    parser.add_argument(
        "--emit-updates",
        action="store_true",
        help="In batch mode, write every session update instead of final sessions only.",
    )
    args = parser.parse_args()

    if args.active_window_minutes < 0:
        parser.error("--active-window-minutes must be zero or greater")
    if not args.follow and not args.append:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    mapper = ClientSessionMapper(
        parse_client_ids(args.client_ids),
        args.actor_id_field,
        args.output,
        args.active_window_minutes,
        args.follow or args.emit_updates,
    )
    offset = 0
    processed = 0
    while True:
        events, offset = read_new_events(args.input, offset)
        events.sort(key=timestamp)
        for event in events:
            if mapper.process(event):
                processed += 1
        if not args.follow:
            break
        time.sleep(args.poll_interval)

    if not args.follow and not args.emit_updates:
        mapper.write_final_sessions()

    print(f"Client session output: {args.output}")
    print(f"Mapped messages: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
