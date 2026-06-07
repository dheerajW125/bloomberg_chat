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


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value)).strip("._")
    return name or "unknown"


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
    with path.open("r", encoding="utf-8-sig") as handle:
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
        client_output_dir: Path | None,
        side_output_dir: Path,
        active_window_minutes: float,
        emit_updates: bool,
    ) -> None:
        self.client_ids = client_ids
        self.id_field = id_field
        self.output = output
        self.client_output_dir = client_output_dir
        self.side_output_dir = side_output_dir
        self.automated_jsonl = side_output_dir / "automated_messages.jsonl"
        self.trader_review_jsonl = side_output_dir / "trader_classification_review.jsonl"
        self.client_review_jsonl = side_output_dir / "client_messages_review.jsonl"
        self.active_window_seconds = active_window_minutes * 60
        self.emit_updates = emit_updates
        self.sessions: dict[str, dict[str, Any]] = {}
        self.active_sessions_by_room: dict[str, list[str]] = {}
        self.session_counter = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.touch(exist_ok=True)
        if self.client_output_dir is not None:
            self.client_output_dir.mkdir(parents=True, exist_ok=True)
        self.side_output_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.automated_jsonl,
            self.trader_review_jsonl,
            self.client_review_jsonl,
        ):
            path.touch(exist_ok=True)

    def process(self, event: dict[str, Any]) -> bool:
        author_id = actor_id(event, self.id_field)
        message = message_text(event)
        if not author_id or not message:
            return False
        if URL_RE.search(message):
            self._write_side_stream(
                self.automated_jsonl,
                event,
                author_id,
                "automated",
                "url_or_terminal_link",
            )
            return True
        if author_id in self.client_ids:
            record = message_record(event, author_id, "client", self.id_field)
            if not record["tickers"]:
                self._write_side_stream(
                    self.client_review_jsonl,
                    event,
                    author_id,
                    "client_review",
                    "client_message_without_ticker",
                )
            session = self._start_session(event, author_id)
            self._write_update(session, "client_message")
            return True

        session, review_reason, candidates = self._select_session(event)
        if not session:
            self._write_side_stream(
                self.trader_review_jsonl,
                event,
                author_id,
                "trader_review",
                review_reason,
                candidates,
            )
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

    def _select_session(
        self,
        event: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
        room = room_id(event)
        session_ids = self.active_sessions_by_room.get(room, [])
        if not session_ids:
            return None, "no_active_client_session_in_room", []
        message = message_text(event)
        message_tickers = extract_message_tickers(message)
        event_time = parse_time(timestamp(event))

        active_sessions = [
            self.sessions[session_id]
            for session_id in session_ids
            if self._is_active(self.sessions[session_id], event_time)
        ]
        if not active_sessions:
            return None, "no_active_client_session_in_time_window", []
        if message_tickers:
            matches = [
                session
                for session in reversed(active_sessions)
                if message_tickers & set(session["tickers"])
            ]
            if len(matches) == 1:
                return matches[0], "", []
            if len(matches) > 1:
                return None, "multiple_matching_client_sessions", matches
            return None, "no_matching_client_session_for_ticker", active_sessions
        return active_sessions[-1], "", []

    @staticmethod
    def _candidate_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "session_id": session["session_id"],
                "client_id": session["client_id"],
                "room_id": session["room_id"],
                "opened_at": session["opened_at"],
                "updated_at": session["updated_at"],
                "tickers": session["tickers"],
            }
            for session in reversed(sessions)
        ]

    def _write_side_stream(
        self,
        path: Path,
        event: dict[str, Any],
        author_id: str,
        message_class: str,
        reason: str,
        candidate_sessions: list[dict[str, Any]] | None = None,
    ) -> None:
        record = message_record(event, author_id, message_class, self.id_field)
        output = {
            **record,
            "message_classification": message_class,
            "review_reason": reason,
            "candidate_sessions": self._candidate_summary(candidate_sessions or []),
            "generated_at": utc_now(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    def _is_active(self, session: dict[str, Any], event_time: datetime | None) -> bool:
        if self.active_window_seconds <= 0 or event_time is None:
            return True
        updated_at = parse_time(clean(session.get("updated_at")))
        if updated_at is None:
            return True
        return (event_time - updated_at).total_seconds() <= self.active_window_seconds

    def _write_update(self, session: dict[str, Any], update_type: str) -> None:
        if self.emit_updates:
            self._write(session, update_type)
        self._write_client_file(clean(session["client_id"]))

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
        self.write_client_files()

    def write_client_files(self) -> None:
        if self.client_output_dir is None:
            return
        client_ids = sorted(
            {clean(session["client_id"]) for session in self.sessions.values()}
        )
        for client_id in client_ids:
            self._write_client_file(client_id)

    def _write_client_file(self, client_id: str) -> None:
        if self.client_output_dir is None:
            return
        sessions = [
            session
            for session in self.sessions.values()
            if clean(session["client_id"]) == client_id
        ]
        sessions.sort(key=lambda item: (clean(item.get("opened_at")), item["session_sequence"]))
        document = {
            "record_type": "client_chat_sessions_by_client",
            "client_id": client_id,
            "generated_at": utc_now(),
            "session_count": len(sessions),
            "sessions": sessions,
        }
        path = self.client_output_dir / f"{safe_filename(client_id)}.json"
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


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
        "--client-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "client_sessions_by_client",
        help="Directory for one normal JSON file per client.",
    )
    parser.add_argument(
        "--side-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory for automated_messages.jsonl, "
            "trader_classification_review.jsonl, and client_messages_review.jsonl."
        ),
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
        if args.client_output_dir.exists():
            for path in args.client_output_dir.glob("*.json"):
                path.unlink()
        args.side_output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "automated_messages.jsonl",
            "trader_classification_review.jsonl",
            "client_messages_review.jsonl",
        ):
            (args.side_output_dir / name).write_text("", encoding="utf-8")

    mapper = ClientSessionMapper(
        parse_client_ids(args.client_ids),
        args.actor_id_field,
        args.output,
        args.client_output_dir,
        args.side_output_dir,
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
    print(f"Per-client JSON output: {args.client_output_dir}")
    print(f"Side stream output: {args.side_output_dir}")
    print(f"Mapped messages: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
