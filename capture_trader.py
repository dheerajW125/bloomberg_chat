#!/usr/bin/env python3
"""
Discover and persist trader IDs from raw group-chat events.

This pipeline classifies message authors as:
  - trader
  - client
  - unknown/review

Automated content is classified at message level because automated and trader
messages can share the same sender ID.

Important: input must contain messages from all room participants, not only the
known client IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRADE_SIDE_RE = re.compile(
    r"\b(?:b|s|buy|sell|bought|sold|buyer|seller|bid|offer|lift|hit|"
    r"compra|comprar|vende|vender)\b",
    re.IGNORECASE,
)
QUANTITY_RE = re.compile(
    r"(?:[$€£]\s*)?\b\d+(?:[.,]\d+)?\s*(?:k|m|mm|mn|pc|pcs|shares|shs|usd)?\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"(?:@|\bat\b|\bfor\b)\s*\d+(?:\.\d+)?", re.IGNORECASE)
COMPACT_EXECUTION_RE = re.compile(
    r"\b(?:B|S|BUY|SELL|BOUGHT|SOLD)\s*\d+(?:[.,]\d+)?\s*(?:K|M|MM|MN)?\s+"
    r"[A-Z][A-Z0-9.\-]{0,9}(?:\s*(?:@|FOR|AT)\s*\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)
URL_RE = re.compile(
    r"(?:https?|ftp|bloomberg)://|www\.|mailto:|<GO>|{GO}",
    re.IGNORECASE,
)
NEWS_RE = re.compile(
    r"\b(?:research|news|report|reported|breaking|headline|earnings|outlook|"
    r"conference call|market update|morning note|closing note|rsvp|webinar|"
    r"analyst|forecast|reuters|bloomberg|cnbc)\b",
    re.IGNORECASE,
)
AUTOMATION_PREFIX_RE = re.compile(
    r"^\s*(?:\*|#|alert:|news:|research:|in \d+ mins?:|early movers|indications)",
    re.IGNORECASE,
)
TRADER_DESK_REPLY_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|np|no problem|see|working(?: on it)?|on it|done|filled|"
    r"noted|copy|got it|will do|yes|yep|sure|thanks|thank you|tks|same for|"
    r"we are good|good catch)(?:\W+(?:thanks|thank you|tks|yes|yep|"
    r"sure|done|noted|np|no problem))*\W*$",
    re.IGNORECASE,
)

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
CONVERSATIONAL_RE = re.compile(
    r"^\s*(?:ok|okay|np|see|done|thanks|thank you|tks|yes|no|morning|hi|hello|"
    r"good catch|we are good|same for)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() != ".jsonl":
        raise SystemExit("Input must be a JSONL file")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def actor_id(event: dict[str, Any], field: str) -> str:
    if field:
        return clean(event.get(field))
    return clean(
        event.get("sender_id")
        or event.get("message_sender_id")
        or event.get("actor_id")
        or event.get("sender")
    )


def actor_name(event: dict[str, Any]) -> str:
    return clean(event.get("sender_name") or event.get("sender"))


def room_id(event: dict[str, Any]) -> str:
    return clean(
        event.get("room_id")
        or event.get("chat_id")
        or event.get("room_name")
        or event.get("room")
        or "UNKNOWN_ROOM"
    )


def event_key(event: dict[str, Any], author_id: str, message: str) -> str:
    explicit = clean(event.get("event_id"))
    if explicit:
        return explicit
    material = "|".join(
        [
            room_id(event),
            author_id,
            clean(event.get("source_timestamp")),
            clean(event.get("source_row_num") or event.get("row_num")),
            message,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def classify_message(message: str) -> dict[str, Any]:
    length = len(message)
    words = message.split()
    compact_legs = len(COMPACT_EXECUTION_RE.findall(message))
    has_side = bool(TRADE_SIDE_RE.search(message))
    has_quantity = bool(QUANTITY_RE.search(message))
    has_price = bool(PRICE_RE.search(message))
    has_url = bool(URL_RE.search(message))
    has_news = bool(NEWS_RE.search(message))
    automation_prefix = bool(AUTOMATION_PREFIX_RE.search(message))
    conversational = bool(CONVERSATIONAL_RE.search(message))
    desk_reply = bool(TRADER_DESK_REPLY_RE.fullmatch(message))
    letters = [char for char in message if char.isalpha()]
    uppercase_ratio = (
        sum(char.isupper() for char in letters) / len(letters)
        if letters
        else 0.0
    )
    headline_style = length >= 30 and uppercase_ratio >= 0.72

    trader_score = 0.0
    automated_score = 0.0
    reasons: list[str] = []

    if compact_legs >= 2:
        trader_score += 7
        reasons.append("multi_leg_execution")
    elif compact_legs == 1:
        trader_score += 5
        reasons.append("compact_trade_execution")

    if has_side and has_quantity:
        trader_score += 4
        reasons.append("side_and_quantity")
    elif has_side:
        trader_score += 1.5
        reasons.append("trade_side_language")

    if has_price and (has_side or compact_legs):
        trader_score += 2
        reasons.append("trade_price")

    if conversational and length <= 80:
        trader_score += 0.25
        reasons.append("short_conversation")

    if desk_reply:
        trader_score += 6
        reasons.append("trader_desk_reply")

    if has_url:
        automated_score += 12
        reasons.append("url_or_terminal_link")
    if automation_prefix:
        automated_score += 8
        reasons.append("automation_prefix")
    if has_news:
        automated_score += 3
        reasons.append("news_or_research_language")
    if headline_style:
        automated_score += 5
        reasons.append("headline_style")
    if length >= 300:
        automated_score += 5
        reasons.append("very_long_message")
    elif length >= 160:
        automated_score += 2.5
        reasons.append("long_message")
    if len(words) >= 35 and not has_side:
        automated_score += 2
        reasons.append("article_style_text")

    headline_is_automated = bool(
        headline_style
        and compact_legs == 0
        and not (has_side and has_quantity)
    )
    news_text_is_automated = bool(
        has_news
        and length >= 50
        and compact_legs == 0
        and not (has_side and has_quantity)
    )
    message_is_automated = bool(
        has_url
        or automation_prefix
        or headline_is_automated
        or news_text_is_automated
        or (length >= 160 and (has_news or len(words) >= 35))
        or automated_score >= 8
    )
    message_is_trader = bool(
        not message_is_automated
        and (desk_reply or compact_legs >= 1 or (has_side and has_quantity))
    )

    return {
        "trader_score": trader_score,
        "automated_score": automated_score,
        "trade_evidence": int(message_is_trader),
        "automated_evidence": int(message_is_automated),
        "message_is_trader": message_is_trader,
        "message_is_automated": message_is_automated,
        "desk_reply": desk_reply,
        "reasons": reasons,
    }


class RegistryStore:
    def __init__(
        self,
        db_path: Path,
        trader_threshold: float,
        automated_threshold: float,
        min_trade_messages: int,
        client_ids: set[str],
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.trader_threshold = trader_threshold
        self.automated_threshold = automated_threshold
        self.min_trade_messages = min_trade_messages
        self.client_ids = client_ids
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS actor_profiles (
                actor_id TEXT PRIMARY KEY,
                actor_name TEXT NOT NULL DEFAULT '',
                classification TEXT NOT NULL DEFAULT 'unknown',
                locked INTEGER NOT NULL DEFAULT 0,
                message_count INTEGER NOT NULL DEFAULT 0,
                trade_message_count INTEGER NOT NULL DEFAULT 0,
                automated_message_count INTEGER NOT NULL DEFAULT 0,
                trader_score REAL NOT NULL DEFAULT 0,
                automated_score REAL NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                last_room_id TEXT NOT NULL DEFAULT '',
                last_reason TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS processed_events (
                event_key TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def already_processed(self, key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM processed_events WHERE event_key = ?",
            (key,),
        ).fetchone()
        return row is not None

    def update(
        self,
        author_id: str,
        name: str,
        event_room_id: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        row = self.connection.execute(
            "SELECT * FROM actor_profiles WHERE actor_id = ?",
            (author_id,),
        ).fetchone()

        if row is None:
            profile = {
                "actor_id": author_id,
                "actor_name": name,
                "classification": "unknown",
                "locked": 0,
                "message_count": 0,
                "trade_message_count": 0,
                "automated_message_count": 0,
                "trader_score": 0.0,
                "automated_score": 0.0,
                "first_seen": now,
                "last_seen": now,
                "last_room_id": event_room_id,
                "last_reason": "",
            }
        else:
            profile = dict(row)

        profile["actor_name"] = name or profile["actor_name"]
        profile["message_count"] += 1
        profile["trade_message_count"] += evidence["trade_evidence"]
        profile["automated_message_count"] += evidence["automated_evidence"]
        profile["trader_score"] += evidence["trader_score"]
        profile["automated_score"] += evidence["automated_score"]
        profile["last_seen"] = now
        profile["last_room_id"] = event_room_id
        profile["last_reason"] = "|".join(evidence["reasons"])

        if author_id in self.client_ids:
            profile["classification"] = "client"
            profile["locked"] = 1
        else:
            # Non-client actors may post both desk replies and automated/news
            # content, so automation is classified only at message level.
            profile["locked"] = 0
            if evidence["message_is_trader"]:
                profile["classification"] = "trader"
            else:
                inferred = self._classification(profile)
                if profile["classification"] != "trader":
                    profile["classification"] = inferred

        self.connection.execute(
            """
            INSERT INTO actor_profiles (
                actor_id, actor_name, classification, locked, message_count,
                trade_message_count, automated_message_count, trader_score,
                automated_score, first_seen, last_seen, last_room_id, last_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(actor_id) DO UPDATE SET
                actor_name = excluded.actor_name,
                classification = excluded.classification,
                locked = excluded.locked,
                message_count = excluded.message_count,
                trade_message_count = excluded.trade_message_count,
                automated_message_count = excluded.automated_message_count,
                trader_score = excluded.trader_score,
                automated_score = excluded.automated_score,
                last_seen = excluded.last_seen,
                last_room_id = excluded.last_room_id,
                last_reason = excluded.last_reason
            """,
            tuple(profile[key] for key in [
                "actor_id",
                "actor_name",
                "classification",
                "locked",
                "message_count",
                "trade_message_count",
                "automated_message_count",
                "trader_score",
                "automated_score",
                "first_seen",
                "last_seen",
                "last_room_id",
                "last_reason",
            ]),
        )
        self.connection.commit()
        return profile

    def mark_processed(self, key: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO processed_events (event_key, processed_at) VALUES (?, ?)",
            (key, utc_now()),
        )
        self.connection.commit()

    def _classification(self, profile: dict[str, Any]) -> str:
        trader_margin = profile["trader_score"] - profile["automated_score"]
        if (
            profile["trade_message_count"] >= self.min_trade_messages
            and profile["trader_score"] >= self.trader_threshold
            and trader_margin >= 4
        ):
            return "trader"

        return "unknown"

    def profiles(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM actor_profiles ORDER BY classification, actor_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.connection.close()


class OutputWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trader_jsonl = output_dir / "trader_messages.jsonl"
        self.automated_jsonl = output_dir / "automated_messages.jsonl"
        self.client_jsonl = output_dir / "client_messages.jsonl"
        self.review_jsonl = output_dir / "trader_classification_review.jsonl"
        for path in (
            self.trader_jsonl,
            self.automated_jsonl,
            self.client_jsonl,
            self.review_jsonl,
        ):
            path.touch(exist_ok=True)

    def append_event(
        self,
        event: dict[str, Any],
        profile: dict[str, Any],
        evidence: dict[str, Any],
        message: str,
    ) -> None:
        enriched = {
            **event,
            "actor_id": profile["actor_id"],
            "actor_classification": profile["classification"],
            "message_classification": self.message_classification(profile, evidence),
            "actor_trader_score": profile["trader_score"],
            "actor_automated_score": profile["automated_score"],
            "classification_reasons": evidence["reasons"],
            "message": message,
        }
        target = {
            "trader": self.trader_jsonl,
            "automated": self.automated_jsonl,
            "client": self.client_jsonl,
            "unknown": self.review_jsonl,
        }[enriched["message_classification"]]
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    @staticmethod
    def message_classification(
        profile: dict[str, Any],
        evidence: dict[str, Any],
    ) -> str:
        if profile["classification"] == "client":
            return "client"
        if evidence["message_is_automated"]:
            return "automated"
        if evidence["message_is_trader"] or profile["classification"] == "trader":
            return "trader"
        return "unknown"


def process_event(
    event: dict[str, Any],
    id_field: str,
    store: RegistryStore,
    writer: OutputWriter,
) -> bool:
    message = clean(event.get("message"))
    author_id = actor_id(event, id_field)
    if not message or not author_id:
        return False

    key = event_key(event, author_id, message)
    if store.already_processed(key):
        return False

    evidence = classify_message(message)
    profile = store.update(author_id, actor_name(event), room_id(event), evidence)
    store.mark_processed(key)
    writer.append_event(event, profile, evidence, message)
    message_class = writer.message_classification(profile, evidence)
    print(
        f'actor_id={author_id} actor_class={profile["classification"]} '
        f'message_class={message_class} '
        f'trader_score={profile["trader_score"]:.2f} '
        f'automated_score={profile["automated_score"]:.2f} message={message}',
        flush=True,
    )
    return True


def run_batch(args: argparse.Namespace, store: RegistryStore, writer: OutputWriter) -> None:
    count = 0
    for event in read_events(Path(args.input)):
        if process_event(event, args.actor_id_field, store, writer):
            count += 1
    print(f"Processed new events: {count}")


def run_follow(args: argparse.Namespace, store: RegistryStore, writer: OutputWriter) -> None:
    input_path = Path(args.input)
    if input_path.suffix.lower() != ".jsonl":
        raise SystemExit("--follow requires JSONL input")

    while not input_path.exists():
        print(f"Waiting for input file: {input_path}", flush=True)
        time.sleep(args.poll_interval)

    with input_path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                time.sleep(args.poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            process_event(json.loads(line), args.actor_id_field, store, writer)


def parse_ids(value: str) -> set[str]:
    ids = set(DEFAULT_CLIENT_IDS)
    ids.update(item.strip() for item in value.split(",") if item.strip())
    return ids


def rebuild_outputs(output_dir: Path, registry_db: Path) -> None:
    for name in (
        "client_messages.jsonl",
        "trader_messages.jsonl",
        "automated_messages.jsonl",
        "trader_classification_review.jsonl",
        # Remove exports produced by older versions of this pipeline.
        "client_registry.csv",
        "trader_registry.csv",
        "automated_registry.csv",
        "actor_review_registry.csv",
        "trader_ids.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    if registry_db.exists():
        registry_db.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw all-participant chat JSONL")
    parser.add_argument(
        "--actor-id-field",
        default="sender_id",
        help="Field containing the actual message author/trader ID",
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--trader-threshold", type=float, default=10.0)
    parser.add_argument("--automated-threshold", type=float, default=12.0)
    parser.add_argument("--min-trade-messages", type=int, default=2)
    parser.add_argument(
        "--additional-client-ids",
        default="",
        help="Comma-separated IDs to add to the built-in non-trader client list",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--registry-db",
        default=str(Path(__file__).resolve().parent / "trader_registry.sqlite3"),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear this pipeline's four JSONL outputs and registry DB before reprocessing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    registry_db = Path(args.registry_db)
    if args.rebuild:
        rebuild_outputs(output_dir, registry_db)

    store = RegistryStore(
        registry_db,
        args.trader_threshold,
        args.automated_threshold,
        args.min_trade_messages,
        parse_ids(args.additional_client_ids),
    )
    writer = OutputWriter(output_dir)
    try:
        if args.follow:
            run_follow(args, store, writer)
        else:
            run_batch(args, store, writer)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
