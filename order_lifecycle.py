#!/usr/bin/env python3
"""
Build persistent order lifecycle state from client intents and trader events.

Inputs can be combined JSONL streams or the per-client JSON directories created
by the client-intent and trader-event NLP stages. The primary output is one
detailed JSON document per client containing source messages, trader replies,
order statuses, and the complete lifecycle audit history.
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


OPEN_STATUSES = ("PENDING_ACK", "ACKNOWLEDGED", "WORKING", "PARTIALLY_FILLED", "FILL_REPORTED")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value)).strip("._")
    return name or "unknown"


def event_id(prefix: str, event: dict[str, Any]) -> str:
    raw = json.dumps(event, sort_keys=True, ensure_ascii=False)
    return prefix + "-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def message_text(event: dict[str, Any]) -> str:
    text = (
        event.get("raw_message")
        or event.get("message")
        or event.get("original_message")
        or ""
    )
    if text:
        return str(text)
    try:
        orders = json.loads(event.get("orders_json") or "[]")
    except json.JSONDecodeError:
        return ""
    if not isinstance(orders, list):
        return ""
    parts = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        side = order.get("side", "")
        quantity = order.get("quantity", "")
        unit = order.get("quantity_unit", "")
        ticker = order.get("ticker", "")
        parts.append(f"{side} {quantity}{unit} {ticker}".strip())
    return " | ".join(part for part in parts if part)


def load_client_session_context(
    directory: Path | None,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    messages: dict[tuple[str, str], list[dict[str, Any]]] = {}
    clients: dict[str, dict[str, Any]] = {}
    if directory is None or not directory.exists():
        return messages, clients

    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        client_id = clean(document.get("client_id"))
        clients[client_id] = {
            "client_id": client_id,
            "client_name": clean(document.get("client_name")),
        }
        for session in document.get("sessions") or []:
            room_id = clean(session.get("room_id"))
            key = (client_id, room_id)
            records = messages.setdefault(key, [])
            for item in session.get("messages") or session.get("client_messages") or []:
                raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
                records.append(
                    {
                        "event_id": clean(item.get("event_id")),
                        "session_id": clean(session.get("session_id")),
                        "timestamp": clean(
                            item.get("timestamp")
                            or raw.get("source_timestamp")
                            or raw.get("captured_at")
                        ),
                        "actor_id": clean(
                            item.get("sender_id")
                            or raw.get("actor_id")
                            or raw.get("sender_id")
                            or client_id
                        ),
                        "message": clean(
                            item.get("message")
                            or raw.get("message")
                            or raw.get("original_message")
                        ),
                        "raw": raw or item,
                    }
                )
    return messages, clients


def load_client_intents(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for client_path in sorted(path.glob("*.json")):
            yield from load_client_intents(client_path)
        return
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        client_id = clean(document.get("client_id"))
        client_name = clean(document.get("client_name"))
        for item in document.get("accepted_intents") or []:
            event = dict(item)
            event.setdefault("sender_id", client_id)
            event.setdefault("sender_name", client_name)
            event.setdefault("processing_status", "accepted")
            yield event
        return
    if path.suffix.lower() != ".jsonl":
        raise SystemExit("--client-intents must be a directory, .json, or .jsonl")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_trader_events(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for client_path in sorted(path.glob("*.json")):
            yield from load_trader_events(client_path)
        return
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        client_id = clean(document.get("client_id"))
        client_name = clean(document.get("client_name"))
        for mapping in document.get("mappings") or []:
            raw_event = mapping.get("trader_event")
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            event.setdefault("client_id", client_id)
            event.setdefault("client_name", client_name)
            event.setdefault("match_basis", mapping.get("match_basis"))
            event.setdefault(
                "match_delta_seconds",
                mapping.get("match_delta_seconds"),
            )
            yield event
        return
    if path.suffix.lower() != ".jsonl":
        raise SystemExit("--trader-events must be a directory, .json, or .jsonl")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                if event.get("mapping_status") != "review":
                    yield event


class OrderStore:
    def __init__(
        self,
        path: Path,
        jsonl_path: Path | None = None,
        client_sessions_dir: Path | None = None,
        client_output_dir: Path | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if client_output_dir:
            client_output_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.jsonl_path = jsonl_path
        self.client_sessions_dir = client_sessions_dir
        self.client_output_dir = client_output_dir
        (
            self.client_session_messages,
            self.client_profiles,
        ) = load_client_session_context(
            self.client_sessions_dir
        )
        self.snapshots_written = 0
        if self.jsonl_path:
            self.jsonl_path.touch(exist_ok=True)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY, room_id TEXT, client_id TEXT,
                assigned_trader_id TEXT DEFAULT '', status TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY, batch_id TEXT, ticker TEXT, side TEXT,
                requested_quantity TEXT, quantity_unit TEXT, filled_quantity TEXT DEFAULT '',
                average_price TEXT DEFAULT '', status TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_id TEXT PRIMARY KEY, event_type TEXT, room_id TEXT,
                actor_id TEXT, batch_id TEXT, order_id TEXT, raw_json TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS unmatched_events (
                event_id TEXT PRIMARY KEY, reason TEXT, raw_json TEXT, created_at TEXT
            );
            """
        )
        self.db.commit()

    def reload_client_session_messages(self) -> None:
        (
            self.client_session_messages,
            self.client_profiles,
        ) = load_client_session_context(
            self.client_sessions_dir
        )

    def seen(self, eid: str) -> bool:
        return self.db.execute("SELECT 1 FROM lifecycle_events WHERE event_id=?", (eid,)).fetchone() is not None

    def seen_event_group(self, eid: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM lifecycle_events WHERE event_id=? OR event_id LIKE ? LIMIT 1",
            (eid, eid + "-%"),
        ).fetchone() is not None

    def add_client_intent(self, event: dict[str, Any]) -> None:
        eid = event_id("client", event)
        if self.seen_event_group(eid) or event.get("processing_status") != "accepted":
            return
        orders = json.loads(event.get("orders_json") or "[]")
        if not orders:
            return
        batch_id = "B-" + eid[-16:]
        ts = now()
        self.db.execute(
            "INSERT OR IGNORE INTO batches VALUES (?,?,?,?,?,?,?)",
            (batch_id, event.get("room_id", ""), event.get("sender_id", ""), "", "PENDING_ACK", ts, ts),
        )
        for index, order in enumerate(orders, 1):
            oid = f"{batch_id}-{index}"
            self.db.execute(
                "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    oid, batch_id, order.get("ticker", ""), order.get("side", ""),
                    order.get("quantity", ""), order.get("quantity_unit", ""), "", "",
                    "PENDING_ACK", ts, ts,
                ),
            )
            self._audit(eid + f"-{index}", "CLIENT_ORDER", event.get("room_id", ""), event.get("sender_id", ""), batch_id, oid, event)
            self._write_order_snapshot(oid, "CLIENT_ORDER")
        self.db.commit()

    def add_trader_event(self, event: dict[str, Any]) -> None:
        eid = event_id("trader", event)
        if self.seen_event_group(eid):
            return
        etype = event.get("event_type")
        room = event.get("room_id", "")
        trader = event.get("trader_id", "")
        client_id = event.get("client_id", "")
        tickers = event.get("tickers") or []
        matched_batch_id = self._matched_batch_id(event)

        if etype == "ACK":
            if matched_batch_id:
                row = self.db.execute(
                    """
                    SELECT * FROM batches
                    WHERE batch_id=? AND status='PENDING_ACK'
                    """,
                    (matched_batch_id,),
                ).fetchone()
            elif client_id:
                row = self.db.execute(
                    """
                    SELECT * FROM batches
                    WHERE room_id=? AND client_id=? AND status='PENDING_ACK'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (room, client_id),
                ).fetchone()
            else:
                row = self.db.execute(
                    """
                    SELECT * FROM batches
                    WHERE room_id=? AND status='PENDING_ACK'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (room,),
                ).fetchone()
            if not row:
                return self._unmatched(eid, "no_pending_batch_for_ack", event)
            self.db.execute(
                "UPDATE batches SET assigned_trader_id=?,status='ACKNOWLEDGED',updated_at=? WHERE batch_id=?",
                (trader, now(), row["batch_id"]),
            )
            self.db.execute(
                "UPDATE orders SET status='ACKNOWLEDGED',updated_at=? WHERE batch_id=? AND status='PENDING_ACK'",
                (now(), row["batch_id"]),
            )
            self._audit(eid, "ACK", room, trader, row["batch_id"], "", event)
            for order in self._batch_orders(row["batch_id"]):
                self._write_order_snapshot(order["order_id"], "ACK")
            return self.db.commit()

        candidates = self._find_orders(
            room,
            trader,
            tickers,
            event.get("side", ""),
            client_id,
            matched_batch_id,
        )
        if not candidates:
            return self._unmatched(eid, "no_compatible_open_order", event)

        for order in candidates:
            status = "FILLED" if etype == "FILL" and event.get("full_fill_claim") else etype
            if status == "PARTIAL_FILL":
                status = "PARTIALLY_FILLED"
            if status == "FILL" and not event.get("full_fill_claim"):
                status = "FILL_REPORTED"
            if status == "REJECT":
                status = "REJECTED"
            self.db.execute(
                "UPDATE orders SET status=?,filled_quantity=?,average_price=?,updated_at=? WHERE order_id=?",
                (
                    status,
                    event.get("filled_quantity", ""),
                    event.get("price", ""),
                    now(),
                    order["order_id"],
                ),
            )
            self._audit(eid + "-" + order["order_id"], etype, room, trader, order["batch_id"], order["order_id"], event)
            self._refresh_batch(order["batch_id"])
            self._write_order_snapshot(order["order_id"], etype)
        self.db.commit()

    def _find_orders(
        self,
        room: str,
        trader: str,
        tickers: list[str],
        side: str,
        client_id: str = "",
        batch_id: str = "",
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in OPEN_STATUSES)
        query = f"""
            SELECT o.*,b.assigned_trader_id,b.created_at AS batch_created
            FROM orders o JOIN batches b ON b.batch_id=o.batch_id
            WHERE b.room_id=? AND o.status IN ({placeholders})
        """
        params: list[Any] = [room, *OPEN_STATUSES]
        if batch_id:
            query += " AND b.batch_id=?"
            params.append(batch_id)
        if client_id:
            query += " AND b.client_id=?"
            params.append(client_id)
        if tickers:
            query += " AND o.ticker IN (" + ",".join("?" for _ in tickers) + ")"
            params.extend(tickers)
        if side:
            query += " AND o.side=?"
            params.append(side)
        query += " ORDER BY CASE WHEN b.assigned_trader_id=? THEN 0 ELSE 1 END,batch_created DESC"
        params.append(trader)
        rows = self.db.execute(query, params).fetchall()
        if rows:
            batch = rows[0]["batch_id"]
            rows = [row for row in rows if row["batch_id"] == batch]
        return rows

    def _matched_batch_id(self, event: dict[str, Any]) -> str:
        matched_source_event_id = clean(
            event.get("matched_client_source_event_id")
        )
        if not matched_source_event_id:
            return ""
        rows = self.db.execute(
            """
            SELECT e.batch_id,e.raw_json
            FROM lifecycle_events e
            JOIN batches b ON b.batch_id=e.batch_id
            WHERE e.event_type='CLIENT_ORDER'
              AND b.room_id=?
              AND b.client_id=?
            ORDER BY e.created_at DESC
            """,
            (
                clean(event.get("room_id")),
                clean(event.get("client_id")),
            ),
        ).fetchall()
        for row in rows:
            raw = json.loads(row["raw_json"])
            source_event_id = clean(
                raw.get("source_event_id") or raw.get("event_id")
            )
            if source_event_id == matched_source_event_id:
                return clean(row["batch_id"])
        return ""

    def _refresh_batch(self, batch_id: str) -> None:
        statuses = [row[0] for row in self.db.execute("SELECT status FROM orders WHERE batch_id=?", (batch_id,))]
        if statuses and all(status == "FILLED" for status in statuses):
            status = "FILLED"
        elif any(status in {"FILLED", "PARTIALLY_FILLED", "FILL_REPORTED"} for status in statuses):
            status = "PARTIALLY_FILLED"
        elif all(status == "REJECTED" for status in statuses):
            status = "REJECTED"
        else:
            status = "ACKNOWLEDGED"
        self.db.execute("UPDATE batches SET status=?,updated_at=? WHERE batch_id=?", (status, now(), batch_id))

    def _batch_orders(self, batch_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM orders WHERE batch_id=? ORDER BY order_id",
            (batch_id,),
        ).fetchall()

    def _order_snapshot(self, order_id: str, trigger_event_type: str) -> dict[str, Any] | None:
        order = self.db.execute(
            """
            SELECT o.*, b.room_id, b.client_id, b.assigned_trader_id,
                   b.status AS batch_status, b.created_at AS batch_created_at,
                   b.updated_at AS batch_updated_at
            FROM orders o JOIN batches b ON b.batch_id=o.batch_id
            WHERE o.order_id=?
            """,
            (order_id,),
        ).fetchone()
        if not order:
            return None

        events = self.db.execute(
            """
            SELECT * FROM lifecycle_events
            WHERE order_id=? OR (batch_id=? AND order_id='')
            ORDER BY created_at,event_id
            """,
            (order_id, order["batch_id"]),
        ).fetchall()
        client_messages: list[dict[str, Any]] = []
        normalized_client_intents: list[dict[str, Any]] = []
        trader_messages: list[dict[str, Any]] = []
        lifecycle_events: list[dict[str, Any]] = []

        client_messages.extend(
            self.client_session_messages.get(
                (clean(order["client_id"]), clean(order["room_id"])),
                [],
            )
        )
        has_client_session_context = bool(client_messages)
        for event in events:
            raw = json.loads(event["raw_json"])
            message_record = {
                "event_id": clean(
                    raw.get("event_id")
                    or raw.get("source_event_id")
                    or event["event_id"]
                ),
                "event_type": event["event_type"],
                "timestamp": raw.get("source_timestamp") or raw.get("captured_at") or "",
                "actor_id": event["actor_id"],
                "message": message_text(raw),
                "raw": raw,
            }
            lifecycle_events.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "actor_id": event["actor_id"],
                    "created_at": event["created_at"],
                    "raw": raw,
                }
            )
            if event["event_type"] == "CLIENT_ORDER":
                normalized_client_intents.append(message_record)
                if not has_client_session_context:
                    client_messages.append(message_record)
            else:
                trader_messages.append(message_record)

        client_messages = self._deduplicate_messages(client_messages)
        normalized_client_intents = self._deduplicate_messages(
            normalized_client_intents
        )
        trader_messages = self._deduplicate_messages(trader_messages)
        return {
            "record_type": "order_lifecycle_snapshot",
            "generated_at": now(),
            "trigger_event_type": trigger_event_type,
            "batch_id": order["batch_id"],
            "order_id": order["order_id"],
            "room_id": order["room_id"],
            "client_id": order["client_id"],
            "assigned_trader_id": order["assigned_trader_id"],
            "batch_status": order["batch_status"],
            "order_status": order["status"],
            "ticker": order["ticker"],
            "side": order["side"],
            "requested_quantity": order["requested_quantity"],
            "quantity_unit": order["quantity_unit"],
            "filled_quantity": order["filled_quantity"],
            "average_price": order["average_price"],
            "order_created_at": order["created_at"],
            "order_updated_at": order["updated_at"],
            "client_messages": client_messages,
            "normalized_client_intents": normalized_client_intents,
            "trader_messages": trader_messages,
            "lifecycle_events": lifecycle_events,
        }

    @staticmethod
    def _deduplicate_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in messages:
            key = clean(message.get("event_id"))
            if not key:
                key = "|".join(
                    [
                        clean(message.get("timestamp")),
                        clean(message.get("actor_id")),
                        clean(message.get("message")),
                    ]
                )
            if key in seen:
                continue
            seen.add(key)
            unique.append(message)
        unique.sort(
            key=lambda item: (
                0 if parse_timestamp(item.get("timestamp")) else 1,
                parse_timestamp(item.get("timestamp"))
                or datetime.max.replace(tzinfo=timezone.utc),
                clean(item.get("event_id")),
            )
        )
        return unique

    def _write_order_snapshot(self, order_id: str, trigger_event_type: str) -> None:
        snapshot = self._order_snapshot(order_id, trigger_event_type)
        if not snapshot:
            return
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            self.snapshots_written += 1
        self._write_client_order_file(clean(snapshot["client_id"]))

    def _write_client_order_file(self, client_id: str) -> None:
        if not self.client_output_dir or not client_id:
            return
        rows = self.db.execute(
            """
            SELECT o.order_id
            FROM orders o JOIN batches b ON b.batch_id=o.batch_id
            WHERE b.client_id=?
            ORDER BY b.created_at,o.order_id
            """,
            (client_id,),
        ).fetchall()
        orders = [
            snapshot
            for row in rows
            if (
                snapshot := self._order_snapshot(
                    row["order_id"],
                    "CLIENT_ORDER_HISTORY",
                )
            )
        ]
        client_messages: list[dict[str, Any]] = []
        for (message_client_id, _), records in self.client_session_messages.items():
            if message_client_id == client_id:
                client_messages.extend(records)
        client_messages = self._deduplicate_messages(client_messages)

        event_rows = self.db.execute(
            """
            SELECT e.*
            FROM lifecycle_events e
            JOIN batches b ON b.batch_id=e.batch_id
            WHERE b.client_id=?
            ORDER BY e.created_at,e.event_id
            """,
            (client_id,),
        ).fetchall()
        normalized_client_intents: list[dict[str, Any]] = []
        trader_messages: list[dict[str, Any]] = []
        lifecycle_history: list[dict[str, Any]] = []
        for event in event_rows:
            raw = json.loads(event["raw_json"])
            message = {
                "event_id": clean(
                    raw.get("event_id")
                    or raw.get("source_event_id")
                    or event["event_id"]
                ),
                "event_type": event["event_type"],
                "timestamp": clean(
                    raw.get("source_timestamp") or raw.get("captured_at")
                ),
                "room_id": event["room_id"],
                "actor_id": event["actor_id"],
                "message": message_text(raw),
                "batch_id": event["batch_id"],
                "order_id": event["order_id"],
                "raw": raw,
            }
            lifecycle_history.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "source_timestamp": message["timestamp"],
                    "processed_at": event["created_at"],
                    "room_id": event["room_id"],
                    "actor_id": event["actor_id"],
                    "batch_id": event["batch_id"],
                    "order_id": event["order_id"],
                    "raw": raw,
                }
            )
            if event["event_type"] == "CLIENT_ORDER":
                normalized_client_intents.append(message)
            else:
                trader_messages.append(message)

        normalized_client_intents = self._deduplicate_messages(
            normalized_client_intents
        )
        trader_messages = self._deduplicate_messages(trader_messages)
        conversation_timeline = [
            {
                "message_type": "CLIENT_MESSAGE",
                **message,
            }
            for message in client_messages
        ]
        conversation_timeline.extend(
            {
                "message_type": "TRADER_MESSAGE",
                **message,
            }
            for message in trader_messages
        )
        conversation_timeline = self._deduplicate_messages(
            conversation_timeline
        )

        status_summary: dict[str, int] = {}
        for order in orders:
            status = clean(order.get("order_status")) or "UNKNOWN"
            status_summary[status] = status_summary.get(status, 0) + 1
        batch_count = len(
            {
                clean(order.get("batch_id"))
                for order in orders
                if clean(order.get("batch_id"))
            }
        )
        client_profile = self.client_profiles.get(client_id, {})
        document = {
            "record_type": "client_order_lifecycle",
            "client_id": client_id,
            "client_name": clean(client_profile.get("client_name")),
            "generated_at": now(),
            "batch_count": batch_count,
            "order_count": len(orders),
            "order_status_summary": status_summary,
            "client_message_count": len(client_messages),
            "trader_message_count": len(trader_messages),
            "client_messages": client_messages,
            "normalized_client_intents": normalized_client_intents,
            "trader_messages": trader_messages,
            "conversation_timeline": conversation_timeline,
            "lifecycle_history": lifecycle_history,
            "orders": orders,
        }
        path = self.client_output_dir / f"{safe_filename(client_id)}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def write_all_client_order_files(self) -> None:
        if not self.client_output_dir:
            return
        rows = self.db.execute(
            "SELECT DISTINCT client_id FROM batches ORDER BY client_id"
        ).fetchall()
        for row in rows:
            self._write_client_order_file(clean(row["client_id"]))

    def write_all_order_snapshots(self, trigger_event_type: str) -> None:
        rows = self.db.execute(
            "SELECT order_id FROM orders ORDER BY batch_id,order_id"
        ).fetchall()
        for row in rows:
            self._write_order_snapshot(row["order_id"], trigger_event_type)

    def _audit(self, eid: str, etype: str, room: str, actor: str, batch: str, order: str, event: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO lifecycle_events VALUES (?,?,?,?,?,?,?,?)",
            (eid, etype, room, actor, batch, order, json.dumps(event, ensure_ascii=False), now()),
        )

    def _unmatched(self, eid: str, reason: str, event: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO unmatched_events VALUES (?,?,?,?)",
            (eid, reason, json.dumps(event, ensure_ascii=False), now()),
        )
        self.db.commit()


def drain(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
        return events, handle.tell()


def parse_timestamp(value: Any) -> datetime | None:
    text = clean(value)
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
        for pattern in (
            "%Y.%m.%dT%H:%M:%S.%f",
            "%Y.%m.%dT%H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_key(event: dict[str, Any]) -> tuple[int, datetime, str]:
    text = clean(event.get("source_timestamp") or event.get("captured_at"))
    parsed = parse_timestamp(text)
    return (
        0 if parsed else 1,
        parsed or datetime.max.replace(tzinfo=timezone.utc),
        text,
    )


def event_source_path(value: str) -> Path:
    path = Path(value)
    if path.exists() and path.is_dir():
        return path
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise argparse.ArgumentTypeError(
            "path must be a directory, .json, or .jsonl"
        )
    return path


def read_client_source(
    path: Path,
    offset: int,
    follow: bool,
) -> tuple[list[dict[str, Any]], int]:
    if follow and path.suffix.lower() == ".jsonl":
        return drain(path, offset)
    return list(load_client_intents(path)), offset


def read_trader_source(
    path: Path,
    offset: int,
    follow: bool,
) -> tuple[list[dict[str, Any]], int]:
    if follow and path.suffix.lower() == ".jsonl":
        return drain(path, offset)
    return list(load_trader_events(path)), offset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-intents",
        required=True,
        type=event_source_path,
        help=(
            "Per-client intent JSON directory, one client JSON, or accepted "
            "intent JSONL"
        ),
    )
    parser.add_argument(
        "--trader-events",
        required=True,
        type=event_source_path,
        help=(
            "Per-client trader-event JSON directory, one client JSON, or "
            "normalized trader event JSONL"
        ),
    )
    parser.add_argument(
        "--client-sessions-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "client_sessions_by_client",
        help="Per-client JSON directory produced by capture_trader_pipeline.py",
    )
    parser.add_argument(
        "--client-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "client_order_lifecycle",
        help="Directory for one current order lifecycle JSON file per client",
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--database",
        default=str(Path(__file__).resolve().parent / "orders.sqlite3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "order_lifecycle_snapshots.jsonl",
        help="Combined order lifecycle snapshots JSONL",
    )
    args = parser.parse_args()

    store = OrderStore(
        Path(args.database),
        args.output,
        args.client_sessions_dir,
        args.client_output_dir,
    )
    client_path = args.client_intents
    trader_path = args.trader_events
    client_offset = trader_offset = 0

    while True:
        store.reload_client_session_messages()
        clients, client_offset = read_client_source(
            client_path,
            client_offset,
            args.follow,
        )
        traders, trader_offset = read_trader_source(
            trader_path,
            trader_offset,
            args.follow,
        )
        combined = [("client", event) for event in clients]
        combined.extend(("trader", event) for event in traders)
        combined.sort(key=lambda item: timestamp_key(item[1]))
        for event_type, event in combined:
            if event_type == "client":
                store.add_client_intent(event)
            else:
                store.add_trader_event(event)
        if clients or traders or not args.follow:
            store.write_all_client_order_files()
        if not args.follow:
            break
        time.sleep(args.poll_interval)

    if (
        not args.follow
        and args.output
        and store.snapshots_written == 0
        and args.output.stat().st_size == 0
    ):
        store.write_all_order_snapshots("SNAPSHOT_EXPORT")

    if args.output:
        print(f"Order lifecycle snapshots: {args.output}")
        print(f"Snapshots written: {store.snapshots_written}")
    print(f"Per-client order lifecycle: {args.client_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
