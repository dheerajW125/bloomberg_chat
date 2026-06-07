#!/usr/bin/env python3
"""Join client order intents and trader events into persistent order lifecycle state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPEN_STATUSES = ("PENDING_ACK", "ACKNOWLEDGED", "WORKING", "PARTIALLY_FILLED", "FILL_REPORTED")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class OrderStore:
    def __init__(self, path: Path, jsonl_path: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.jsonl_path = jsonl_path
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
        tickers = event.get("tickers") or []

        if etype == "ACK":
            row = self.db.execute(
                "SELECT * FROM batches WHERE room_id=? AND status='PENDING_ACK' ORDER BY created_at DESC LIMIT 1",
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

        candidates = self._find_orders(room, trader, tickers, event.get("side", ""))
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

    def _find_orders(self, room: str, trader: str, tickers: list[str], side: str) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in OPEN_STATUSES)
        query = f"""
            SELECT o.*,b.assigned_trader_id,b.created_at AS batch_created
            FROM orders o JOIN batches b ON b.batch_id=o.batch_id
            WHERE b.room_id=? AND o.status IN ({placeholders})
        """
        params: list[Any] = [room, *OPEN_STATUSES]
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
        trader_messages: list[dict[str, Any]] = []
        lifecycle_events: list[dict[str, Any]] = []

        for event in events:
            raw = json.loads(event["raw_json"])
            message_record = {
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
                client_messages.append(message_record)
            else:
                trader_messages.append(message_record)

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
            "trader_messages": trader_messages,
            "lifecycle_events": lifecycle_events,
        }

    def _write_order_snapshot(self, order_id: str, trigger_event_type: str) -> None:
        if not self.jsonl_path:
            return
        snapshot = self._order_snapshot(order_id, trigger_event_type)
        if not snapshot:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

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


def timestamp_key(event: dict[str, Any]) -> str:
    return str(event.get("source_timestamp") or event.get("captured_at") or "")


def jsonl_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".jsonl":
        raise argparse.ArgumentTypeError("path must be a .jsonl file")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-intents",
        required=True,
        type=jsonl_path,
        help="Accepted client order intents JSONL",
    )
    parser.add_argument(
        "--trader-events",
        required=True,
        type=jsonl_path,
        help="Normalized trader order events JSONL",
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--database",
        default=str(Path(__file__).resolve().parent / "orders.sqlite3"),
    )
    parser.add_argument(
        "--output",
        type=jsonl_path,
        default=Path(__file__).resolve().parent / "order_lifecycle_snapshots.jsonl",
        help="Order lifecycle snapshots JSONL",
    )
    args = parser.parse_args()

    store = OrderStore(Path(args.database), args.output)
    client_path = args.client_intents
    trader_path = args.trader_events
    client_offset = trader_offset = 0

    while True:
        clients, client_offset = drain(client_path, client_offset)
        traders, trader_offset = drain(trader_path, trader_offset)
        combined = [("client", event) for event in clients]
        combined.extend(("trader", event) for event in traders)
        combined.sort(key=lambda item: timestamp_key(item[1]))
        for event_type, event in combined:
            if event_type == "client":
                store.add_client_intent(event)
            else:
                store.add_trader_event(event)
        if not args.follow:
            break
        time.sleep(args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
