#!/usr/bin/env python3
"""Convert captured trader messages into normalized order lifecycle events."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ACK_RE = re.compile(r"^\s*(?:ok|okay|see|working|on it|will do|got it|yes)\b", re.I)
REJECT_RE = re.compile(r"\b(?:cannot|can't|unable|no liquidity|reject|decline)\b", re.I)
PARTIAL_RE = re.compile(r"\b(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?:done|filled)\b", re.I)
FILL_WORD_RE = re.compile(r"\b(?:bought|sold|filled|done)\b", re.I)
SIDE_RE = re.compile(r"\b(?P<side>B|S|BUY|SELL|BOUGHT|SOLD)\b", re.I)
URL_RE = re.compile(r"(?:https?|ftp|bloomberg)://|www\.|mailto:|<GO>|{GO}", re.I)
NEWS_RE = re.compile(
    r"\b(?:news|research|report|reported|headline|stake|takeover|m[&a]|"
    r"acquisition|merger|analyst|rating|target|upgrade|downgrade|outperform|"
    r"underperform|neutral|initiated|raised|lowered|pt|price target|reuters|"
    r"bloomberg|cnbc|earnings|guidance|forecast|outlook|shares|stock|"
    r"would|could|may|said|says|plans|expects|seeks|mulls|weighs)\b",
    re.I,
)
QTY_TICKER_RE = re.compile(
    r"\b(?P<qty>\d+(?:[.,]\d+)?)\s*(?P<unit>k|m|mm|mn)?\s+"
    r"(?P<ticker>[A-Za-z][A-Za-z0-9.\-]{0,9})\b",
    re.I,
)
PRICE_RE = re.compile(r"(?:@|\bat\b|\bfor\b)\s*(?P<price>\d+(?:\.\d+)?)", re.I)
COMPACT_EXECUTION_RE = re.compile(
    r"\b(?:B|S|BUY|SELL|BOUGHT|SOLD)\s*\d+(?:[.,]\d+)?\s*(?:K|M|MM|MN)?\s+"
    r"[A-Z][A-Z0-9.\-]{0,9}(?:\s*(?:@|FOR|AT)\s*\d+(?:\.\d+)?)?",
    re.I,
)


def clean(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", clean(value))
    return cleaned[:180] or "unknown-client"


def extract_tickers(message: str, symbols: set[str]) -> list[str]:
    found: list[str] = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9.\-]{1,9}\b", message):
        ticker = token.upper()
        if ticker in symbols and ticker not in found:
            found.append(ticker)
    return found


def load_symbols(path: Path, symbol_column: str = "Symbol") -> set[str]:
    suffix = path.suffix.lower()
    values: list[Any]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise SystemExit(f"No rows found in symbol file: {path}")
        headers = [clean(value).lower() for value in rows[0]]
        requested = clean(symbol_column).lower()
        if requested in headers:
            column_index = headers.index(requested)
            data_rows = rows[1:]
        elif "symbol" in headers:
            column_index = headers.index("symbol")
            data_rows = rows[1:]
        elif "ticker" in headers:
            column_index = headers.index("ticker")
            data_rows = rows[1:]
        else:
            column_index = 0
            data_rows = rows
        values = [
            row[column_index]
            for row in data_rows
            if column_index < len(row) and clean(row[column_index])
        ]
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("Install pandas/openpyxl: pip install pandas openpyxl") from exc
        frame = pd.read_excel(path, header=None)
        if frame.empty:
            raise SystemExit(f"No rows found in symbol file: {path}")
        headers = [clean(value).lower() for value in frame.iloc[0].tolist()]
        requested = clean(symbol_column).lower()
        if requested in headers:
            column_index = headers.index(requested)
            start_row = 1
        elif "symbol" in headers:
            column_index = headers.index("symbol")
            start_row = 1
        elif "ticker" in headers:
            column_index = headers.index("ticker")
            start_row = 1
        else:
            column_index = 0
            start_row = 0
        values = frame.iloc[start_row:, column_index].dropna().tolist()
    else:
        raise SystemExit("--symbol-file must be .csv, .xlsx, .xls, or .xlsm")

    symbols = {clean(value).upper() for value in values if clean(value)}
    if not symbols:
        raise SystemExit(f"No symbols found in {path}")
    return symbols


def parse_timestamp(value: Any) -> datetime | None:
    text = clean(value)
    if not text:
        return None
    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
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


def order_tickers(intent: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    raw_orders = intent.get("orders_json") or []
    if isinstance(raw_orders, str):
        try:
            raw_orders = json.loads(raw_orders)
        except json.JSONDecodeError:
            raw_orders = []
    for order in raw_orders if isinstance(raw_orders, list) else []:
        ticker = clean(order.get("ticker")).upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        for ticker in clean(intent.get("candidate_tickers")).split("|"):
            ticker = ticker.strip().upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers


def accepted_client_intents(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for client_path in sorted(path.glob("*.json")):
            document = json.loads(client_path.read_text(encoding="utf-8-sig"))
            client_id = clean(document.get("client_id"))
            client_name = clean(document.get("client_name"))
            for intent in document.get("accepted_intents") or []:
                row = dict(intent)
                row.setdefault("sender_id", client_id)
                row.setdefault("sender_name", client_name)
                yield row
        return
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        client_id = clean(document.get("client_id"))
        client_name = clean(document.get("client_name"))
        for intent in document.get("accepted_intents") or []:
            row = dict(intent)
            row.setdefault("sender_id", client_id)
            row.setdefault("sender_name", client_name)
            yield row
        return
    if path.suffix.lower() != ".jsonl":
        raise SystemExit("--client-intents must be a directory, .json, or .jsonl")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("processing_status") in {"", None, "accepted"}:
                    yield row


class ClientIntentMatcher:
    def __init__(self, path: Path, max_minutes: float) -> None:
        self.path = path
        self.max_seconds = max_minutes * 60
        self.intents: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        intents: list[dict[str, Any]] = []
        for intent in accepted_client_intents(self.path):
            timestamp = parse_timestamp(
                intent.get("source_timestamp") or intent.get("captured_at")
            )
            if timestamp is None:
                continue
            row = dict(intent)
            row["_parsed_timestamp"] = timestamp
            row["_tickers"] = order_tickers(intent)
            intents.append(row)
        self.intents = intents

    def match(
        self,
        trader_event: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, float | None]:
        trader_timestamp = parse_timestamp(
            trader_event.get("source_timestamp") or trader_event.get("captured_at")
        )
        if trader_timestamp is None:
            return None, "invalid_trader_source_timestamp", None

        room_id = clean(trader_event.get("room_id"))
        trader_tickers = {
            clean(ticker).upper() for ticker in trader_event.get("tickers") or []
        }
        trader_side = clean(trader_event.get("side")).upper()
        candidates: list[tuple[tuple[int, int, float], dict[str, Any], float]] = []

        for intent in self.intents:
            if clean(intent.get("room_id")) != room_id:
                continue
            delta = (trader_timestamp - intent["_parsed_timestamp"]).total_seconds()
            if delta < 0 or delta > self.max_seconds:
                continue
            intent_tickers = set(intent["_tickers"])
            ticker_mismatch = bool(
                trader_tickers and not trader_tickers.intersection(intent_tickers)
            )
            if ticker_mismatch:
                continue
            intent_side = clean(intent.get("trade_side")).upper()
            side_mismatch = bool(
                trader_side and intent_side and trader_side != intent_side
            )
            if side_mismatch:
                continue
            ticker_penalty = 0 if trader_tickers else 1
            side_penalty = 0 if trader_side and intent_side else 1
            candidates.append(
                ((ticker_penalty, side_penalty, delta), intent, delta)
            )

        if not candidates:
            return None, "no_preceding_compatible_client_intent", None
        _, matched, delta = min(candidates, key=lambda item: item[0])
        clean_match = {
            key: value
            for key, value in matched.items()
            if not key.startswith("_")
        }
        basis = "room+source_timestamp"
        if trader_tickers:
            basis += "+ticker"
        if trader_side:
            basis += "+side"
        return clean_match, basis, delta


def is_automated_or_news_message(message: str) -> bool:
    if URL_RE.search(message):
        return True
    words = message.split()
    if NEWS_RE.search(message) and (len(words) >= 10 or not FILL_WORD_RE.search(message)):
        return True
    letters = [char for char in message if char.isalpha()]
    uppercase_ratio = (
        sum(char.isupper() for char in letters) / len(letters)
        if letters
        else 0.0
    )
    return len(message) >= 40 and uppercase_ratio >= 0.72


def has_execution_evidence(message: str, tickers: list[str]) -> bool:
    if not tickers:
        return False
    if COMPACT_EXECUTION_RE.search(message):
        return True
    if PRICE_RE.search(message) and SIDE_RE.search(message):
        return True
    if FILL_WORD_RE.search(message):
        words = message.split()
        return len(words) <= 8 or bool(QTY_TICKER_RE.search(message) or PRICE_RE.search(message))
    return False


def parse_event(event: dict[str, Any], symbols: set[str]) -> dict[str, Any] | None:
    message = clean(event.get("message"))
    room = clean(event.get("room_id") or event.get("chat_id") or event.get("room_name"))
    trader = clean(event.get("actor_id") or event.get("sender_id"))
    base = {
        "source_event_id": clean(event.get("event_id")),
        "source_timestamp": clean(event.get("source_timestamp")),
        "room_id": room,
        "trader_id": trader,
        "raw_message": message,
    }

    if (
        clean(event.get("message_classification")) == "automated"
        or clean(event.get("actor_classification")) == "automated"
        or is_automated_or_news_message(message)
    ):
        return None

    if REJECT_RE.search(message):
        return {**base, "event_type": "REJECT", "tickers": extract_tickers(message, symbols)}

    partial = PARTIAL_RE.search(message)
    if partial:
        return {
            **base,
            "event_type": "PARTIAL_FILL",
            "tickers": extract_tickers(message, symbols),
            "fill_percent": partial.group("pct"),
        }

    tickers = extract_tickers(message, symbols)
    if has_execution_evidence(message, tickers):
        quantity_match = QTY_TICKER_RE.search(message)
        price_match = PRICE_RE.search(message)
        side_match = SIDE_RE.search(message)
        side_token = side_match.group("side").upper() if side_match else ""
        side = "BUY" if side_token in {"B", "BUY", "BOUGHT"} else "SELL" if side_token else ""
        return {
            **base,
            "event_type": "FILL",
            "side": side,
            "tickers": tickers,
            "filled_quantity": quantity_match.group("qty").replace(",", ".") if quantity_match else "",
            "quantity_unit": (quantity_match.group("unit") or "").lower() if quantity_match else "",
            "price": price_match.group("price") if price_match else "",
            "full_fill_claim": bool(re.search(r"\b(?:bought|sold|filled|done)\b", message, re.I)),
        }

    if ACK_RE.search(message):
        return {**base, "event_type": "ACK", "tickers": tickers}
    return None


def mapped_event(
    trader_event: dict[str, Any],
    client_intent: dict[str, Any],
    match_basis: str,
    delta_seconds: float,
) -> dict[str, Any]:
    return {
        **trader_event,
        "client_id": clean(client_intent.get("sender_id")),
        "client_name": clean(client_intent.get("sender_name")),
        "client_session_id": clean(client_intent.get("client_session_id")),
        "matched_client_source_event_id": clean(
            client_intent.get("source_event_id")
        ),
        "matched_client_source_timestamp": clean(
            client_intent.get("source_timestamp")
            or client_intent.get("captured_at")
        ),
        "match_delta_seconds": round(delta_seconds, 3),
        "match_basis": match_basis,
    }


def upsert_client_event_file(
    event: dict[str, Any],
    client_intent: dict[str, Any],
    output_dir: Path,
) -> None:
    client_id = clean(event.get("client_id"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_filename(client_id)}.json"
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        document = {
            "record_type": "client_trader_events",
            "client_id": client_id,
            "client_name": clean(event.get("client_name")),
            "generated_at": "",
            "trader_event_count": 0,
            "mappings": [],
        }

    event_key = (
        clean(event.get("source_event_id")),
        clean(event.get("trader_id")),
        clean(event.get("source_timestamp")),
    )
    if any(
        (
            clean(item.get("trader_event", {}).get("source_event_id")),
            clean(item.get("trader_event", {}).get("trader_id")),
            clean(item.get("trader_event", {}).get("source_timestamp")),
        )
        == event_key
        for item in document.get("mappings") or []
    ):
        return

    document.setdefault("mappings", []).append(
        {
            "match_basis": event["match_basis"],
            "match_delta_seconds": event["match_delta_seconds"],
            "client_intent": client_intent,
            "trader_event": event,
        }
    )
    document["mappings"].sort(
        key=lambda item: clean(
            item.get("trader_event", {}).get("source_timestamp")
        )
    )
    document["generated_at"] = datetime.now(timezone.utc).isoformat()
    document["trader_event_count"] = len(document["mappings"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="trader_messages.jsonl")
    parser.add_argument(
        "--symbol-file",
        "--symbol-excel",
        dest="symbol_file",
        required=True,
        help="Symbol master in CSV or Excel format",
    )
    parser.add_argument(
        "--symbol-column",
        default="Symbol",
        help="Symbol column header; defaults to Symbol",
    )
    parser.add_argument(
        "--client-intents",
        type=Path,
        help=(
            "Per-client intent JSON directory, one client JSON, or accepted "
            "intent JSONL. Enables timestamp-based client mapping."
        ),
    )
    parser.add_argument(
        "--client-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "client_trader_events",
        help="Directory for one mapped trader-event JSON file per client",
    )
    parser.add_argument(
        "--mapping-review",
        type=Path,
        default=Path(__file__).resolve().parent
        / "trader_event_mapping_review.jsonl",
        help="JSONL for parsed trader events that could not be mapped",
    )
    parser.add_argument(
        "--max-match-minutes",
        type=float,
        default=120.0,
        help="Maximum time after a client intent that a trader event can match",
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "trader_order_events.jsonl"),
    )
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbol_file), args.symbol_column)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matcher = (
        ClientIntentMatcher(args.client_intents, args.max_match_minutes)
        if args.client_intents
        else None
    )
    if matcher:
        args.mapping_review.parent.mkdir(parents=True, exist_ok=True)

    while not input_path.exists() and args.follow:
        time.sleep(args.poll_interval)

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "a", encoding="utf-8"
    ) as target:
        while True:
            line = source.readline()
            if not line:
                if args.follow:
                    if matcher:
                        matcher.reload()
                    time.sleep(args.poll_interval)
                    continue
                break
            normalized = parse_event(json.loads(line), symbols)
            if normalized:
                client_id = ""
                if matcher:
                    client_intent, basis, delta = matcher.match(normalized)
                    if client_intent is None or delta is None:
                        normalized["mapping_status"] = "review"
                        normalized["mapping_review_reason"] = basis
                        with args.mapping_review.open(
                            "a", encoding="utf-8"
                        ) as review:
                            review.write(
                                json.dumps(
                                    {
                                        "review_reason": basis,
                                        "trader_event": normalized,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        print(
                            f'UNMATCHED {normalized["event_type"]} '
                            f'trader={normalized["trader_id"]} '
                            f'room={normalized["room_id"]}',
                            flush=True,
                        )
                    else:
                        normalized = mapped_event(
                            normalized,
                            client_intent,
                            basis,
                            delta,
                        )
                        normalized["mapping_status"] = "matched"
                        client_id = normalized["client_id"]
                        upsert_client_event_file(
                            normalized,
                            client_intent,
                            args.client_output_dir,
                        )
                target.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                target.flush()
                print(
                    f'{normalized["event_type"]} trader={normalized["trader_id"]} '
                    f'client={client_id or "unmapped"} '
                    f'room={normalized["room_id"]}',
                    flush=True,
                )
    print(f"Trader event JSONL: {output_path}")
    if matcher:
        print(f"Per-client trader events: {args.client_output_dir}")
        print(f"Mapping review JSONL: {args.mapping_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
