#!/usr/bin/env python3
"""Convert captured trader messages into normalized order lifecycle events."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


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


def extract_tickers(message: str, symbols: set[str]) -> list[str]:
    found: list[str] = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9.\-]{1,9}\b", message):
        ticker = token.upper()
        if ticker in symbols and ticker not in found:
            found.append(ticker)
    return found


def load_symbols(path: Path) -> set[str]:
    if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        raise SystemExit("--symbol-excel must be an Excel file (.xlsx, .xls, or .xlsm)")
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas/openpyxl: pip install pandas openpyxl") from exc
    frame = pd.read_excel(path, header=None)
    return {str(value).strip().upper() for value in frame.iloc[:, 0].dropna() if str(value).strip()}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="trader_messages.jsonl")
    parser.add_argument("--symbol-excel", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "trader_order_events.jsonl"),
    )
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbol_excel))
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    while not input_path.exists() and args.follow:
        time.sleep(args.poll_interval)

    with input_path.open("r", encoding="utf-8") as source, output_path.open("a", encoding="utf-8") as target:
        while True:
            line = source.readline()
            if not line:
                if args.follow:
                    time.sleep(args.poll_interval)
                    continue
                break
            normalized = parse_event(json.loads(line), symbols)
            if normalized:
                target.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                target.flush()
                print(f'{normalized["event_type"]} trader={normalized["trader_id"]} room={normalized["room_id"]}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
