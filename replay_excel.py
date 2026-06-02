#!/usr/bin/env python3
"""
Replay Bloomberg-style chat rows from Excel as realtime JSON events.

Default column mapping matches the screenshots:
  A = date, B = timestamp, C = sender, D = chat_id, E = participant_id, F = message
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("Install dependencies first: pip install pandas openpyxl requests") from exc


def excel_col_to_index(col: str) -> int:
    value = 0
    for char in col.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid Excel column: {col}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def cell(row: Any, col: str) -> Any:
    idx = excel_col_to_index(col)
    if idx >= len(row):
        return None
    value = row.iloc[idx]
    return None if pd.isna(value) else value


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def build_event(row_num: int, row: Any, args: argparse.Namespace) -> dict[str, Any]:
    date_value = clean_text(cell(row, args.date_col))
    ts_value = clean_text(cell(row, args.timestamp_col))
    sender = clean_text(cell(row, args.sender_col))
    message = clean_text(cell(row, args.message_col))

    return {
        "event_id": f"{args.source_name}-{row_num}",
        "source": args.source_name,
        "row_num": row_num,
        "source_date": date_value,
        "source_timestamp": ts_value,
        "sender": sender,
        "chat_id": clean_text(cell(row, args.chat_id_col)),
        "participant_id": clean_text(cell(row, args.participant_id_col)),
        "message": message,
        "simulated_at": datetime.now(timezone.utc).isoformat(),
    }


def emit_stdout(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def emit_http(event: dict[str, Any], url: str) -> None:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("HTTP sink needs requests: pip install requests") from exc

    response = requests.post(url, json=event, timeout=10)
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .xlsx/.xls/.csv file")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or zero-based index")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between messages")
    parser.add_argument("--skip-rows", type=int, default=0, help="Rows to skip at top")
    parser.add_argument("--start-row", type=int, default=1, help="1-based input row to start replaying")
    parser.add_argument("--limit", type=int, default=0, help="Maximum events to emit; 0 means all")
    parser.add_argument("--source-name", default="bloomberg-chat-excel")
    parser.add_argument("--date-col", default="A")
    parser.add_argument("--timestamp-col", default="B")
    parser.add_argument("--sender-col", default="C")
    parser.add_argument("--chat-id-col", default="D")
    parser.add_argument("--participant-id-col", default="E")
    parser.add_argument("--message-col", default="F")
    parser.add_argument("--http-url", help="Optional pipeline endpoint to POST every JSON event")
    args = parser.parse_args()

    sheet: str | int = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
