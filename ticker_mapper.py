#!/usr/bin/env python3
"""
Excel-source ticker matching pipeline.

It validates exact tickers against your symbol Excel/CSV, then corrects likely
misspellings such as "cercl" -> "CRCL" using only symbols present in that file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"\$?[A-Za-z][A-Za-z0-9.\-]{0,15}")
CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9.\-]{0,9})\b")

STOPWORDS = {
    "A",
    "ABOUT",
    "ALL",
    "ALSO",
    "AM",
    "AN",
    "AND",
    "ANY",
    "ARE",
    "AT",
    "BACK",
    "BETTER",
    "BID",
    "BUY",
    "CALL",
    "CAN",
    "CASH",
    "CLIENT",
    "CN",
    "CO",
    "COME",
    "DONE",
    "FOR",
    "FROM",
    "GET",
    "GIVE",
    "GOOD",
    "HI",
    "IF",
    "IN",
    "IS",
    "IT",
    "LET",
    "ME",
    "MSG",
    "NO",
    "OF",
    "OK",
    "ON",
    "OR",
    "ORDER",
    "PLEASE",
    "PUT",
    "SELL",
    "SEND",
    "SO",
    "THE",
    "TO",
    "TRADE",
    "USD",
    "US",
    "WE",
    "YES",
    "YOU",
}


@dataclass(frozen=True)
class MarketSymbol:
    symbol: str
    security_name: str
    exchange: str
    source_file: str
    is_etf: bool


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def excel_col_to_index(col: str) -> int:
    value = 0
    for char in col.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid Excel column: {col}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def load_symbols_from_excel(
    file_path: Path,
    sheet: str | int,
    symbol_col: str,
    skip_rows: int,
) -> list[MarketSymbol]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Excel symbol source needs pandas/openpyxl: pip install pandas openpyxl") from exc

    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path, header=None, skiprows=skip_rows)
    else:
        frame = pd.read_excel(file_path, sheet_name=sheet, header=None, skiprows=skip_rows)

    col_idx = excel_col_to_index(symbol_col)
    symbols: list[MarketSymbol] = []
    seen: set[str] = set()

    for _, row in frame.iterrows():
        if col_idx >= len(row) or pd.isna(row.iloc[col_idx]):
            continue

        raw_symbol = str(row.iloc[col_idx]).strip().upper()
        symbol_key = normalize(raw_symbol)
        if not symbol_key or symbol_key in seen:
            continue

        seen.add(symbol_key)
        symbols.append(
            MarketSymbol(
                symbol=raw_symbol,
                security_name="",
                exchange="USER_EXCEL",
                source_file=str(file_path),
                is_etf=False,
            )
        )

    return symbols


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def extract_tokens(message: str) -> list[tuple[str, str, bool]]:
    tokens: list[tuple[str, str, bool]] = []
    for raw in TOKEN_RE.findall(message):
        has_cash_tag = raw.startswith("$")
        token = normalize(raw.removeprefix("$"))
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append((raw, token, has_cash_tag))
    return tokens


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def add_match(
    matches: dict[str, dict[str, Any]],
    symbol: MarketSymbol,
    input_text: str,
    match_type: str,
    confidence: float,
) -> None:
    value = {
        "ticker": symbol.symbol,
        "security_name": symbol.security_name,
        "exchange": symbol.exchange,
        "is_etf": symbol.is_etf,
        "source_file": symbol.source_file,
        "input_text": input_text,
        "match_type": match_type,
        "confidence": round(confidence, 3),
    }
    existing = matches.get(symbol.symbol)
    if existing is None or value["confidence"] > existing["confidence"]:
        matches[symbol.symbol] = value


def find_market_matches(
    message: str,
    symbol_index: dict[str, MarketSymbol],
    symbols_for_fuzzy: list[MarketSymbol],
    fuzzy_threshold: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}

    for cashtag in CASHTAG_RE.findall(message):
        token = normalize(cashtag)
        exact = symbol_index.get(token)
        if exact:
            add_match(matches, exact, f"${cashtag}", "source_exact_cashtag", 1.0)

    for raw, token, has_cash_tag in extract_tokens(message):
        exact = symbol_index.get(token)
        if exact:
            add_match(
                matches,
                exact,
                raw,
                "source_exact_token" if not has_cash_tag else "source_exact_cashtag",
                1.0,
            )
            continue

        if len(token) > 7:
            continue

        candidates: list[tuple[float, MarketSymbol]] = []
        for symbol in symbols_for_fuzzy:
            normalized_symbol = normalize(symbol.symbol)
            if abs(len(normalized_symbol) - len(token)) > 2:
                continue
            score = similarity(token, normalized_symbol)
            if score >= fuzzy_threshold:
                candidates.append((score, symbol))

        candidates.sort(key=lambda item: (-item[0], item[1].symbol))
        for score, symbol in candidates[:max_candidates]:
            add_match(matches, symbol, raw, "source_fuzzy_symbol", score)

    return sorted(matches.values(), key=lambda item: (-item["confidence"], item["ticker"]))


def output_fields() -> list[str]:
    return [
        "captured_at",
        "source_timestamp",
        "sender_id",
        "sender_name",
        "message",
        "tickers",
        "security_names",
        "match_details",
        "source_row_num",
    ]


def enrich_event(
    event: dict[str, Any],
    symbol_index: dict[str, MarketSymbol],
    symbols_for_fuzzy: list[MarketSymbol],
    args: argparse.Namespace,
) -> dict[str, str]:
    message = str(event.get("message") or "")
    matches = find_market_matches(
        message,
        symbol_index,
        symbols_for_fuzzy,
        args.fuzzy_threshold,
        args.max_candidates,
    )
    return {
        "captured_at": str(event.get("captured_at") or ""),
        "source_timestamp": str(event.get("source_timestamp") or ""),
        "sender_id": str(event.get("sender_id") or event.get("participant_id") or ""),
        "sender_name": str(event.get("sender_name") or event.get("sender") or ""),
        "message": message,
        "tickers": "|".join(match["ticker"] for match in matches),
        "security_names": "|".join(match["security_name"] for match in matches),
        "match_details": json.dumps(matches, ensure_ascii=False),
        "source_row_num": str(event.get("source_row_num") or event.get("row_num") or ""),
    }


def prepare_symbols(args: argparse.Namespace) -> tuple[dict[str, MarketSymbol], list[MarketSymbol]]:
    symbol_sheet: str | int = args.symbol_sheet
    if isinstance(symbol_sheet, str) and symbol_sheet.isdigit():
        symbol_sheet = int(symbol_sheet)
    symbols = load_symbols_from_excel(
        Path(args.symbol_excel),
        symbol_sheet,
        args.symbol_col,
        args.symbol_skip_rows,
    )
    print(f"Loaded {len(symbols)} symbols from Excel source: {args.symbol_excel}")

    symbol_index = {normalize(symbol.symbol): symbol for symbol in symbols}
    symbols_for_fuzzy = [
        symbol
        for symbol in symbols
        if 1 < len(normalize(symbol.symbol)) <= 7 and normalize(symbol.symbol) not in STOPWORDS
    ]
    return symbol_index, symbols_for_fuzzy


def run_batch(args: argparse.Namespace) -> None:
    symbol_index, symbols_for_fuzzy = prepare_symbols(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "market_ticker_enriched_messages.csv"
    jsonl_path = output_dir / "market_ticker_enriched_messages.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=output_fields())
        writer.writeheader()
        for event in read_events(Path(args.input)):
            row = enrich_event(event, symbol_index, symbols_for_fuzzy, args)
            if not row["tickers"] and not args.keep_unmatched:
                continue
            writer.writerow(row)
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Market ticker enriched CSV:   {csv_path}")
    print(f"Market ticker enriched JSONL: {jsonl_path}")


def run_follow(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if input_path.suffix.lower() != ".jsonl":
        raise SystemExit("--follow expects JSONL input so it can tail new captured rows")

    symbol_index, symbols_for_fuzzy = prepare_symbols(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "market_ticker_enriched_messages.csv"
    jsonl_path = output_dir / "market_ticker_enriched_messages.jsonl"
    write_header = not csv_path.exists()

    print(f"Following input JSONL: {input_path}")
    print(f"Market ticker enriched CSV:   {csv_path}")
    print(f"Market ticker enriched JSONL: {jsonl_path}")

    while not input_path.exists():
        print(f"Waiting for input file: {input_path}", flush=True)
        time.sleep(args.poll_interval)

    with input_path.open("r", encoding="utf-8") as input_handle, csv_path.open(
        "a", encoding="utf-8", newline=""
    ) as csv_handle, jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=output_fields())
        if write_header:
            writer.writeheader()

        while True:
            line = input_handle.readline()
            if not line:
                time.sleep(args.poll_interval)
                continue

            row = enrich_event(json.loads(line), symbol_index, symbols_for_fuzzy, args)
            if not row["tickers"] and not args.keep_unmatched:
                continue
            writer.writerow(row)
            csv_handle.flush()
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            print(
                f'{row["source_timestamp"]} sender_id={row["sender_id"]} '
                f'tickers={row["tickers"]} message={row["message"]}',
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Captured CSV/JSONL file")
    parser.add_argument("--follow", action="store_true", help="Continuously watch input JSONL")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.84)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--keep-unmatched", action="store_true")
    parser.add_argument(
        "--symbol-excel",
        required=True,
        help="Excel/CSV file containing valid symbols in the first column",
    )
    parser.add_argument("--symbol-sheet", default=0, help="Sheet name or zero-based index for --symbol-excel")
    parser.add_argument("--symbol-col", default="A", help="Column containing ticker symbols in --symbol-excel")
    parser.add_argument("--symbol-skip-rows", type=int, default=0, help="Header rows to skip in --symbol-excel")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Folder where enriched files are saved",
    )
    args = parser.parse_args()

    if args.follow:
        run_follow(args)
    else:
        run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
