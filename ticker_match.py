#!/usr/bin/env python3
"""
Source-backed ticker matching pipeline.

This uses Nasdaq Trader's official symbol directory files as the market universe:
  - nasdaqlisted.txt
  - otherlisted.txt

It validates exact tickers against that universe, then corrects likely misspellings
such as "cercl" -> "CRCL" using only symbols present in the downloaded market list.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

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


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def refresh_market_cache(cache_dir: Path) -> tuple[Path, Path]:
    nasdaq_path = cache_dir / "nasdaqlisted.txt"
    other_path = cache_dir / "otherlisted.txt"
    download_file(NASDAQ_LISTED_URL, nasdaq_path)
    download_file(OTHER_LISTED_URL, other_path)
    return nasdaq_path, other_path


def ensure_market_cache(cache_dir: Path, refresh: bool) -> tuple[Path, Path]:
    nasdaq_path = cache_dir / "nasdaqlisted.txt"
    other_path = cache_dir / "otherlisted.txt"
    if refresh or not nasdaq_path.exists() or not other_path.exists():
        return refresh_market_cache(cache_dir)
    return nasdaq_path, other_path


def parse_pipe_file(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = [line for line in lines if line and not line.startswith("File Creation Time")]
    reader = csv.DictReader(data_lines, delimiter="|")
    return [dict(row) for row in reader]


def load_market_symbols(cache_dir: Path, refresh: bool, include_etfs: bool) -> list[MarketSymbol]:
    nasdaq_path, other_path = ensure_market_cache(cache_dir, refresh)
    symbols: list[MarketSymbol] = []

    for row in parse_pipe_file(nasdaq_path):
        if row.get("Test Issue") == "Y":
            continue
        is_etf = row.get("ETF") == "Y"
        if is_etf and not include_etfs:
            continue
        symbol = (row.get("Symbol") or "").strip().upper()
        if symbol:
            symbols.append(
                MarketSymbol(
                    symbol=symbol,
                    security_name=(row.get("Security Name") or "").strip(),
                    exchange="NASDAQ",
                    source_file=nasdaq_path.name,
                    is_etf=is_etf,
                )
            )

    for row in parse_pipe_file(other_path):
        if row.get("Test Issue") == "Y":
            continue
        is_etf = row.get("ETF") == "Y"
        if is_etf and not include_etfs:
            continue
        symbol = (row.get("ACT Symbol") or "").strip().upper()
        if symbol:
            symbols.append(
                MarketSymbol(
                    symbol=symbol,
                    security_name=(row.get("Security Name") or "").strip(),
                    exchange=(row.get("Exchange") or "").strip(),
                    source_file=other_path.name,
                    is_etf=is_etf,
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
    symbols = load_market_symbols(Path(args.cache_dir), args.refresh_symbols, args.include_etfs)
    symbol_index = {normalize(symbol.symbol): symbol for symbol in symbols}
    symbols_for_fuzzy = [
        symbol
        for symbol in symbols
        if 1 < len(normalize(symbol.symbol)) <= 7 and normalize(symbol.symbol) not in STOPWORDS
    ]
    print(f"Loaded {len(symbols)} market symbols from Nasdaq Trader cache")
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
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument("--refresh-symbols", action="store_true", help="Download latest symbol files")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / "market_symbol_cache"),
        help="Folder for Nasdaq Trader symbol cache",
    )
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
