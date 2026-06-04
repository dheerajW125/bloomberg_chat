#!/usr/bin/env python3
"""
NLP trade intent layer for group chat messages.

Purpose:
  - Filter out casual chat such as "hi", "hello", "hey mate".
  - Use NLTK tokenization/stemming to detect English and Portuguese trade intent.
  - Carry forward side/quantity context per sender.
  - Output only trade-like messages for the ticker matching pipeline.

Example:
  Sender says: "buy 10pc NVDA"
  Later same sender says: "CRCL"
  Output becomes: "BUY 10pc CRCL"
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


QTY_RE = re.compile(
    r"\b(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>pc|pcs|pct|%|k|m|mm|mn|usd|eur|gbp|shares|shs|lots?)\b",
    re.IGNORECASE,
)

BUY_TERMS = {
    "buy",
    "bought",
    "bid",
    "lift",
    "lifted",
    "compra",
    "comprar",
    "compro",
    "comprei",
    "comprando",
}

SELL_TERMS = {
    "sell",
    "sold",
    "offer",
    "offered",
    "short",
    "vende",
    "vender",
    "venda",
    "vendo",
    "vendido",
}

NOISE_EXACT = {
    "HI",
    "HELLO",
    "HEY",
    "HEY MATE",
    "HI MATE",
    "GOOD MORNING",
    "GOOD AFTERNOON",
    "GOOD EVENING",
    "OK",
    "OKAY",
    "THANKS",
    "THANK YOU",
    "OBRIGADO",
    "OBRIGADA",
    "BOM DIA",
    "BOA TARDE",
    "BOA NOITE",
}

STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AT",
    "BOA",
    "BOM",
    "CAN",
    "CLIENT",
    "DIA",
    "DE",
    "DO",
    "DOS",
    "FOR",
    "FROM",
    "GOOD",
    "HELLO",
    "HEY",
    "HI",
    "IF",
    "IN",
    "IS",
    "IT",
    "MATE",
    "ME",
    "NO",
    "NOITE",
    "OF",
    "OK",
    "ON",
    "OR",
    "PLEASE",
    "TARDE",
    "THANKS",
    "THE",
    "TO",
    "USD",
    "US",
    "YOU",
}


@dataclass
class SenderContext:
    side: str = ""
    quantity: str = ""
    quantity_unit: str = ""


class NlpEngine:
    def __init__(self) -> None:
        try:
            from nltk.stem import SnowballStemmer
            from nltk.tokenize import RegexpTokenizer
        except ImportError as exc:
            raise SystemExit("Install NLTK first: pip install nltk") from exc

        self.tokenizer = RegexpTokenizer(r"\$?[^\W\d_][\w.\-]{0,20}", flags=re.UNICODE)
        self.english_stemmer = SnowballStemmer("english")
        self.portuguese_stemmer = SnowballStemmer("portuguese")
        self.buy_keys = self._term_keys(BUY_TERMS)
        self.sell_keys = self._term_keys(SELL_TERMS)

    def tokens(self, message: str) -> list[str]:
        return self.tokenizer.tokenize(message)

    def normalized_tokens(self, message: str) -> list[str]:
        values: list[str] = []
        for token in self.tokens(message):
            normalized = normalize(token.removeprefix("$"))
            if normalized:
                values.append(normalized)
        return values

    def lexical_keys(self, token: str) -> set[str]:
        raw = token.lower().strip("$")
        raw = re.sub(r"[^a-z0-9.\-]+", "", raw)
        if not raw:
            return set()
        return {
            raw,
            self.english_stemmer.stem(raw),
            self.portuguese_stemmer.stem(raw),
        }

    def _term_keys(self, terms: set[str]) -> set[str]:
        keys: set[str] = set()
        for term in terms:
            keys.update(self.lexical_keys(term))
        return keys

    def detect_side(self, message: str) -> str:
        for token in self.tokens(message):
            keys = self.lexical_keys(token)
            if keys & self.buy_keys:
                return "BUY"
            if keys & self.sell_keys:
                return "SELL"
        return ""


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def clean_message(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def excel_col_to_index(col: str) -> int:
    value = 0
    for char in col.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid Excel column: {col}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def load_symbol_set(file_path: Path, sheet: str | int, symbol_col: str, skip_rows: int) -> set[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Symbol Excel source needs pandas/openpyxl: pip install pandas openpyxl") from exc

    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path, header=None, skiprows=skip_rows)
    else:
        frame = pd.read_excel(file_path, sheet_name=sheet, header=None, skiprows=skip_rows)

    col_idx = excel_col_to_index(symbol_col)
    symbols: set[str] = set()
    for _, row in frame.iterrows():
        if col_idx >= len(row) or pd.isna(row.iloc[col_idx]):
            continue
        symbol = normalize(str(row.iloc[col_idx]).strip())
        if symbol:
            symbols.add(symbol)
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


def detect_quantity(message: str) -> tuple[str, str]:
    match = QTY_RE.search(message)
    if not match:
        return "", ""
    quantity = match.group("num").replace(",", ".")
    unit = match.group("unit").lower()
    return quantity, unit


def fuzzy_symbol(token: str, symbol_set: set[str], threshold: float) -> str:
    best_symbol = ""
    best_score = 0.0
    for symbol in symbol_set:
        if abs(len(symbol) - len(token)) > 2:
            continue
        score = SequenceMatcher(None, token, symbol).ratio()
        if score > best_score:
            best_score = score
            best_symbol = symbol
    return best_symbol if best_score >= threshold else ""


def extract_symbols(message: str, symbol_set: set[str], fuzzy_threshold: float, nlp: NlpEngine) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in nlp.tokens(message):
        token = normalize(raw.removeprefix("$"))
        if len(token) < 2 or token in STOPWORDS:
            continue
        symbol = token if token in symbol_set else fuzzy_symbol(token, symbol_set, fuzzy_threshold)
        if symbol and symbol not in seen:
            found.append(symbol)
            seen.add(symbol)
    return found


def is_noise(message: str, nlp: NlpEngine) -> bool:
    normalized_words = " ".join(nlp.normalized_tokens(message))
    if normalized_words in NOISE_EXACT:
        return True
    words = normalized_words.split()
    return len(words) <= 3 and all(word in STOPWORDS for word in words)


def sender_key(event: dict[str, Any]) -> str:
    return str(
        event.get("sender_id")
        or event.get("participant_id")
        or event.get("sender_name")
        or event.get("sender")
        or "UNKNOWN"
    )


def build_output_row(
    event: dict[str, Any],
    original_message: str,
    side: str,
    quantity: str,
    quantity_unit: str,
    tickers: list[str],
    context_applied: bool,
) -> dict[str, str]:
    quantity_text = f"{quantity}{quantity_unit}" if quantity and quantity_unit else quantity
    normalized_instruction = " ".join(part for part in [side, quantity_text, " ".join(tickers)] if part)
    return {
        "captured_at": str(event.get("captured_at") or ""),
        "source_timestamp": str(event.get("source_timestamp") or ""),
        "sender_id": str(event.get("sender_id") or event.get("participant_id") or ""),
        "sender_name": str(event.get("sender_name") or event.get("sender") or ""),
        "message": normalized_instruction,
        "original_message": original_message,
        "trade_side": side,
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "candidate_tickers": "|".join(tickers),
        "context_applied": str(context_applied),
        "source_row_num": str(event.get("source_row_num") or event.get("row_num") or ""),
    }


def output_fields() -> list[str]:
    return [
        "captured_at",
        "source_timestamp",
        "sender_id",
        "sender_name",
        "message",
        "original_message",
        "trade_side",
        "quantity",
        "quantity_unit",
        "candidate_tickers",
        "context_applied",
        "source_row_num",
    ]


def process_one_event(
    event: dict[str, Any],
    symbol_set: set[str],
    contexts: dict[str, SenderContext],
    keep_unmatched_intent: bool,
    fuzzy_threshold: float,
    nlp: NlpEngine,
) -> dict[str, str] | None:
    message = clean_message(event.get("message"))
    if not message or is_noise(message, nlp):
        return None

    key = sender_key(event)
    context = contexts.setdefault(key, SenderContext())

    side = nlp.detect_side(message)
    quantity, quantity_unit = detect_quantity(message)
    tickers = extract_symbols(message, symbol_set, fuzzy_threshold, nlp)

    has_explicit_intent = bool(side or quantity)
    can_use_context = bool(context.side and context.quantity)

    if has_explicit_intent and tickers:
        final_side = side or context.side
        final_quantity = quantity or context.quantity
        final_unit = quantity_unit or context.quantity_unit
        context.side = final_side
        context.quantity = final_quantity
        context.quantity_unit = final_unit
        return build_output_row(event, message, final_side, final_quantity, final_unit, tickers, False)

    if tickers and can_use_context:
        return build_output_row(
            event,
            message,
            context.side,
            context.quantity,
            context.quantity_unit,
            tickers,
            True,
        )

    if has_explicit_intent and not tickers and keep_unmatched_intent:
        if side:
            context.side = side
        if quantity:
            context.quantity = quantity
            context.quantity_unit = quantity_unit
        return build_output_row(event, message, side, quantity, quantity_unit, [], False)

    return None


def process_events(events: Iterable[dict[str, Any]], symbol_set: set[str], args: argparse.Namespace) -> list[dict[str, str]]:
    contexts: dict[str, SenderContext] = {}
    output_rows: list[dict[str, str]] = []
    nlp = NlpEngine()

    for event in events:
        row = process_one_event(
            event,
            symbol_set,
            contexts,
            args.keep_unmatched_intent,
            args.fuzzy_threshold,
            nlp,
        )
        if row:
            output_rows.append(row)

    return output_rows


def write_rows(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nlp_trade_intent_messages.csv"
    jsonl_path = output_dir / "nlp_trade_intent_messages.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=output_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"NLP trade intent CSV:   {csv_path}")
    print(f"NLP trade intent JSONL: {jsonl_path}")


def follow_jsonl(input_path: Path, symbol_set: set[str], args: argparse.Namespace) -> None:
    if input_path.suffix.lower() != ".jsonl":
        raise SystemExit("--follow expects JSONL input")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nlp_trade_intent_messages.csv"
    jsonl_path = output_dir / "nlp_trade_intent_messages.jsonl"
    write_header = not csv_path.exists()
    contexts: dict[str, SenderContext] = {}
    nlp = NlpEngine()

    print(f"Following chat JSONL: {input_path}")
    print(f"NLP trade intent CSV:   {csv_path}")
    print(f"NLP trade intent JSONL: {jsonl_path}")

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

            row = process_one_event(
                json.loads(line),
                symbol_set,
                contexts,
                args.keep_unmatched_intent,
                args.fuzzy_threshold,
                nlp,
            )
            if not row:
                continue

            writer.writerow(row)
            csv_handle.flush()
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            print(
                f'{row["source_timestamp"]} sender_id={row["sender_id"]} '
                f'intent={row["message"]} original={row["original_message"]}',
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw/captured chat CSV or JSONL")
    parser.add_argument("--symbol-excel", required=True, help="Excel/CSV with valid symbols")
    parser.add_argument("--symbol-sheet", default=0, help="Sheet name or zero-based index")
    parser.add_argument("--symbol-col", default="A", help="Column containing symbols")
    parser.add_argument("--symbol-skip-rows", type=int, default=0)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.84)
    parser.add_argument("--keep-unmatched-intent", action="store_true")
    parser.add_argument("--follow", action="store_true", help="Continuously watch input JSONL")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Folder where NLP-filtered files are saved",
    )
    args = parser.parse_args()

    symbol_sheet: str | int = args.symbol_sheet
    if isinstance(symbol_sheet, str) and symbol_sheet.isdigit():
        symbol_sheet = int(symbol_sheet)

    symbols = load_symbol_set(Path(args.symbol_excel), symbol_sheet, args.symbol_col, args.symbol_skip_rows)
    if args.follow:
        follow_jsonl(Path(args.input), symbols, args)
    else:
        rows = process_events(read_events(Path(args.input)), symbols, args)
        write_rows(rows, Path(args.output_dir))
        print(f"Filtered trade-like messages: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
