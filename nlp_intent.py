#!/usr/bin/env python3
"""
NLP trade intent layer for group chat messages.

Purpose:
  - Filter out casual chat such as "hi", "hello", "hey mate".
  - Use NLTK tokenization/stemming to detect English and Portuguese trade intent.
  - Carry forward side/quantity context per sender.
  - Read per-client JSON session files or a chat JSONL stream.
  - Output accepted intents and reviews as JSONL.

Example:
  Sender says: "buy 10pc NVDA"
  Later same sender says: "CRCL"
  Output becomes: "BUY 10pc CRCL"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


QTY_RE = re.compile(
    r"\b(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>pc|pcs|pct|%|k|m|mm|mn|usd|eur|gbp|shares|shs|lots?)\b",
    re.IGNORECASE,
)

LIMIT_RANGE_PATTERNS = [
    re.compile(
        r"\b(?:at\s+)?(?:limit\s+)?(?P<low>\d+(?:\.\d+)?)\s*(?:-|to)\s*"
        r"(?P<high>\d+(?:\.\d+)?)\s*(?:limit)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blimit\s+(?P<low>\d+(?:\.\d+)?)\s*(?:-|to)\s*"
        r"(?P<high>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
]

LIMIT_SINGLE_PATTERNS = [
    re.compile(
        r"\bat\s+(?P<price>\d+(?:\.\d+)?)\s+limit\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blimit\s+(?P<price>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"@\s*(?P<price>\d+(?:\.\d+)?)\s*(?:limit|lim)\b",
        re.IGNORECASE,
    ),
]

COMPACT_ORDER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<side>B|S|BUY|SELL|BOUGHT|SOLD)\s*"
    r"(?P<quantity>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>K|M|MM|MN)?\s+"
    r"(?P<ticker>[A-Za-z][A-Za-z0-9.\-]{0,9})"
    r"(?:\s*(?:@|FOR|AT)\s*(?P<price>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

BUY_TERMS = {
    "b",
    "buy",
    "buyer",
    "bought",
    "bid",
    "lift",
    "lifted",
    "load",
    "loaded",
    "pay",
    "paid",
    "payer",
    "take",
    "took",
    "work",
    "working",
    "compra",
    "comprar",
    "compro",
    "comprei",
    "comprando",
    "pagar",
    "pago",
}

SELL_TERMS = {
    "s",
    "sell",
    "seller",
    "selling",
    "sold",
    "offer",
    "offered",
    "hit",
    "hitting",
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
    "EACH",
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
    "LIMIT",
    "MATE",
    "ME",
    "NO",
    "NOITE",
    "OF",
    "OK",
    "ON",
    "OR",
    "PLEASE",
    "PRICE",
    "PC",
    "PCS",
    "TARDE",
    "THANKS",
    "THE",
    "TO",
    "USD",
    "US",
    "YOU",
}

NEWS_TERMS = {
    "ALERT",
    "CALL",
    "CNBC",
    "CONFERENCE",
    "EARNINGS",
    "EQUITY",
    "FORECAST",
    "HOSTED",
    "INDICATIONS",
    "MOVERS",
    "NEWS",
    "NOTE",
    "OUTLOOK",
    "REPORT",
    "REPORTED",
    "RESEARCH",
    "TV",
}

CONTINUATION_TERMS = {
    "ALSO",
    "AND",
    "EACH",
    "INLINE",
    "POV",
    "SAME",
    "TOP",
}

RELOAD_TERMS = {"REBUY", "RELOAD"}
RESET_PHRASES = {
    "CANCEL",
    "CANCEL ALL",
    "DONE",
    "FINISHED",
    "THATS ALL",
    "THATS IT",
    "WE ARE GOOD",
}

PORTUGUESE_STRONG_HINTS = {
    "COMPRAR",
    "COMPRA",
    "COMPREI",
    "COMPRO",
    "CONSOLIDADO",
    "FALA",
    "MANDA",
    "MESTRE",
    "PAGAR",
    "PAGO",
    "TRABALHA",
    "VENDA",
    "VENDE",
    "VENDER",
    "VENDIDO",
}

PORTUGUESE_COMMON_HINTS = {
    "COMO",
    "CONSEGUIR",
    "DE",
    "ESSA",
    "ISSO",
    "OBRIGADO",
    "PARA",
    "POR",
    "TUDO",
}


@dataclass
class SenderContext:
    room_id: str = ""
    sender_id: str = ""
    side: str = ""
    quantity: str = ""
    quantity_unit: str = ""
    limit_price_low: str = ""
    limit_price_high: str = ""
    tickers: tuple[str, ...] = ()
    last_updated: str = ""
    active: bool = False


class SessionStore:
    def __init__(self, db_path: Path, timeout_minutes: int) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.timeout = timedelta(minutes=timeout_minutes)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                room_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT '',
                quantity TEXT NOT NULL DEFAULT '',
                quantity_unit TEXT NOT NULL DEFAULT '',
                limit_price_low TEXT NOT NULL DEFAULT '',
                limit_price_high TEXT NOT NULL DEFAULT '',
                tickers_json TEXT NOT NULL DEFAULT '[]',
                last_updated TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (room_id, sender_id)
            )
            """
        )
        columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
        }
        if "limit_price_low" not in columns:
            self.connection.execute(
                "ALTER TABLE chat_sessions ADD COLUMN limit_price_low TEXT NOT NULL DEFAULT ''"
            )
        if "limit_price_high" not in columns:
            self.connection.execute(
                "ALTER TABLE chat_sessions ADD COLUMN limit_price_high TEXT NOT NULL DEFAULT ''"
            )
        self.connection.commit()

    def get(self, room_id: str, sender_id: str) -> SenderContext:
        row = self.connection.execute(
            "SELECT * FROM chat_sessions WHERE room_id = ? AND sender_id = ?",
            (room_id, sender_id),
        ).fetchone()
        if row is None:
            return SenderContext(room_id=room_id, sender_id=sender_id)

        context = SenderContext(
            room_id=row["room_id"],
            sender_id=row["sender_id"],
            side=row["side"],
            quantity=row["quantity"],
            quantity_unit=row["quantity_unit"],
            limit_price_low=row["limit_price_low"],
            limit_price_high=row["limit_price_high"],
            tickers=tuple(json.loads(row["tickers_json"])),
            last_updated=row["last_updated"],
            active=bool(row["active"]),
        )
        if self.is_expired(context):
            self.clear(room_id, sender_id)
            return SenderContext(room_id=room_id, sender_id=sender_id)
        return context

    def save(self, context: SenderContext) -> None:
        context.last_updated = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO chat_sessions (
                room_id, sender_id, side, quantity, quantity_unit,
                limit_price_low, limit_price_high, tickers_json, last_updated, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_id, sender_id) DO UPDATE SET
                side = excluded.side,
                quantity = excluded.quantity,
                quantity_unit = excluded.quantity_unit,
                limit_price_low = excluded.limit_price_low,
                limit_price_high = excluded.limit_price_high,
                tickers_json = excluded.tickers_json,
                last_updated = excluded.last_updated,
                active = excluded.active
            """,
            (
                context.room_id,
                context.sender_id,
                context.side,
                context.quantity,
                context.quantity_unit,
                context.limit_price_low,
                context.limit_price_high,
                json.dumps(list(context.tickers)),
                context.last_updated,
                int(context.active),
            ),
        )
        self.connection.commit()

    def clear(self, room_id: str, sender_id: str) -> None:
        self.connection.execute(
            "DELETE FROM chat_sessions WHERE room_id = ? AND sender_id = ?",
            (room_id, sender_id),
        )
        self.connection.commit()

    def is_expired(self, context: SenderContext) -> bool:
        if not context.active or not context.last_updated:
            return True
        try:
            updated = datetime.fromisoformat(context.last_updated)
        except ValueError:
            return True
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - updated > self.timeout

    def close(self) -> None:
        self.connection.close()


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


class MessageTranslator:
    def __init__(self, retries: int = 3) -> None:
        self.cache: dict[str, tuple[str, str, str]] = {}
        self.retries = retries
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise SystemExit("Install mandatory translation support: pip install deep-translator") from exc
        self.translator: Any = GoogleTranslator(source="pt", target="en")

    @staticmethod
    def looks_portuguese(message: str, nlp: NlpEngine) -> bool:
        tokens = set(nlp.normalized_tokens(message))
        if tokens & PORTUGUESE_STRONG_HINTS:
            return True
        return len(tokens & PORTUGUESE_COMMON_HINTS) >= 2

    def translate(self, message: str, nlp: NlpEngine) -> tuple[str, str, str]:
        if message in self.cache:
            return self.cache[message]
        if not self.looks_portuguese(message, nlp):
            result = (message, "en_or_unknown", "not_needed")
            self.cache[message] = result
            return result

        result = (message, "pt", "translation_error:unknown")
        for attempt in range(1, self.retries + 1):
            try:
                translated = str(self.translator.translate(message)).strip()
                if translated and translated.casefold() != message.casefold():
                    result = (translated, "pt", "translated")
                    break
                result = (message, "pt", "translation_error:unchanged")
            except Exception as exc:
                result = (message, "pt", f"translation_error:{type(exc).__name__}")
            if attempt < self.retries:
                time.sleep(0.5 * attempt)

        self.cache[message] = result
        return result


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def clean_message(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_message(value)).strip("._")
    return name or "unknown"


def excel_col_to_index(col: str) -> int:
    value = 0
    for char in col.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid Excel column: {col}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def load_symbol_set(
    file_path: Path,
    sheet: str | int,
    symbol_col: str,
    skip_rows: int,
) -> set[str]:
    suffix = file_path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".xlsm"}:
        raise SystemExit("Symbol file must be CSV or Excel (.csv, .xlsx, .xls, .xlsm)")
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Symbol source needs pandas/openpyxl: pip install pandas openpyxl") from exc

    if suffix == ".csv":
        frame = pd.read_csv(file_path, header=None, skiprows=skip_rows, dtype=str)
    else:
        frame = pd.read_excel(
            file_path,
            sheet_name=sheet,
            header=None,
            skiprows=skip_rows,
            dtype=str,
        )

    requested = clean_message(symbol_col)
    header_values = [
        clean_message(value).casefold()
        for value in frame.iloc[0].tolist()
    ] if not frame.empty else []
    requested_header = requested.casefold()
    if requested_header in header_values:
        col_idx = header_values.index(requested_header)
        start_row = 1
    elif requested_header == "symbol" and "symbol" in header_values:
        col_idx = header_values.index("symbol")
        start_row = 1
    elif re.fullmatch(r"[A-Za-z]{1,3}", requested):
        col_idx = excel_col_to_index(requested)
        start_row = 1 if (
            col_idx < len(header_values)
            and header_values[col_idx] in {"symbol", "ticker"}
        ) else 0
    else:
        raise SystemExit(
            f"Symbol column '{symbol_col}' was not found. "
            "Use a header name such as Symbol or an Excel column such as A."
        )

    symbols: set[str] = set()
    for _, row in frame.iloc[start_row:].iterrows():
        if col_idx >= len(row) or pd.isna(row.iloc[col_idx]):
            continue
        symbol = normalize(str(row.iloc[col_idx]).strip())
        if symbol:
            symbols.add(symbol)
    if not symbols:
        raise SystemExit(f"No symbols found in {file_path}")
    return symbols


def client_document_events(path: Path) -> Iterable[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    client_id = clean_message(document.get("client_id"))
    client_name = clean_message(document.get("client_name"))
    for session in document.get("sessions") or []:
        session_id = clean_message(session.get("session_id"))
        room_id = clean_message(session.get("room_id")) or "UNKNOWN_ROOM"
        messages = session.get("messages") or session.get("client_messages") or []
        for item in messages:
            raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
            event = dict(raw)
            event.setdefault("event_id", item.get("event_id"))
            event.setdefault("source_timestamp", item.get("timestamp"))
            event.setdefault("room_id", room_id)
            event.setdefault("sender_id", item.get("sender_id") or client_id)
            event.setdefault("sender_name", item.get("sender_name") or client_name)
            event.setdefault("message", item.get("message"))
            event["client_session_id"] = session_id
            yield event


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for client_path in sorted(path.glob("*.json")):
            yield from client_document_events(client_path)
        return
    if path.suffix.lower() == ".json":
        yield from client_document_events(path)
        return
    if path.suffix.lower() != ".jsonl":
        raise SystemExit("--input must be a client JSON directory, .json, or .jsonl")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def detect_quantity(message: str, tickers: list[str] | None = None) -> tuple[str, str]:
    match = QTY_RE.search(message)
    if match:
        quantity = match.group("num").replace(",", ".")
        unit = match.group("unit").lower()
        return quantity, unit

    if not tickers:
        return "", ""

    ticker_pattern = "|".join(re.escape(ticker) for ticker in tickers)
    bare_qty_re = re.compile(
        rf"\b(?:B|S|BUY|SELL|SOLD|VENDE|COMPRA)?\s*"
        rf"(?P<num>\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+(?:\.\d+)?)\s+"
        rf"(?P<ticker>{ticker_pattern})\b",
        re.IGNORECASE,
    )
    bare_match = bare_qty_re.search(message)
    if not bare_match:
        return "", ""
    return bare_match.group("num").replace(",", ""), ""


def detect_limit_price(message: str) -> tuple[str, str]:
    for pattern in LIMIT_RANGE_PATTERNS:
        match = pattern.search(message)
        if match:
            low = match.group("low")
            high = match.group("high")
            if float(low) > float(high):
                low, high = high, low
            return low, high

    for pattern in LIMIT_SINGLE_PATTERNS:
        match = pattern.search(message)
        if match:
            price = match.group("price")
            return price, price

    return "", ""


def parse_compact_order_legs(message: str, symbol_set: set[str]) -> list[dict[str, str]]:
    normalized_message = re.sub(
        r"(?<=\d)(?=(?:B|S)\s*\d)",
        " ",
        message,
        flags=re.IGNORECASE,
    )
    legs: list[dict[str, str]] = []

    for match in COMPACT_ORDER_RE.finditer(normalized_message):
        ticker = normalize(match.group("ticker"))
        if ticker not in symbol_set:
            continue

        side_token = match.group("side").upper()
        side = "BUY" if side_token in {"B", "BUY", "BOUGHT"} else "SELL"
        quantity = match.group("quantity").replace(",", ".")
        unit = (match.group("unit") or "").lower()
        price = match.group("price") or ""

        legs.append(
            {
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "quantity_unit": unit,
                "limit_price_low": price,
                "limit_price_high": price,
                "price_type": "EXECUTED_PRICE" if price else "MARKET_OR_UNSPECIFIED",
            }
        )

    return legs


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
    command_words = {
        normalize(term)
        for term in BUY_TERMS | SELL_TERMS | RELOAD_TERMS
    }
    for raw in nlp.tokens(message):
        token = normalize(raw.removeprefix("$"))
        if len(token) < 2 or token in STOPWORDS or token in command_words:
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


def looks_like_news_or_research(message: str, nlp: NlpEngine) -> bool:
    tokens = set(nlp.normalized_tokens(message))
    return bool(tokens & NEWS_TERMS)


def is_context_continuation(message: str, tickers: list[str], nlp: NlpEngine) -> bool:
    if not tickers or looks_like_news_or_research(message, nlp):
        return False

    tokens = nlp.normalized_tokens(message)
    if len(tokens) <= 6:
        return True

    if tokens and tokens[0] in CONTINUATION_TERMS:
        return True

    lowered = message.lower()
    return "same for" in lowered or "each of" in lowered or "on top of prior" in lowered


def event_sender_id(event: dict[str, Any]) -> str:
    return str(
        event.get("sender_id")
        or event.get("participant_id")
        or event.get("sender_name")
        or event.get("sender")
        or "UNKNOWN"
    )


def event_room_id(event: dict[str, Any]) -> str:
    return str(
        event.get("room_id")
        or event.get("chat_id")
        or event.get("room_name")
        or event.get("room")
        or "UNKNOWN_ROOM"
    )


def normalized_phrase(message: str, nlp: NlpEngine) -> str:
    return " ".join(nlp.normalized_tokens(message))


def is_reset_message(message: str, nlp: NlpEngine) -> bool:
    return normalized_phrase(message, nlp) in RESET_PHRASES


def is_reload_message(message: str, nlp: NlpEngine) -> bool:
    return bool(set(nlp.normalized_tokens(message)) & RELOAD_TERMS)


def build_output_row(
    event: dict[str, Any],
    original_message: str,
    translated_message: str,
    detected_language: str,
    translation_status: str,
    side: str,
    quantity: str,
    quantity_unit: str,
    tickers: list[str],
    context_applied: bool,
    limit_price_low: str = "",
    limit_price_high: str = "",
    intent_type: str = "NEW_ORDER",
    confidence: float = 0.95,
    processing_status: str = "accepted",
    review_reason: str = "",
) -> dict[str, str]:
    quantity_text = f"{quantity}{quantity_unit}" if quantity and quantity_unit else quantity
    if limit_price_low and limit_price_high and limit_price_low != limit_price_high:
        limit_text = f"LIMIT {limit_price_low}-{limit_price_high}"
        price_type = "LIMIT_RANGE"
    elif limit_price_low:
        limit_text = f"LIMIT {limit_price_low}"
        price_type = "LIMIT"
    else:
        limit_text = ""
        price_type = "MARKET_OR_UNSPECIFIED"

    normalized_instruction = " ".join(
        part for part in [side, quantity_text, " ".join(tickers), limit_text] if part
    )
    orders = [
        {
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "quantity_unit": quantity_unit,
            "limit_price_low": limit_price_low,
            "limit_price_high": limit_price_high,
            "price_type": price_type,
        }
        for ticker in tickers
    ]
    return {
        "source_event_id": str(
            event.get("event_id") or event.get("source_event_id") or ""
        ),
        "captured_at": str(event.get("captured_at") or ""),
        "source_timestamp": str(event.get("source_timestamp") or ""),
        "room_id": event_room_id(event),
        "sender_id": event_sender_id(event),
        "sender_name": str(event.get("sender_name") or event.get("sender") or ""),
        "session_key": f"{event_room_id(event)}:{event_sender_id(event)}",
        "client_session_id": str(event.get("client_session_id") or ""),
        "message": normalized_instruction,
        "original_message": original_message,
        "translated_message": translated_message,
        "detected_language": detected_language,
        "translation_status": translation_status,
        "trade_side": side,
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "limit_price_low": limit_price_low,
        "limit_price_high": limit_price_high,
        "price_type": price_type,
        "candidate_tickers": "|".join(tickers),
        "orders_json": json.dumps(orders, ensure_ascii=False),
        "context_applied": str(context_applied),
        "intent_type": intent_type,
        "confidence": f"{confidence:.2f}",
        "processing_status": processing_status,
        "review_reason": review_reason,
        "source_row_num": str(event.get("source_row_num") or event.get("row_num") or ""),
    }


def output_fields() -> list[str]:
    return [
        "source_event_id",
        "captured_at",
        "source_timestamp",
        "room_id",
        "sender_id",
        "sender_name",
        "session_key",
        "client_session_id",
        "message",
        "original_message",
        "translated_message",
        "detected_language",
        "translation_status",
        "trade_side",
        "quantity",
        "quantity_unit",
        "limit_price_low",
        "limit_price_high",
        "price_type",
        "candidate_tickers",
        "orders_json",
        "context_applied",
        "intent_type",
        "confidence",
        "processing_status",
        "review_reason",
        "source_row_num",
    ]


def build_review_row(
    event: dict[str, Any],
    original_message: str,
    translated_message: str,
    detected_language: str,
    translation_status: str,
    reason: str,
) -> dict[str, str]:
    return build_output_row(
        event,
        original_message,
        translated_message,
        detected_language,
        translation_status,
        "",
        "",
        "",
        [],
        False,
        intent_type="REVIEW",
        confidence=0.0,
        processing_status="review",
        review_reason=reason,
    )


def build_multi_leg_row(
    event: dict[str, Any],
    original_message: str,
    translated_message: str,
    detected_language: str,
    translation_status: str,
    legs: list[dict[str, str]],
) -> dict[str, str]:
    tickers = [leg["ticker"] for leg in legs]
    row = build_output_row(
        event,
        original_message,
        translated_message,
        detected_language,
        translation_status,
        "MIXED",
        "",
        "",
        tickers,
        False,
        intent_type="MULTI_LEG_EXECUTION",
        confidence=0.99,
    )
    row["message"] = " | ".join(
        " ".join(
            part
            for part in [
                leg["side"],
                f'{leg["quantity"]}{leg["quantity_unit"]}',
                leg["ticker"],
                f'@ {leg["limit_price_low"]}' if leg["limit_price_low"] else "",
            ]
            if part
        )
        for leg in legs
    )
    row["trade_side"] = "MIXED"
    row["quantity"] = ""
    row["quantity_unit"] = ""
    row["limit_price_low"] = ""
    row["limit_price_high"] = ""
    row["price_type"] = "MULTI_LEG_EXECUTION"
    row["orders_json"] = json.dumps(legs, ensure_ascii=False)
    return row


def process_one_event(
    event: dict[str, Any],
    symbol_set: set[str],
    sessions: SessionStore,
    keep_unmatched_intent: bool,
    fuzzy_threshold: float,
    nlp: NlpEngine,
    translator: MessageTranslator,
) -> dict[str, str] | None:
    message = clean_message(event.get("message"))
    if not message:
        return None

    translated_message, detected_language, translation_status = translator.translate(message, nlp)
    if detected_language == "pt" and translation_status != "translated":
        return build_review_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            "mandatory_portuguese_translation_failed",
        )

    if is_noise(message, nlp) or (
        translated_message != message and is_noise(translated_message, nlp)
    ):
        return None

    room_id = event_room_id(event)
    sender_id = event_sender_id(event)
    context = sessions.get(room_id, sender_id)

    if is_reset_message(translated_message, nlp):
        sessions.clear(room_id, sender_id)
        return None

    compact_legs = parse_compact_order_legs(message, symbol_set)
    if len(compact_legs) >= 2:
        return build_multi_leg_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            compact_legs,
        )

    side = nlp.detect_side(translated_message) or nlp.detect_side(message)
    tickers = extract_symbols(message, symbol_set, fuzzy_threshold, nlp)
    quantity, quantity_unit = detect_quantity(message, tickers)
    limit_price_low, limit_price_high = detect_limit_price(translated_message)
    reload_requested = is_reload_message(translated_message, nlp)

    if reload_requested:
        reload_tickers = tickers or list(context.tickers)
        reload_quantity = quantity or context.quantity
        reload_unit = quantity_unit if quantity else context.quantity_unit
        reload_limit_low = limit_price_low or context.limit_price_low
        reload_limit_high = limit_price_high or context.limit_price_high
        if not context.active or not reload_tickers or not reload_quantity:
            return build_review_row(
                event,
                message,
                translated_message,
                detected_language,
                translation_status,
                "reload_without_prior_active_order",
            )

        context.side = "BUY"
        context.quantity = reload_quantity
        context.quantity_unit = reload_unit
        context.limit_price_low = reload_limit_low
        context.limit_price_high = reload_limit_high
        context.tickers = tuple(reload_tickers)
        context.active = True
        sessions.save(context)
        return build_output_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            "BUY",
            reload_quantity,
            reload_unit,
            reload_tickers,
            True,
            limit_price_low=reload_limit_low,
            limit_price_high=reload_limit_high,
            intent_type="RELOAD",
            confidence=0.96,
        )

    has_explicit_intent = bool(side or quantity or limit_price_low)
    can_use_context = bool(context.side and context.quantity) and is_context_continuation(
        translated_message,
        tickers,
        nlp,
    )

    if has_explicit_intent and tickers:
        final_side = side or context.side
        final_quantity = quantity or context.quantity
        final_unit = quantity_unit if quantity else context.quantity_unit
        final_limit_low = limit_price_low or context.limit_price_low
        final_limit_high = limit_price_high or context.limit_price_high
        context.side = final_side
        context.quantity = final_quantity
        context.quantity_unit = final_unit
        context.limit_price_low = final_limit_low
        context.limit_price_high = final_limit_high
        context.tickers = tuple(tickers)
        context.active = True
        sessions.save(context)
        return build_output_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            final_side,
            final_quantity,
            final_unit,
            tickers,
            False,
            limit_price_low=final_limit_low,
            limit_price_high=final_limit_high,
            intent_type="NEW_ORDER",
            confidence=0.98 if side and quantity else 0.92,
        )

    if tickers and can_use_context:
        context.tickers = tuple(tickers)
        context.active = True
        sessions.save(context)
        return build_output_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            context.side,
            context.quantity,
            context.quantity_unit,
            tickers,
            True,
            limit_price_low=context.limit_price_low,
            limit_price_high=context.limit_price_high,
            intent_type="CONTINUATION",
            confidence=0.90,
        )

    if has_explicit_intent and not tickers:
        if side:
            context.side = side
        if quantity:
            context.quantity = quantity
            context.quantity_unit = quantity_unit
        if limit_price_low:
            context.limit_price_low = limit_price_low
            context.limit_price_high = limit_price_high
        context.active = bool(context.side and context.quantity)
        sessions.save(context)
        if not keep_unmatched_intent:
            return None
        return build_output_row(
            event,
            message,
            translated_message,
            detected_language,
            translation_status,
            side,
            quantity,
            quantity_unit,
            [],
            False,
            limit_price_low=limit_price_low,
            limit_price_high=limit_price_high,
            intent_type="CONTEXT_SETUP",
            confidence=0.88,
        )

    return None


def process_events(events: Iterable[dict[str, Any]], symbol_set: set[str], args: argparse.Namespace) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    nlp = NlpEngine()
    translator = MessageTranslator(args.translation_retries)
    sessions = SessionStore(Path(args.session_db), args.session_timeout_minutes)

    try:
        for event in events:
            row = process_one_event(
                event,
                symbol_set,
                sessions,
                args.keep_unmatched_intent,
                args.fuzzy_threshold,
                nlp,
                translator,
            )
            if row:
                output_rows.append(row)
    finally:
        sessions.close()

    return output_rows


def write_rows(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "nlp_trade_intent_messages.jsonl"
    review_jsonl_path = output_dir / "nlp_trade_review.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, review_jsonl_path.open(
        "w", encoding="utf-8"
    ) as review_jsonl_handle:
        for row in rows:
            if row["processing_status"] == "review":
                review_jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"NLP trade intent JSONL: {jsonl_path}")
    print(f"NLP review JSONL:       {review_jsonl_path}")


def write_client_intent_files(
    rows: list[dict[str, str]],
    client_output_dir: Path,
) -> None:
    client_output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(clean_message(row.get("sender_id")), []).append(row)

    for client_id, client_rows in grouped.items():
        accepted = [
            row for row in client_rows if row["processing_status"] == "accepted"
        ]
        reviews = [
            row for row in client_rows if row["processing_status"] == "review"
        ]
        document = {
            "record_type": "client_trade_intents",
            "client_id": client_id,
            "client_name": next(
                (
                    clean_message(row.get("sender_name"))
                    for row in client_rows
                    if clean_message(row.get("sender_name"))
                ),
                "",
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "accepted_intent_count": len(accepted),
            "review_message_count": len(reviews),
            "accepted_intents": accepted,
            "review_messages": reviews,
        }
        path = client_output_dir / f"{safe_filename(client_id)}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def upsert_client_intent_file(
    row: dict[str, str],
    client_output_dir: Path,
) -> None:
    client_id = clean_message(row.get("sender_id"))
    path = client_output_dir / f"{safe_filename(client_id)}.json"
    existing_rows: list[dict[str, str]] = []
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        existing_rows.extend(document.get("accepted_intents") or [])
        existing_rows.extend(document.get("review_messages") or [])

    source_event_id = clean_message(row.get("source_event_id"))
    if any(
        clean_message(item.get("source_event_id")) == source_event_id
        and clean_message(item.get("client_session_id"))
        == clean_message(row.get("client_session_id"))
        for item in existing_rows
    ):
        return
    existing_rows.append(row)
    write_client_intent_files(existing_rows, client_output_dir)


def follow_jsonl(input_path: Path, symbol_set: set[str], args: argparse.Namespace) -> None:
    if input_path.suffix.lower() != ".jsonl":
        raise SystemExit("--follow expects JSONL input")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "nlp_trade_intent_messages.jsonl"
    review_jsonl_path = output_dir / "nlp_trade_review.jsonl"
    nlp = NlpEngine()
    translator = MessageTranslator(args.translation_retries)
    sessions = SessionStore(Path(args.session_db), args.session_timeout_minutes)

    print(f"Following chat JSONL: {input_path}")
    print(f"NLP trade intent JSONL: {jsonl_path}")
    print(f"NLP review JSONL:       {review_jsonl_path}")

    while not input_path.exists():
        print(f"Waiting for input file: {input_path}", flush=True)
        time.sleep(args.poll_interval)

    try:
        with input_path.open("r", encoding="utf-8-sig") as input_handle, jsonl_path.open(
            "a", encoding="utf-8"
        ) as jsonl_handle, review_jsonl_path.open(
            "a", encoding="utf-8"
        ) as review_jsonl_handle:
            while True:
                line = input_handle.readline()
                if not line:
                    time.sleep(args.poll_interval)
                    continue

                row = process_one_event(
                    json.loads(line),
                    symbol_set,
                    sessions,
                    args.keep_unmatched_intent,
                    args.fuzzy_threshold,
                    nlp,
                    translator,
                )
                if not row:
                    continue

                if row["processing_status"] == "review":
                    review_jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    review_jsonl_handle.flush()
                else:
                    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_handle.flush()
                upsert_client_intent_file(row, Path(args.client_output_dir))
                print(
                    f'{row["source_timestamp"]} session={row["session_key"]} '
                    f'status={row["processing_status"]} intent={row["message"]} '
                    f'original={row["original_message"]}',
                    flush=True,
                )
    finally:
        sessions.close()


def follow_client_directory(
    input_dir: Path,
    symbol_set: set[str],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "nlp_trade_intent_messages.jsonl"
    review_jsonl_path = output_dir / "nlp_trade_review.jsonl"
    nlp = NlpEngine()
    translator = MessageTranslator(args.translation_retries)
    sessions = SessionStore(Path(args.session_db), args.session_timeout_minutes)
    seen: set[str] = set()
    for existing_path in (jsonl_path, review_jsonl_path):
        if not existing_path.exists():
            continue
        with existing_path.open("r", encoding="utf-8-sig") as existing:
            for line in existing:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_id = clean_message(row.get("source_event_id"))
                if source_id:
                    seen.add(
                        "|".join(
                            [
                                clean_message(row.get("sender_id")),
                                clean_message(row.get("room_id")),
                                source_id,
                            ]
                        )
                    )

    print(f"Following client JSON directory: {input_dir}")
    print(f"NLP trade intent JSONL: {jsonl_path}")
    print(f"NLP review JSONL:       {review_jsonl_path}")

    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl_handle, review_jsonl_path.open(
            "a", encoding="utf-8"
        ) as review_jsonl_handle:
            while True:
                events = sorted(
                    read_events(input_dir),
                    key=lambda event: str(
                        event.get("source_timestamp")
                        or event.get("captured_at")
                        or ""
                    ),
                )
                for event in events:
                    source_id = clean_message(
                        event.get("event_id") or event.get("source_event_id")
                    )
                    key = "|".join(
                        [
                            event_sender_id(event),
                            event_room_id(event),
                            source_id,
                        ]
                    )
                    if not source_id:
                        key = json.dumps(event, sort_keys=True, ensure_ascii=False)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = process_one_event(
                        event,
                        symbol_set,
                        sessions,
                        args.keep_unmatched_intent,
                        args.fuzzy_threshold,
                        nlp,
                        translator,
                    )
                    if not row:
                        continue
                    target = (
                        review_jsonl_handle
                        if row["processing_status"] == "review"
                        else jsonl_handle
                    )
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    target.flush()
                    upsert_client_intent_file(row, Path(args.client_output_dir))
                    print(
                        f'{row["source_timestamp"]} client={row["sender_id"]} '
                        f'session={row["client_session_id"]} '
                        f'status={row["processing_status"]} intent={row["message"]}',
                        flush=True,
                    )
                time.sleep(args.poll_interval)
    finally:
        sessions.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Per-client JSON directory, one client JSON file, or chat JSONL",
    )
    parser.add_argument(
        "--symbol-file",
        "--symbol-excel",
        dest="symbol_file",
        required=True,
        help="CSV or Excel symbol master containing a Symbol column",
    )
    parser.add_argument("--symbol-sheet", default=0, help="Sheet name or zero-based index")
    parser.add_argument(
        "--symbol-col",
        default="Symbol",
        help="Symbol header name or Excel column letter (default: Symbol)",
    )
    parser.add_argument("--symbol-skip-rows", type=int, default=0)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.84)
    parser.add_argument("--translation-retries", type=int, default=3)
    parser.add_argument("--session-timeout-minutes", type=int, default=10)
    parser.add_argument(
        "--session-db",
        default=str(Path(__file__).resolve().parent / "nlp_chat_sessions.sqlite3"),
        help="SQLite database for persistent room-plus-user sessions",
    )
    parser.add_argument("--keep-unmatched-intent", action="store_true")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuously watch input JSONL or per-client JSON directory",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Folder where NLP-filtered files are saved",
    )
    parser.add_argument(
        "--client-output-dir",
        default="",
        help=(
            "Directory for one NLP JSON file per client. "
            "Defaults to <output-dir>/client_intents_by_client."
        ),
    )
    args = parser.parse_args()
    if not args.client_output_dir:
        args.client_output_dir = str(
            Path(args.output_dir) / "client_intents_by_client"
        )

    symbol_sheet: str | int = args.symbol_sheet
    if isinstance(symbol_sheet, str) and symbol_sheet.isdigit():
        symbol_sheet = int(symbol_sheet)

    input_path = Path(args.input)
    symbols = load_symbol_set(
        Path(args.symbol_file),
        symbol_sheet,
        args.symbol_col,
        args.symbol_skip_rows,
    )
    if args.follow:
        if input_path.is_dir():
            follow_client_directory(input_path, symbols, args)
        else:
            follow_jsonl(input_path, symbols, args)
    else:
        events = sorted(
            read_events(input_path),
            key=lambda event: str(
                event.get("source_timestamp")
                or event.get("captured_at")
                or ""
            ),
        )
        rows = process_events(events, symbols, args)
        write_rows(rows, Path(args.output_dir))
        client_output_dir = Path(args.client_output_dir)
        client_output_dir.mkdir(parents=True, exist_ok=True)
        for path in client_output_dir.glob("*.json"):
            path.unlink()
        write_client_intent_files(rows, client_output_dir)
        print(f"Per-client NLP output:  {client_output_dir}")
        print(f"Filtered trade-like messages: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
