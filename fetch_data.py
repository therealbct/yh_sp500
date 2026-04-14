# fetch_data.py
import os
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, date
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import pytz
import requests


SP500_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

LOOKBACK_YEARS = 4
STOOQ_URL_TMPL = "https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"
STOOQ_APIKEY = os.getenv("STOOQ_APIKEY", "").strip()

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQ_TIMEOUT = 20
PER_TICKER_RETRIES = 3
SLEEP_BASE = 0.25

PAUSE_EVERY = 1
PAUSE_SECS = 0.25

OUT_PARQUET = "sp500_etf.parquet"
OUT_FAILURES = "sp500_etf_failures.csv"
OUT_META = "sp500_etf_meta.json"


def normalize_symbol_for_yahoo(s: str) -> str:
    return str(s).strip().replace(".", "-")


def to_stooq_symbol(us_ticker: str) -> str:
    return f"{str(us_ticker).strip().lower()}.us"


def _yyyymmdd(x: str) -> str:
    return pd.Timestamp(x).strftime("%Y%m%d")


def get_sp500_tickers() -> List[str]:
    df = pd.read_csv(SP500_CONSTITUENTS_URL, usecols=["Symbol"])
    return (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .map(normalize_symbol_for_yahoo)
        .dropna()
        .unique()
        .tolist()
    )


def get_additional_etfs() -> List[str]:
    return [
        "SPY", "XOP", "XLE", "USO", "DBC", "GLD", "JETS", "PEJ",
        "VNQ", "IYR", "HYG", "JNK", "ANGL", "DVY", "VYM", "SDIV", "EMB", "HYEM",
    ]


def _build_stooq_url(symbol: str, start: str, end: str) -> str:
    url = STOOQ_URL_TMPL.format(symbol=symbol, d1=_yyyymmdd(start), d2=_yyyymmdd(end))
    if STOOQ_APIKEY:
        url = f"{url}&apikey={STOOQ_APIKEY}"
    return url


def _get_text(session: requests.Session, url: str):
    headers = {
        "User-Agent": UA,
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://stooq.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    r = session.get(url, timeout=REQ_TIMEOUT, headers=headers)
    return r.status_code, r.headers.get("content-type", ""), r.text


def download_stooq_close_one(
    session: requests.Session, ticker: str, start: str, end: str
) -> pd.Series:
    sym = to_stooq_symbol(ticker)
    url = _build_stooq_url(sym, start, end)

    last_err = None
    for attempt in range(1, PER_TICKER_RETRIES + 1):
        try:
            status, ct, txt = _get_text(session, url)
            head = (txt[:400] or "").strip().lower()

            if "get your apikey" in head or "captcha" in head or "&get_apikey" in head:
                raise RuntimeError("stooq requires apikey/captcha for csv download")

            if head.startswith("no data"):
                raise RuntimeError("no data")

            is_htmlish = (
                head.startswith("<!doctype")
                or head.startswith("<html")
                or "too many requests" in head
            )
            is_not_csv = "date,open,high,low,close" not in head

            if status in (429, 500, 502, 503, 504) or is_htmlish or is_not_csv:
                raise RuntimeError(
                    f"non-csv response status={status} ct={ct} head={head[:160]!r}"
                )

            df = pd.read_csv(StringIO(txt))
            if df.empty or "Date" not in df.columns or "Close" not in df.columns:
                raise RuntimeError(f"bad csv columns={list(df.columns)}")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

            s = df["Close"].dropna()
            s.name = ticker
            if s.empty:
                raise RuntimeError("no close data")

            return s

        except Exception as e:
            last_err = e
            time.sleep(min(5.0, SLEEP_BASE * (2 ** attempt)))

    raise RuntimeError(f"{ticker}: {last_err}")


def download_stooq_prices(
    tickers: List[str], start: str, end: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    prices = []
    failures: Dict[str, str] = {}

    with requests.Session() as session:
        for i, t in enumerate(tickers, 1):
            try:
                prices.append(download_stooq_close_one(session, t, start=start, end=end))
            except Exception as e:
                failures[t] = str(e)

            if i % 25 == 0:
                print(f"[stooq] done {i}/{len(tickers)}")
                time.sleep(2.0)

            if PAUSE_EVERY and (i % PAUSE_EVERY == 0):
                time.sleep(PAUSE_SECS)

    df = pd.concat(prices, axis=1) if prices else pd.DataFrame()
    if not df.empty:
        df.index.name = "date"
        df = df.sort_index()
        df = df.loc[:, ~df.columns.duplicated()]

    return df, failures


@dataclass
class Meta:
    generated_at_et: str
    lookback_years: int
    start: str
    end: str
    tickers_requested: int
    tickers_ok: int
    failures: int
    max_date: str | None


def fetch_and_save_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    tz = pytz.timezone("America/New_York")
    end_dt = datetime.now(tz).date()

    start_dt = date(end_dt.year - LOOKBACK_YEARS, 1, 1)
    start = start_dt.isoformat()
    end = end_dt.isoformat()

    tickers = get_sp500_tickers() + get_additional_etfs()
    tickers = list(dict.fromkeys(tickers))

    print(f"Tickers: {len(tickers)} | start={start} end={end}")
    data, failures = download_stooq_prices(tickers, start=start, end=end)

    if failures:
        pd.Series(failures, name="error").to_csv(OUT_FAILURES, header=True)

    if data.empty:
        sample = dict(list(failures.items())[:10])
        raise RuntimeError(
            f"No data downloaded from Stooq. "
            f"apikey_present={bool(STOOQ_APIKEY)} "
            f"sample_failures={sample}"
        )

    max_date = pd.to_datetime(data.index.max()).date().isoformat()
    data.to_parquet(OUT_PARQUET, engine="pyarrow")

    meta = Meta(
        generated_at_et=datetime.now(tz).isoformat(),
        lookback_years=LOOKBACK_YEARS,
        start=start,
        end=end,
        tickers_requested=len(tickers),
        tickers_ok=int(data.shape[1]),
        failures=len(failures),
        max_date=max_date,
    )
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"Saved: {OUT_PARQUET} | cols={data.shape[1]} | max_date={max_date} | failures={len(failures)}")
    return data, failures


if __name__ == "__main__":
    fetch_and_save_data()
