# fetch_data.py
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

# Stooq per-symbol daily CSV with bounded range:
# https://stooq.com/q/d/l/?s=aapl.us&i=d&d1=20170101&d2=20260106
STOOQ_URL_TMPL = "https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQ_TIMEOUT = 20
PER_TICKER_RETRIES = 3
SLEEP_BASE = 0.25

PAUSE_EVERY = 1
PAUSE_SECS = 0.25

OUT_PARQUET = "sp500_etf.parquet"
OUT_FAILURES = "sp500_etf_failures.csv"
OUT_META = "sp500_etf_meta.json"


def normalize_symbol_for_yahoo(s: str) -> str:
    return str(s).strip().replace(".", "-")  # BRK.B -> BRK-B


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


def _get_text(session: requests.Session, url: str):
    r = session.get(url, timeout=REQ_TIMEOUT, headers={"User-Agent": UA, "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.1"})
    return r.status_code, r.headers.get("content-type", ""), r.text


def download_stooq_close_one(
    session: requests.Session, ticker: str, start: str, end: str
) -> pd.Series:
    sym = to_stooq_symbol(ticker)
    url = STOOQ_URL_TMPL.format(symbol=sym, d1=_yyyymmdd(start), d2=_yyyymmdd(end))

    last_err = None
    for attempt in range(1, PER_TICKER_RETRIES + 1):
        try:
            status, ct, txt = _get_text(session, url)
            
            head = (txt[:200] or "").strip().lower()
            
            # Permanent "no data" from Stooq
            if head.startswith("no data"):
                raise RuntimeError("no data")
            
            # Transient: rate-limit / block / HTML / anything non-CSV
            is_htmlish = head.startswith("<!doctype") or head.startswith("<html") or "too many requests" in head
            is_not_csv = (not head.startswith("date,")) and ("date,open,high,low,close" not in head)
            
            if status in (429, 500, 502, 503, 504) or is_htmlish or is_not_csv:
                raise RuntimeError(f"transient non-csv response (status={status}, ct={ct}, head={head[:80]})")
            
            # df = pd.read_csv(StringIO(txt), usecols=["Date", "Close"])            
            df = pd.read_csv(StringIO(txt))  # Date,Open,High,Low,Close,Volume
            if df.empty or "Date" not in df.columns or "Close" not in df.columns:
                raise RuntimeError("empty or missing columns")

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
                time.sleep(2.0)    # longer pause every 25 ticker to prevent rate limits
    
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
    max_date: str


def fetch_and_save_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    tz = pytz.timezone("America/New_York")
    end_dt = datetime.now(tz).date()

    start_dt = date(end_dt.year - LOOKBACK_YEARS, 1, 1)
    start = start_dt.isoformat()
    end = end_dt.isoformat()

    tickers = get_sp500_tickers() + get_additional_etfs()
    tickers = list(dict.fromkeys(tickers))  # stable de-dupe

    print(f"Tickers: {len(tickers)} | start={start} end={end}")
    data, failures = download_stooq_prices(tickers, start=start, end=end)

    if data.empty:
        raise RuntimeError("No data downloaded from Stooq.")

    max_date = pd.to_datetime(data.index.max()).date().isoformat()

    # Save parquet
    data.to_parquet(OUT_PARQUET, engine="pyarrow")

    # Save failures (for debugging only)
    if failures:
        pd.Series(failures, name="error").to_csv(OUT_FAILURES, header=True)

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
