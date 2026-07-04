"""
Binance Futures (USDT-M) gecmis OHLCV verisi cekme - ccxt ile, sayfalamali.
"""
from __future__ import annotations

import os
import time

import ccxt
import pandas as pd

_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _exchange() -> ccxt.binanceusdm:
    return ccxt.binanceusdm({"enableRateLimit": True})


def fetch_recent(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """Son `bars` kadar mumu ceker (cache YOK — canli kullanim icin taze veri).
    Not: son eleman OLUSMAKTA olan (kapanmamis) mum olabilir; cagiran tarafta at."""
    ex = _exchange()
    tf_ms = _TF_MS[timeframe]
    now = ex.milliseconds()
    since = now - (bars + 2) * tf_ms
    rows: list = []
    cursor = since
    while cursor < now:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1500)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:
            break
        cursor = last + tf_ms
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    return df.iloc[-bars:]


def fetch_ohlcv(symbol: str, timeframe: str, days: int, cache: bool = True) -> pd.DataFrame:
    """symbol ornek: 'BTC/USDT'. Son `days` gunluk veriyi ceker.

    Sonucu data/ altinda cache'ler; tekrar cagrildiginda diskten okur.
    """
    safe = symbol.replace("/", "").upper()
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{safe}_{timeframe}_{days}d.csv")
    cache_path = os.path.abspath(cache_path)
    if cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["time"], index_col="time")
        return df

    ex = _exchange()
    tf_ms = _TF_MS[timeframe]
    now = ex.milliseconds()
    since = now - days * 86_400_000
    all_rows: list = []
    limit = 1500
    cursor = since
    while cursor < now:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:
            break
        cursor = last + tf_ms
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    df = df[~df.index.duplicated(keep="first")]

    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
    return df
