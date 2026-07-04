"""
Sinyal uretimi: Pine'daki 3'lu teyit + HTF filtresi + tradeState mantiginin portu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class StrategyParams:
    ssl_len: int = 60
    at_atr_mult: float = 1.0
    at_atr_len: int = 14
    at_mfi_len: int = 14
    ema_fast_len: int = 50
    ema_slow_len: int = 200
    use_htf_filter: bool = True
    # HTF secimi: None -> Pine'daki "Auto" mantigi (asagida map ile belirlenir)
    htf_timeframe: str | None = None
    # Teyit penceresi: SSL kesismesinden sonra AlphaTrend+EMA teyidi icin taninan
    # mum sayisi. 1 = Pine'daki orijinal "hepsi ayni mumda" davranisi.
    confirm_window: int = 1


# Pine higherTimeframe() fonksiyonunun map karsiligi (base tf -> HTF)
_AUTO_HTF = {
    "1m": "15m", "3m": "15m", "5m": "15m",
    "15m": "1h",
    "30m": "4h", "1h": "4h",
    "2h": "1d", "4h": "1d",
    "1d": "1w",
}


def resolve_htf(base_tf: str, params: StrategyParams) -> str:
    if params.htf_timeframe:
        return params.htf_timeframe
    return _AUTO_HTF.get(base_tf, base_tf)


# ccxt/pandas resample kurallari
_TF_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D", "1w": "1W",
}


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _TF_TO_PANDAS[tf]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    htf = df.resample(rule, label="right", closed="left").agg(agg).dropna()
    return htf


def htf_ema_filter(df: pd.DataFrame, base_tf: str, params: StrategyParams) -> pd.DataFrame:
    """Yuksek zaman dilimi EMA bulutu filtresi.

    Pine: request.security(..., lookahead_off) + [1] ile repaint onlenir.
    Burada HTF EMA'yi KAPANMIS HTF mumlari uzerinde hesaplayip, her base mumuna
    o ana kadar KAPANMIS son HTF degerini merge_asof ile eslestiriyoruz
    (lookahead yok). Ardindan bir HTF mumu daha geri kaydiriyoruz ([1] etkisi).
    """
    htf_tf = resolve_htf(base_tf, params)
    htf = _resample_ohlcv(df, htf_tf)

    htf_fast = ind.ema(htf["close"], params.ema_fast_len)
    htf_slow = ind.ema(htf["close"], params.ema_slow_len)
    # [1]: bir onceki KAPANMIS HTF mumunun degerini kullan
    htf_vals = pd.DataFrame({
        "htf_ema_fast": htf_fast.shift(1),
        "htf_ema_slow": htf_slow.shift(1),
    }).dropna()
    htf_vals = htf_vals.reset_index().rename(columns={htf_vals.index.name or "index": "htf_time"})
    htf_vals.columns = ["htf_time", "htf_ema_fast", "htf_ema_slow"]

    base = df.reset_index().rename(columns={df.index.name or "index": "time"})
    merged = pd.merge_asof(
        base.sort_values("time"),
        htf_vals.sort_values("htf_time"),
        left_on="time", right_on="htf_time", direction="backward",
    ).set_index("time")

    out = pd.DataFrame(index=df.index)
    out["htf_ema_fast"] = merged["htf_ema_fast"].to_numpy()
    out["htf_ema_slow"] = merged["htf_ema_slow"].to_numpy()
    out["htf_bullish"] = out["htf_ema_fast"] > out["htf_ema_slow"]
    out["htf_bearish"] = out["htf_ema_fast"] < out["htf_ema_slow"]
    return out


def generate_signals(df: pd.DataFrame, base_tf: str, params: StrategyParams) -> pd.DataFrame:
    """Tum indikatorleri hesaplar ve buy/sell sinyallerini uretir.

    df: index = timezone-aware/naive DatetimeIndex, kolonlar open/high/low/close/volume
    """
    ssl = ind.ssl_hybrid(df, params.ssl_len)
    at = ind.alpha_trend(df, params.at_atr_len, params.at_atr_mult, params.at_mfi_len)
    ec = ind.ema_cloud(df, params.ema_fast_len, params.ema_slow_len)
    htf = htf_ema_filter(df, base_tf, params)

    out = df.copy()
    out = out.join([ssl, at, ec, htf])

    w = max(1, int(params.confirm_window))
    if w <= 1:
        # Orijinal Pine davranisi: SSL kesismesi ile teyitler ayni mumda
        ssl_up_recent = out["ssl_cross_up"]
        ssl_down_recent = out["ssl_cross_down"]
    else:
        # Teyit penceresi: son w mum icinde SSL kesismesi oldu VE su an yon dogru.
        # rolling(w).max() bool serisinde "son w mumda en az bir True" demektir.
        ssl_up_recent = (
            out["ssl_cross_up"].rolling(w, min_periods=1).max().astype(bool)
            & (out["ssl_dir"] == 1.0)
        )
        ssl_down_recent = (
            out["ssl_cross_down"].rolling(w, min_periods=1).max().astype(bool)
            & (out["ssl_dir"] == -1.0)
        )

    buy_trigger = ssl_up_recent & out["alpha_up_confirmed"] & out["ema_up_confirmed"]
    sell_trigger = ssl_down_recent & out["alpha_down_confirmed"] & out["ema_down_confirmed"]

    if params.use_htf_filter:
        buy_ok = buy_trigger & out["htf_bullish"].fillna(False)
        sell_ok = sell_trigger & out["htf_bearish"].fillna(False)
    else:
        buy_ok = buy_trigger
        sell_ok = sell_trigger

    # tradeState: ayni yonde tekrar sinyali engelle (Pine ile birebir)
    n = len(out)
    buy_ok_arr = buy_ok.to_numpy()
    sell_ok_arr = sell_ok.to_numpy()
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    state = 0
    for i in range(n):
        b = buy_ok_arr[i] and state != 1
        s = sell_ok_arr[i] and state != -1
        if b:
            state = 1
        if s:
            state = -1
        buy_signal[i] = b
        sell_signal[i] = s

    out["buy_trigger"] = buy_trigger
    out["sell_trigger"] = sell_trigger
    out["buy_signal"] = buy_signal
    out["sell_signal"] = sell_signal
    return out
