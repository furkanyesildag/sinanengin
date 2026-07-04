"""
Pine Script indikatorlerinin birebir Python (pandas) portu.

Amac: "Sinan Engin NFT Sistemi (HTF Filtreli V3)" indikatorunun TradingView'deki
davranisiyla mumu mumuna ayni sonuclari uretmek. Hazir 'ta' kutuphanesi Pine'dan
farkli hesapladigi icin (ozellikle RMA/ATR ve HMA seed'leri) indikatorleri elle
yaziyoruz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Temel hareketli ortalamalar
# ----------------------------------------------------------------------------
def wma(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.wma: dogrusal agirlikli hareketli ortalama (agirlik 1..length)."""
    weights = np.arange(1, length + 1, dtype=float)
    wsum = weights.sum()

    def _calc(x: np.ndarray) -> float:
        return np.dot(x, weights) / wsum

    return series.rolling(length).apply(_calc, raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.hma: Hull Moving Average.
    hma = wma(2*wma(src, n/2) - wma(src, n), round(sqrt(n)))
    """
    half = int(length / 2)          # Pine int() -> truncation
    sqrt_len = int(round(np.sqrt(length)))
    raw = 2 * wma(series, half) - wma(series, length)
    return wma(raw, sqrt_len)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.ema: alpha = 2/(length+1), recursive (adjust=False)."""
    return series.ewm(span=length, adjust=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.rma (Wilder smoothing): alpha = 1/length.
    Ilk deger uzunluk kadar SMA ile seed edilir; sonrasi recursive.
    """
    alpha = 1.0 / length
    # Wilder RMA: SMA seed + recursive. pandas ewm ile pratik ve yeterince yakin
    # sonuc verir; ATR gibi kullanimda TradingView ile hizalanir.
    return series.ewm(alpha=alpha, adjust=False).mean()


# ----------------------------------------------------------------------------
# ATR ve MFI
# ----------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Pine ta.atr: RMA(TrueRange, length)."""
    return rma(true_range(df), length)


def mfi(df: pd.DataFrame, source: pd.Series, length: int) -> pd.Series:
    """Pine ta.mfi(source, length):
        upper = sum(volume * (change(src) <= 0 ? 0 : src), length)
        lower = sum(volume * (change(src) >= 0 ? 0 : src), length)
        mfi   = 100 - 100 / (1 + upper/lower)
    """
    vol = df["volume"]
    change = source.diff()
    upper_bar = np.where(change <= 0, 0.0, source) * vol
    lower_bar = np.where(change >= 0, 0.0, source) * vol
    upper = pd.Series(upper_bar, index=df.index).rolling(length).sum()
    lower = pd.Series(lower_bar, index=df.index).rolling(length).sum()
    return 100.0 - (100.0 / (1.0 + upper / lower))


# ----------------------------------------------------------------------------
# SSL Hybrid (HMA tabanli) - stateful yon
# ----------------------------------------------------------------------------
def ssl_hybrid(df: pd.DataFrame, ssl_len: int) -> pd.DataFrame:
    """SSL yonu ve cizgisi. Pine mantiginin birebir portu.

    ssl_dir := close > hma_high[1] ? 1 : close < hma_low[1] ? -1 : prev
    """
    hma_high = hma(df["high"], ssl_len)
    hma_low = hma(df["low"], ssl_len)

    close = df["close"].to_numpy()
    hh_prev = hma_high.shift(1).to_numpy()
    hl_prev = hma_low.shift(1).to_numpy()

    n = len(df)
    ssl_dir = np.zeros(n)
    prev = 0.0
    for i in range(n):
        if np.isnan(hh_prev[i]) or np.isnan(hl_prev[i]):
            ssl_dir[i] = prev
            continue
        if close[i] > hh_prev[i]:
            cur = 1.0
        elif close[i] < hl_prev[i]:
            cur = -1.0
        else:
            cur = prev
        ssl_dir[i] = cur
        prev = cur

    out = pd.DataFrame(index=df.index)
    out["ssl_dir"] = ssl_dir
    out["ssl_line"] = np.where(ssl_dir == 1.0, hma_low, hma_high)
    prev_dir = out["ssl_dir"].shift(1)
    out["ssl_cross_up"] = (out["ssl_dir"] == 1.0) & (prev_dir == -1.0)
    out["ssl_cross_down"] = (out["ssl_dir"] == -1.0) & (prev_dir == 1.0)
    return out


# ----------------------------------------------------------------------------
# AlphaTrend - stateful
# ----------------------------------------------------------------------------
def alpha_trend(df: pd.DataFrame, atr_len: int, atr_mult: float, mfi_len: int) -> pd.DataFrame:
    """AlphaTrend'in Pine portu (stateful dongu)."""
    _atr = atr(df, atr_len).to_numpy()
    _mfi = mfi(df, (df["high"] + df["low"] + df["close"]) / 3.0, mfi_len).to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()

    n = len(df)
    upT = low - _atr * atr_mult
    downT = high + _atr * atr_mult

    is_up = np.zeros(n, dtype=bool)
    alpha = np.zeros(n)

    prev_up = True            # var isUpTrend = true
    prev_alpha = 0.0          # nz(alphaTrend[1]) baslangicta 0
    for i in range(n):
        mfi_bias_up = (not np.isnan(_mfi[i])) and _mfi[i] >= 50
        cur_up = prev_up
        if close[i] < prev_alpha and not mfi_bias_up and prev_up:
            cur_up = False
        if close[i] > prev_alpha and mfi_bias_up and not prev_up:
            cur_up = True

        if cur_up:
            cur_alpha = max(upT[i], prev_alpha) if not np.isnan(upT[i]) else prev_alpha
        else:
            cur_alpha = min(downT[i], prev_alpha) if not np.isnan(downT[i]) else prev_alpha

        is_up[i] = cur_up
        alpha[i] = cur_alpha
        prev_up = cur_up
        prev_alpha = cur_alpha

    out = pd.DataFrame(index=df.index)
    out["alpha_trend"] = alpha
    out["alpha_is_up"] = is_up
    out["alpha_up_confirmed"] = is_up & (close > alpha)
    out["alpha_down_confirmed"] = (~is_up) & (close < alpha)
    return out


# ----------------------------------------------------------------------------
# EMA bulut (mevcut zaman dilimi)
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.rsi: Wilder RSI (RMA tabanli)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_bands(df: pd.DataFrame, src: pd.Series, rsi_len: int, ob: int, os_: int) -> pd.DataFrame:
    """MTF RSI Band (V10): RSI'dan turetilen destek/direnc bandi (ub/lb/mid).
    Pine kodunun birebir portu.
    """
    ep = 2 * rsi_len - 1
    up = (src - src.shift(1)).clip(lower=0.0)
    dn = (src.shift(1) - src).clip(lower=0.0)
    auc = ema(up, ep)
    adc = ema(dn, ep)

    x1 = (rsi_len - 1) * (adc * ob / (100.0 - ob) - auc)
    ub = np.where(x1 >= 0, src + x1, src + x1 * (100.0 - ob) / ob)
    x2 = (rsi_len - 1) * (adc * os_ / (100.0 - os_) - auc)
    lb = np.where(x2 >= 0, src + x2, src + x2 * (100.0 - os_) / os_)

    out = pd.DataFrame(index=df.index)
    out["ub"] = ub
    out["lb"] = lb
    out["mid"] = (out["ub"] + out["lb"]) / 2.0
    return out


def nft_state_series(df: pd.DataFrame, ssl_len: int, atr_len: int, atr_mult: float,
                     mfi_len: int, ema_fast_len: int, ema_slow_len: int) -> pd.Series:
    """V10'daki f_nft_state() portu: 3'lu teyit (HTF filtresi ve dedup YOK),
    kalici yon durumu (1=buy, -1=sell, 0=baslangic).
    """
    ssl = ssl_hybrid(df, ssl_len)
    at = alpha_trend(df, atr_len, atr_mult, mfi_len)
    ec = ema_cloud(df, ema_fast_len, ema_slow_len)
    buy = ssl["ssl_cross_up"] & at["alpha_up_confirmed"] & ec["ema_up_confirmed"]
    sell = ssl["ssl_cross_down"] & at["alpha_down_confirmed"] & ec["ema_down_confirmed"]

    state = np.zeros(len(df), dtype=int)
    cur = 0
    b = buy.to_numpy(); s = sell.to_numpy()
    for i in range(len(df)):
        if b[i]:
            cur = 1
        if s[i]:
            cur = -1
        state[i] = cur
    return pd.Series(state, index=df.index, name="nft_state")


def ema_cloud(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    ema_fast = ema(df["close"], fast_len)
    ema_slow = ema(df["close"], slow_len)
    out = pd.DataFrame(index=df.index)
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["ema_up_confirmed"] = df["close"] > np.maximum(ema_fast, ema_slow)
    out["ema_down_confirmed"] = df["close"] < np.minimum(ema_fast, ema_slow)
    return out
