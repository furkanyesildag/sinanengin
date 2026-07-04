"""
V10 stratejisinin ("SE NFT + MTF RSI Band") birebir Python simulatoru.

V10 bir 'strategy' (indicator degil): giris + cikis + SL/TP + state makinesi
Pine icinde tanimli. Bu yuzden strateji ve backtest tek bir bar-bar simulasyonda
birlestirilmistir.

MODELLEME NOTLARI (gerceklige yakinlik icin):
  - Sinyal bar i kapanisinda belirlenir; GIRIS bar i+1 acilisinda dolar (+slippage).
    (lookahead yok)
  - SL: gercek stop emri gibi kilitli seviyeden dolar (aleyhte slippage ile).
  - Band-TP: limit gibi o anki band seviyesinden dolar. %-TP: kilitli seviyeden.
  - Ayni barda hem SL hem TP degerse: KOTUMSER varsayim -> once SL.
  - Pozisyon boyutu: Pine'daki gibi ozkaynagin %100'u (1x), her islemde tek pozisyon.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategy import htf_ema_filter, StrategyParams


@dataclass
class V10Params:
    # NFT cekirdek
    ssl_len: int = 60
    at_atr_mult: float = 1.0
    at_atr_len: int = 14
    at_mfi_len: int = 14
    ema_fast_len: int = 50
    ema_slow_len: int = 200
    use_htf_filter: bool = True
    htf_timeframe: str | None = None

    # RSI Band
    ob_level: int = 70
    os_level: int = 30
    rsi_len: int = 14
    band_tf: str = ""       # "" = mevcut TF; "15m" = TradingView'deki gibi 15dk banttan

    # Giris modu: "touch" (Banda Degince) | "signal_then_touch" (Sinyalden Sonra Banda)
    entry_mode: str = "touch"

    # SL/TP
    atr_sl_len: int = 14
    atr_sl_buffer: float = 0.5
    use_pct_sl: bool = False
    pct_sl: float = 2.0
    use_pct_tp: bool = False
    pct_tp: float = 4.0
    band_tp_active: bool = True   # V10 orijinal: band temasi TP'si hep aktif.
                                  # False -> sadece %/SL cikisi (saf 1:R profili)

    # Guclu hareket filtresi
    use_strength_filter: bool = True
    strength_metric: str = "both"     # "both" | "rsi" | "pen"
    deep_os_level: int = 25
    deep_ob_level: int = 75
    pen_atr_thresh: float = 0.6
    confirm_timeout: int = 6
    require_rsi_turn: bool = True
    require_candle: bool = True

    # NFT yon filtresi (referans sembol)
    use_nft_dir_filter: bool = True
    nft_ref_symbol: str = "ETH/USDT"
    nft_dir_tf: str = "3m"


@dataclass
class V10Costs:
    initial_equity: float = 1000.0
    fee_rate: float = 0.0005      # Pine commission 0.05%
    slippage: float = 0.0003      # her yonde aleyhte kayma
    equity_pct: float = 1.0       # "full_equity" modunda islem basina ozkaynak orani
    # Pozisyon boyutu modu:
    #   "full_equity" -> Pine ayari (%100 equity, 1x)
    #   "risk_pct"    -> SL mesafesine gore islem basina sabit risk (ONERILEN)
    sizing_mode: str = "full_equity"
    risk_per_trade: float = 0.01  # risk_pct modunda: her islemde equity'nin %'si
    max_leverage: float = 10.0    # risk_pct modunda notional/equity ust siniri


def _rsi_bands_mtf(df: pd.DataFrame, base_tf: str, p: V10Params) -> pd.DataFrame:
    """RSI band'ini p.band_tf zaman diliminde hesaplayip base index'e hizalar.
    band_tf bos veya base ile ayni ise dogrudan base uzerinde hesaplar.
    TradingView request.security(..., lookahead_off) davranisi: son KAPANMIS
    band mumunun degeri kullanilir (merge_asof backward)."""
    from .strategy import _resample_ohlcv, _TF_TO_PANDAS
    bt = p.band_tf
    if not bt or bt == base_tf:
        return ind.rsi_bands(df, df["close"], p.rsi_len, p.ob_level, p.os_level)
    htf = _resample_ohlcv(df, bt)
    b = ind.rsi_bands(htf, htf["close"], p.rsi_len, p.ob_level, p.os_level)
    b_df = b.reset_index(); b_df.columns = ["band_time", "ub", "lb", "mid"]
    base = df.reset_index().rename(columns={df.index.name or "index": "time"})
    base = base.rename(columns={base.columns[0]: "time"})
    merged = pd.merge_asof(base.sort_values("time"), b_df.sort_values("band_time"),
                           left_on="time", right_on="band_time", direction="backward")
    out = pd.DataFrame(index=df.index)
    out["ub"] = merged["ub"].to_numpy()
    out["lb"] = merged["lb"].to_numpy()
    out["mid"] = merged["mid"].to_numpy()
    return out


def _build_features(df: pd.DataFrame, base_tf: str, p: V10Params,
                    nft_ref_df: pd.DataFrame | None) -> pd.DataFrame:
    ssl = ind.ssl_hybrid(df, p.ssl_len)
    at = ind.alpha_trend(df, p.at_atr_len, p.at_atr_mult, p.at_mfi_len)
    ec = ind.ema_cloud(df, p.ema_fast_len, p.ema_slow_len)
    bands = _rsi_bands_mtf(df, base_tf, p)
    rsi_raw = ind.rsi(df["close"], p.rsi_len)
    atr_now = ind.atr(df, p.atr_sl_len)

    sp = StrategyParams(ema_fast_len=p.ema_fast_len, ema_slow_len=p.ema_slow_len,
                        use_htf_filter=p.use_htf_filter, htf_timeframe=p.htf_timeframe)
    htf = htf_ema_filter(df, base_tf, sp)

    f = df.copy()
    f = f.join([ssl, at, ec, bands, htf])
    f["rsi_raw"] = rsi_raw
    f["atr_now"] = atr_now

    # NFT yon durumu (referans sembol, secilen TF) -> base index'e hizala + [1]
    if p.use_nft_dir_filter and nft_ref_df is not None and len(nft_ref_df):
        st = ind.nft_state_series(nft_ref_df, p.ssl_len, p.at_atr_len, p.at_atr_mult,
                                  p.at_mfi_len, p.ema_fast_len, p.ema_slow_len)
        st = st.shift(1)  # Pine nz(nft_state_raw[1])
        st_df = st.reset_index()
        st_df.columns = ["ref_time", "nft_state"]
        base = f.reset_index()
        base = base.rename(columns={base.columns[0]: "time"})
        merged = pd.merge_asof(base.sort_values("time"), st_df.sort_values("ref_time"),
                               left_on="time", right_on="ref_time", direction="backward")
        f["nft_state"] = merged["nft_state"].fillna(0).to_numpy()
    else:
        f["nft_state"] = 0
    return f


def derived_frame(df: pd.DataFrame, base_tf: str, p: V10Params,
                  nft_ref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canli motor icin bar-basi turetilmis kolonlar (simulate_v10 ile ayni mantik).
    Boylece canli bot ile backtest ayni hesaplamayi kullanir (davranis birligi)."""
    f = _build_features(df, base_tf, p, nft_ref_df)
    o = f["open"].to_numpy(); h = f["high"].to_numpy()
    l = f["low"].to_numpy(); c = f["close"].to_numpy()
    ub = f["ub"].to_numpy(); lb = f["lb"].to_numpy()
    atr_now = f["atr_now"].to_numpy(); rsi_raw = f["rsi_raw"].to_numpy()
    scu = f["ssl_cross_up"].to_numpy(); scd = f["ssl_cross_down"].to_numpy()
    a_up = f["alpha_up_confirmed"].to_numpy(); a_dn = f["alpha_down_confirmed"].to_numpy()
    e_up = f["ema_up_confirmed"].to_numpy(); e_dn = f["ema_down_confirmed"].to_numpy()
    htf_bull = f["htf_bullish"].fillna(False).to_numpy()
    htf_bear = f["htf_bearish"].fillna(False).to_numpy()

    raw_buy = scu & a_up & e_up & (True if not p.use_htf_filter else htf_bull)
    raw_sell = scd & a_dn & e_dn & (True if not p.use_htf_filter else htf_bear)
    touched_lb = (l <= lb) & ~np.isnan(lb)
    touched_ub = (h >= ub) & ~np.isnan(ub)
    with np.errstate(invalid="ignore", divide="ignore"):
        pen_long = np.nan_to_num(np.where(atr_now > 0, (lb - l) / atr_now, 0.0))
        pen_short = np.nan_to_num(np.where(atr_now > 0, (h - ub) / atr_now, 0.0))
    rsi_sd = rsi_raw <= p.deep_os_level; rsi_su = rsi_raw >= p.deep_ob_level
    pen_sd = pen_long >= p.pen_atr_thresh; pen_su = pen_short >= p.pen_atr_thresh
    if p.strength_metric == "rsi":
        strong_down, strong_up = rsi_sd, rsi_su
    elif p.strength_metric == "pen":
        strong_down, strong_up = pen_sd, pen_su
    else:
        strong_down, strong_up = (rsi_sd | pen_sd), (rsi_su | pen_su)
    green = c > o; red = c < o
    rsi_up = rsi_raw > np.roll(rsi_raw, 1); rsi_dn = rsi_raw < np.roll(rsi_raw, 1)
    reclaim_long = (~np.isnan(lb)) & (c > lb) & (~p.require_rsi_turn | rsi_up) & (~p.require_candle | green)
    reclaim_short = (~np.isnan(ub)) & (c < ub) & (~p.require_rsi_turn | rsi_dn) & (~p.require_candle | red)

    d = pd.DataFrame(index=f.index)
    d["open"], d["high"], d["low"], d["close"] = o, h, l, c
    d["lb"], d["ub"], d["atr_now"] = lb, ub, atr_now
    d["raw_buy"], d["raw_sell"] = raw_buy, raw_sell
    d["touched_lb"], d["touched_ub"] = touched_lb, touched_ub
    d["strong_down"], d["strong_up"] = strong_down, strong_up
    d["reclaim_long"], d["reclaim_short"] = reclaim_long, reclaim_short
    d["nft_state"] = f["nft_state"].to_numpy()
    return d


def simulate_v10(df: pd.DataFrame, base_tf: str, p: V10Params, costs: V10Costs,
                 nft_ref_df: pd.DataFrame | None = None) -> dict:
    f = _build_features(df, base_tf, p, nft_ref_df)

    o = f["open"].to_numpy(); h = f["high"].to_numpy()
    l = f["low"].to_numpy(); c = f["close"].to_numpy()
    op = f["open"].to_numpy()
    ub = f["ub"].to_numpy(); lb = f["lb"].to_numpy()
    atr_now = f["atr_now"].to_numpy()
    rsi_raw = f["rsi_raw"].to_numpy()
    scu = f["ssl_cross_up"].to_numpy(); scd = f["ssl_cross_down"].to_numpy()
    a_up = f["alpha_up_confirmed"].to_numpy(); a_dn = f["alpha_down_confirmed"].to_numpy()
    e_up = f["ema_up_confirmed"].to_numpy(); e_dn = f["ema_down_confirmed"].to_numpy()
    htf_bull = f["htf_bullish"].fillna(False).to_numpy()
    htf_bear = f["htf_bearish"].fillna(False).to_numpy()
    nft_state = f["nft_state"].to_numpy()
    idx = f.index
    n = len(f)

    raw_buy = scu & a_up & e_up & (True if not p.use_htf_filter else htf_bull)
    raw_sell = scd & a_dn & e_dn & (True if not p.use_htf_filter else htf_bear)

    # band temaslari
    touched_lb = (l <= lb) & ~np.isnan(lb)
    touched_ub = (h >= ub) & ~np.isnan(ub)

    # guc olcumu
    with np.errstate(invalid="ignore", divide="ignore"):
        pen_long = np.where((atr_now > 0), (lb - l) / atr_now, 0.0)
        pen_short = np.where((atr_now > 0), (h - ub) / atr_now, 0.0)
    pen_long = np.nan_to_num(pen_long)
    pen_short = np.nan_to_num(pen_short)
    rsi_sd = rsi_raw <= p.deep_os_level
    rsi_su = rsi_raw >= p.deep_ob_level
    pen_sd = pen_long >= p.pen_atr_thresh
    pen_su = pen_short >= p.pen_atr_thresh
    if p.strength_metric == "rsi":
        strong_down, strong_up = rsi_sd, rsi_su
    elif p.strength_metric == "pen":
        strong_down, strong_up = pen_sd, pen_su
    else:
        strong_down, strong_up = (rsi_sd | pen_sd), (rsi_su | pen_su)

    green = c > o
    red = c < o
    rsi_up = rsi_raw > np.roll(rsi_raw, 1)
    rsi_dn = rsi_raw < np.roll(rsi_raw, 1)
    reclaim_long = (~np.isnan(lb)) & (c > lb) & (~p.require_rsi_turn | rsi_up) & (~p.require_candle | green)
    reclaim_short = (~np.isnan(ub)) & (c < ub) & (~p.require_rsi_turn | rsi_dn) & (~p.require_candle | red)

    # --- state ---
    equity = costs.initial_equity
    equity_curve = np.full(n, np.nan)
    trades = []

    tradeState = 0
    waitLB = waitUB = False
    pendingLong = pendingShort = False
    pendingLongBar = pendingShortBar = -1

    position = 0
    entry_fill = np.nan
    entry_bar = -1
    entry_notional = 0.0
    qty = 0.0
    locked_sl = np.nan
    locked_tp = np.nan
    entry_time = None

    pending_fill = 0            # +1/-1: bir sonraki barda dol
    pend_sl = np.nan; pend_tp = np.nan; pend_signal_time = None

    def _open(i, side, sl, tp, sig_time):
        nonlocal position, entry_fill, entry_bar, entry_notional, qty, locked_sl, locked_tp, entry_time
        fill = op[i] * (1 + costs.slippage * side)
        position = side
        entry_fill = fill
        entry_bar = i
        if costs.sizing_mode == "risk_pct" and not np.isnan(sl) and abs(fill - sl) > 0:
            # SL vurulunca tam olarak equity*risk_per_trade kaybedilecek sekilde boyutla
            risk_cash = equity * costs.risk_per_trade
            stop_dist_frac = abs(fill - sl) / fill
            entry_notional = min(risk_cash / stop_dist_frac, equity * costs.max_leverage)
        else:
            entry_notional = equity * costs.equity_pct
        qty = entry_notional / fill
        locked_sl = sl
        locked_tp = tp
        entry_time = idx[i]

    def _close(i, exit_price, reason):
        nonlocal equity, position, qty, tradeState, locked_sl, locked_tp
        gross = (exit_price - entry_fill) * qty * position
        exit_notional = qty * exit_price
        fees = entry_notional * costs.fee_rate + exit_notional * costs.fee_rate
        pnl = gross - fees
        equity += pnl
        trades.append(dict(
            side="LONG" if position == 1 else "SHORT",
            entry_time=entry_time, entry=entry_fill,
            exit_time=idx[i], exit=exit_price, qty=qty,
            pnl=pnl, pnl_pct=pnl / entry_notional * 100 if entry_notional else 0.0,
            reason=reason, equity_after=equity,
        ))
        position = 0; qty = 0.0; tradeState = 0
        locked_sl = np.nan; locked_tp = np.nan

    for i in range(n):
        # 0) bekleyen girisi bu barin acilisinda doldur
        if pending_fill != 0 and position == 0:
            if not np.isnan(atr_now[i]):
                _open(i, pending_fill, pend_sl, pend_tp, pend_signal_time)
            pending_fill = 0

        # 1) acik pozisyonda cikis kontrolu (giris barindan itibaren)
        if position != 0 and i >= entry_bar:
            if position == 1:
                hit_sl = (not np.isnan(locked_sl)) and l[i] <= locked_sl
                band_tp = p.band_tp_active and touched_ub[i]
                pct_tp = p.use_pct_tp and (not np.isnan(locked_tp)) and h[i] >= locked_tp
                if hit_sl:
                    _close(i, locked_sl * (1 - costs.slippage), "SL")
                elif band_tp:
                    _close(i, ub[i], "TP")
                elif pct_tp:
                    _close(i, locked_tp, "TP%")
            else:
                hit_sl = (not np.isnan(locked_sl)) and h[i] >= locked_sl
                band_tp = p.band_tp_active and touched_lb[i]
                pct_tp = p.use_pct_tp and (not np.isnan(locked_tp)) and l[i] <= locked_tp
                if hit_sl:
                    _close(i, locked_sl * (1 + costs.slippage), "SL")
                elif band_tp:
                    _close(i, lb[i], "TP")
                elif pct_tp:
                    _close(i, locked_tp, "TP%")

        # 2) giris mantigi (yalnizca bostayken)
        do_long = do_short = False
        base_long = base_short = False
        if tradeState == 0 and position == 0 and pending_fill == 0:
            if p.entry_mode == "touch":
                if touched_lb[i] and not waitLB:
                    waitLB = True
                if (waitLB or touched_lb[i]) and raw_buy[i]:
                    base_long = True
                if touched_ub[i] and not waitUB:
                    waitUB = True
                if (waitUB or touched_ub[i]) and raw_sell[i]:
                    base_short = True
            else:  # signal_then_touch
                if raw_buy[i]:
                    waitLB = True
                if waitLB and touched_lb[i]:
                    base_long = True
                if raw_sell[i]:
                    waitUB = True
                if waitUB and touched_ub[i]:
                    base_short = True

            # guc kapisi
            if base_long:
                waitLB = False
                if (not p.use_strength_filter) or (not strong_down[i]):
                    do_long = True
                else:
                    pendingLong = True; pendingLongBar = i
            if base_short:
                waitUB = False
                if (not p.use_strength_filter) or (not strong_up[i]):
                    do_short = True
                else:
                    pendingShort = True; pendingShortBar = i

            # bekleyen teyit (reclaim / timeout)
            if pendingLong and not do_long:
                if reclaim_long[i]:
                    do_long = True; pendingLong = False
                elif i - pendingLongBar >= p.confirm_timeout:
                    pendingLong = False
            if pendingShort and not do_short:
                if reclaim_short[i]:
                    do_short = True; pendingShort = False
                elif i - pendingShortBar >= p.confirm_timeout:
                    pendingShort = False

            # NFT yon vetosu
            if p.use_nft_dir_filter:
                if do_long and nft_state[i] == -1:
                    do_long = False
                if do_short and nft_state[i] == 1:
                    do_short = False

            # 3) giris: SL/TP kilitle, bir sonraki bara fill planla
            if do_long:
                waitLB = waitUB = False; pendingLong = pendingShort = False
                tradeState = 1
                pend_sl = (c[i] * (1 - p.pct_sl / 100.0)) if p.use_pct_sl else (lb[i] - atr_now[i] * p.atr_sl_buffer)
                pend_tp = (c[i] * (1 + p.pct_tp / 100.0)) if p.use_pct_tp else np.nan
                pending_fill = 1; pend_signal_time = idx[i]
            elif do_short:
                waitLB = waitUB = False; pendingLong = pendingShort = False
                tradeState = -1
                pend_sl = (c[i] * (1 + p.pct_sl / 100.0)) if p.use_pct_sl else (ub[i] + atr_now[i] * p.atr_sl_buffer)
                pend_tp = (c[i] * (1 - p.pct_tp / 100.0)) if p.use_pct_tp else np.nan
                pending_fill = -1; pend_signal_time = idx[i]

        # mark-to-market equity
        unreal = 0.0 if position == 0 else (c[i] - entry_fill) * qty * position
        equity_curve[i] = equity + unreal

    ec = pd.Series(equity_curve, index=idx, name="equity")
    return {"trades": trades, "equity_curve": ec, "final_equity": equity, "features": f}
