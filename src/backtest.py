"""
Event-driven backtest motoru.

Pine indikatorunde OLMAYAN ve bir bot icin sart olan seyleri ekler:
  - Cikis mantigi: stop-loss (ATR tabanli), take-profit (R katsayisi), ters sinyal
  - Pozisyon boyutu: islem basina sabit risk %'si
  - Komisyon + slippage
  - Lookahead yok: sinyal kapanan mumda hesaplanir, giris BIR SONRAKI mum acilisinda

Long ve short (Binance Futures) desteklenir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01      # equity'nin %1'i her islemde riske
    fee_rate: float = 0.0004          # Binance futures taker ~%0.04
    slippage: float = 0.0005          # %0.05 kayma varsayimi
    sl_atr_len: int = 14
    sl_atr_mult: float = 2.0          # stop = giris -/+ 2*ATR
    tp_r_multiple: float = 2.0        # take-profit = 2R (risk:odul 1:2)
    leverage: float = 1.0             # bilgi amacli; risk hesabi notional uzerinden
    allow_short: bool = True
    exit_on_opposite: bool = True     # ters sinyal gelince pozisyonu kapat


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    qty: float
    pnl: float
    pnl_pct: float
    reason: str
    equity_after: float


def run_backtest(df_sig: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """df_sig: strategy.generate_signals ciktisi (buy_signal/sell_signal iceren)."""
    atr = ind.atr(df_sig, cfg.sl_atr_len).to_numpy()
    o = df_sig["open"].to_numpy()
    h = df_sig["high"].to_numpy()
    l = df_sig["low"].to_numpy()
    c = df_sig["close"].to_numpy()
    buy = df_sig["buy_signal"].to_numpy()
    sell = df_sig["sell_signal"].to_numpy()
    idx = df_sig.index

    equity = cfg.initial_equity
    trades: list[Trade] = []
    equity_curve = np.full(len(df_sig), np.nan)

    pos = 0            # 0 flat, 1 long, -1 short
    entry_price = 0.0
    stop = 0.0
    take = 0.0
    qty = 0.0
    entry_time = None

    # sinyal[i] mumun kapanisinda olusur -> i+1 acilisinda islem
    pending = 0  # +1 long, -1 short, i+1'de uygulanacak

    def close_position(exit_price, t, reason):
        nonlocal equity, pos, qty
        gross = (exit_price - entry_price) * qty * pos
        fees = (entry_price + exit_price) * qty * cfg.fee_rate
        pnl = gross - fees
        equity += pnl
        trades.append(Trade(
            side="LONG" if pos == 1 else "SHORT",
            entry_time=entry_time, entry=entry_price,
            exit_time=t, exit=exit_price, qty=qty,
            pnl=pnl, pnl_pct=pnl / cfg.initial_equity * 100,
            reason=reason, equity_after=equity,
        ))
        pos = 0
        qty = 0.0

    for i in range(len(df_sig)):
        # --- 1) bekleyen girisi uygula (bu mumun acilisinda) ---
        if pending != 0 and pos == 0:
            fill = o[i] * (1 + cfg.slippage * pending)   # long'da yukari, short'ta asagi kayma
            a = atr[i]
            if np.isnan(a) or a <= 0:
                pending = 0
            else:
                pos = pending
                entry_price = fill
                entry_time = idx[i]
                if pos == 1:
                    stop = entry_price - cfg.sl_atr_mult * a
                    take = entry_price + cfg.sl_atr_mult * a * cfg.tp_r_multiple
                else:
                    stop = entry_price + cfg.sl_atr_mult * a
                    take = entry_price - cfg.sl_atr_mult * a * cfg.tp_r_multiple
                risk_per_unit = abs(entry_price - stop)
                risk_cash = equity * cfg.risk_per_trade
                qty = risk_cash / risk_per_unit if risk_per_unit > 0 else 0.0
                pending = 0

        # --- 2) acik pozisyonda SL/TP kontrolu (mum ici) ---
        if pos != 0:
            if pos == 1:
                hit_sl = l[i] <= stop
                hit_tp = h[i] >= take
            else:
                hit_sl = h[i] >= stop
                hit_tp = l[i] <= take
            # Kotu senaryo varsayimi: ayni mumda ikisi de olduysa once STOP
            if hit_sl:
                close_position(stop, idx[i], "SL")
            elif hit_tp:
                close_position(take, idx[i], "TP")

        # --- 3) sinyal degerlendir (mum kapanisi) ---
        if pos != 0 and cfg.exit_on_opposite:
            if pos == 1 and sell[i]:
                close_position(c[i], idx[i], "OPP")
            elif pos == -1 and buy[i]:
                close_position(c[i], idx[i], "OPP")

        if pos == 0:
            if buy[i]:
                pending = 1
            elif sell[i] and cfg.allow_short:
                pending = -1

        equity_curve[i] = equity + _unrealized(pos, entry_price, c[i], qty)

    ec = pd.Series(equity_curve, index=idx, name="equity")
    return {"trades": trades, "equity_curve": ec, "final_equity": equity, "config": cfg}


def _unrealized(pos, entry, price, qty):
    if pos == 0:
        return 0.0
    return (price - entry) * qty * pos
