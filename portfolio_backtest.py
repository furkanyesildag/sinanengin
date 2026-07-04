#!/usr/bin/env python3
"""
50 coin PORTFOY backtest'i — tek hesap, zaman-sirali, paylasimli %1 risk.

Paper bot'un canli davranisiyla ayni: tum coin'lerin 3m mumlari ayni anda kapanir,
ortak equity havuzundan %1 risk ile pozisyon acilir. Es zamanli pozisyon sayisi
sinirli (max_concurrent) -> toplam maruziyet kontrollu.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data import fetch_ohlcv
from src.strategy_v10 import V10Params, derived_frame
from src.live_engine import V10LiveEngine
from src.broker import PaperBroker
from run_v10 import best_params


def portfolio_backtest(coins, ref, days=60, equity=1000.0, risk=0.01,
                       max_concurrent=10, warmup_frac=0.15):
    p = best_params()
    # her coin icin turetilmis cerceve
    frames = {}
    for sym in coins:
        df = fetch_ohlcv(sym, "3m", days)
        frames[sym] = derived_frame(df, "3m", p, nft_ref_df=ref)

    # ortak zaman index'i (kesisim)
    common = None
    for d in frames.values():
        idx = d.index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    n = len(common)
    warm = int(n * warmup_frac)

    # numpy'a cevir (hiz)
    cols = ["open", "high", "low", "close", "lb", "ub", "atr_now", "raw_buy", "raw_sell",
            "touched_lb", "touched_ub", "strong_down", "strong_up",
            "reclaim_long", "reclaim_short", "nft_state"]
    arr = {sym: {c: frames[sym].reindex(common)[c].to_numpy() for c in cols} for sym in coins}

    broker = PaperBroker(initial_equity=equity, risk_per_trade=risk)
    engines = {sym: V10LiveEngine(p) for sym in coins}
    eq_curve = np.empty(n); peak = equity; skips = 0

    for i in range(n):
        ts = common[i]
        warmup = i < warm
        for sym in coins:
            a = arr[sym]
            row = {c: a[c][i] for c in cols}
            if np.isnan(row["close"]):
                continue
            eng = engines[sym]
            # cikis
            if not warmup:
                tr = broker.on_bar(sym, row["high"], row["low"], row["close"], ts)
                if tr:
                    eng.notify_closed()
            else:
                broker.mark(sym, row["close"])
            # giris
            act = eng.step(row, warmup=warmup)
            if (not warmup) and act["action"] != "none" and sym not in broker.positions:
                if len(broker.positions) >= max_concurrent:
                    skips += 1
                else:
                    side = 1 if act["action"] == "long" else -1
                    pos = broker.market_open(sym, side, row["close"], act["sl"], act["tp"], ts)
                    if pos:
                        eng.notify_entered(side)
        eq_curve[i] = broker.equity_now()
        peak = max(peak, eq_curve[i])

    return broker, pd.Series(eq_curve, index=common), skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--max-concurrent", type=int, default=10)
    args = ap.parse_args()

    coins = json.load(open("results/universe.json"))
    ref = fetch_ohlcv("ETH/USDT", "3m", args.days)
    print(f"Portfoy: {len(coins)} coin | {args.days}g | %{args.risk*100:.0f} risk | "
          f"max {args.max_concurrent} es zamanli pozisyon\n")

    broker, ec, skips = portfolio_backtest(coins, ref, args.days, args.equity,
                                           args.risk, args.max_concurrent)
    tr = broker.trades
    wins = [t for t in tr if t["pnl"] > 0]
    roll_max = ec.cummax(); dd = ((ec - roll_max) / roll_max).min() * 100

    print("=== 50 COIN PORTFOY SONUCU ===")
    print(f"  Baslangic       : ${args.equity:.0f}")
    print(f"  Son equity      : ${broker.equity:.2f}  ({(broker.equity/args.equity-1)*100:+.2f}%)")
    print(f"  Islem sayisi    : {len(tr)}")
    print(f"  Kazanma orani   : {len(wins)/max(1,len(tr))*100:.1f}%")
    print(f"  Max drawdown    : {dd:.2f}%")
    print(f"  Atlanan sinyal  : {skips} (max es zamanli doluydu)")
    # sembol bazinda kar/zarar dagilimi
    from collections import defaultdict
    bys = defaultdict(float)
    for t in tr:
        bys[t["symbol"]] += t["pnl"]
    top = sorted(bys.items(), key=lambda x: -x[1])
    print("\n  En cok kazandiran 5 :", ", ".join(f"{s.replace('/USDT','')} {v:+.0f}" for s, v in top[:5]))
    print("  En cok kaybettiren 5:", ", ".join(f"{s.replace('/USDT','')} {v:+.0f}" for s, v in top[-5:]))
    ec.to_csv("results/portfolio_equity.csv")
    json.dump([{k: t[k] for k in ("symbol","side","reason","pnl","pnl_pct","exit_time")} for t in tr],
              open("results/portfolio_trades.json", "w"))
    print("\n  Kaydedildi: results/portfolio_equity.csv, portfolio_trades.json")


if __name__ == "__main__":
    main()
