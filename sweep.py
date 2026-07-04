#!/usr/bin/env python3
"""Teyit penceresi ve zaman dilimi taramasi — karsilastirmali tablo."""
from __future__ import annotations

import itertools

from src.data import fetch_ohlcv
from src.strategy import StrategyParams, generate_signals
from src.backtest import BacktestConfig, run_backtest
from src.metrics import compute_metrics

# (sembol, tf, gun) — dusuk tf'lerde veri hacmi nedeniyle daha kisa periyot
COMBOS = [
    ("BTC/USDT", "1m", 30),
    ("BTC/USDT", "3m", 60),
    ("BTC/USDT", "5m", 90),
    ("BTC/USDT", "15m", 180),
    ("BTC/USDT", "1h", 365),
    ("ETH/USDT", "3m", 60),
    ("ETH/USDT", "15m", 180),
    ("ETH/USDT", "1h", 365),
    ("SOL/USDT", "3m", 60),
    ("SOL/USDT", "15m", 180),
]
WINDOWS = [1, 3, 5]

hdr = f"{'sembol':<10}{'tf':<5}{'win':<5}{'islem':<7}{'kazan%':<8}{'getiri%':<9}{'PF':<7}{'maxDD%':<8}{'sharpe':<7}"
print(hdr)
print("-" * len(hdr))

for symbol, tf, days in COMBOS:
    df = fetch_ohlcv(symbol, tf, days)
    for w in WINDOWS:
        params = StrategyParams(confirm_window=w)
        sig = generate_signals(df, tf, params)
        res = run_backtest(sig, BacktestConfig())
        m = compute_metrics(res, tf)
        print(f"{symbol:<10}{tf:<5}{w:<5}{m['islem_sayisi']:<7}"
              f"{m['kazanma_orani_%']:<8}{m['toplam_getiri_%']:<9}"
              f"{str(m['profit_factor']):<7}{m['max_drawdown_%']:<8}{m['sharpe']:<7}")
    print()
