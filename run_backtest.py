#!/usr/bin/env python3
"""
Sinan Engin NFT Sistemi - Backtest calistirici.

Ornek:
  python run_backtest.py --symbol BTC/USDT --tf 1h --days 365
  python run_backtest.py --symbol ETH/USDT --tf 4h --days 730 --risk 0.02 --tp-r 3
"""
from __future__ import annotations

import argparse

from src.data import fetch_ohlcv
from src.strategy import StrategyParams, generate_signals
from src.backtest import BacktestConfig, run_backtest
from src.metrics import compute_metrics, print_report, trades_to_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--tf", default="1h")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--risk", type=float, default=0.01, help="islem basina risk (0.01 = %1)")
    p.add_argument("--sl-atr", type=float, default=2.0, help="stop = giris -/+ N*ATR")
    p.add_argument("--tp-r", type=float, default=2.0, help="take-profit R katsayisi")
    p.add_argument("--confirm-window", type=int, default=1, help="SSL sonrasi teyit penceresi (mum)")
    p.add_argument("--no-htf", action="store_true", help="HTF filtresini kapat")
    p.add_argument("--no-short", action="store_true", help="sadece long")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--save", action="store_true", help="islemleri results/ altina yaz")
    args = p.parse_args()

    print(f"Veri cekiliyor: {args.symbol} {args.tf} ({args.days} gun)...")
    df = fetch_ohlcv(args.symbol, args.tf, args.days, cache=not args.no_cache)
    print(f"  {len(df)} mum yuklendi ({df.index[0]} -> {df.index[-1]})")

    params = StrategyParams(use_htf_filter=not args.no_htf, confirm_window=args.confirm_window)
    df_sig = generate_signals(df, args.tf, params)
    n_buy = int(df_sig["buy_signal"].sum())
    n_sell = int(df_sig["sell_signal"].sum())
    print(f"  Sinyaller: {n_buy} BUY, {n_sell} SELL")

    cfg = BacktestConfig(
        risk_per_trade=args.risk,
        sl_atr_mult=args.sl_atr,
        tp_r_multiple=args.tp_r,
        allow_short=not args.no_short,
    )
    result = run_backtest(df_sig, cfg)
    metrics = compute_metrics(result, args.tf)
    print_report(metrics, args.symbol, args.tf, args.days)

    if args.save:
        safe = args.symbol.replace("/", "")
        tdf = trades_to_df(result)
        out = f"results/{safe}_{args.tf}_{args.days}d_trades.csv"
        tdf.to_csv(out, index=False)
        result["equity_curve"].to_csv(f"results/{safe}_{args.tf}_{args.days}d_equity.csv")
        print(f"  Kaydedildi: {out}")


if __name__ == "__main__":
    main()
