#!/usr/bin/env python3
"""
V10 tek calistirma — dogrulanmis "kazanan" config varsayilan.

KAZANAN CONFIG (60g + OOS train/test ile 5 sembolun 4'unde pozitif):
  - entry_mode = signal_then_touch (NFT sinyali -> banda geri cekilmede gir)
  - SL/TP      = saf % 2/4 (1:2 risk/odul), band-TP KAPALI
  - guc filtresi ON, HTF filtresi ON, NFT yon filtresi OFF

Ornek:
  python run_v10.py --symbol SOL/USDT --tf 3m --days 60 --save
"""
from __future__ import annotations

import argparse

from src.data import fetch_ohlcv
from src.strategy_v10 import V10Params, V10Costs, simulate_v10
from src.metrics import compute_metrics, print_report, trades_to_df


def best_params() -> V10Params:
    # 50 coin SL/TP optimizasyonu + kullanici TV ayarlari ile kilitli config.
    return V10Params(
        entry_mode="signal_then_touch",
        band_tf="15m",
        use_strength_filter=True, strength_metric="both",
        deep_os_level=25, deep_ob_level=75,
        use_htf_filter=True,
        use_nft_dir_filter=True, nft_ref_symbol="ETH/USDT", nft_dir_tf="3m",
        use_pct_sl=True, use_pct_tp=True, band_tp_active=False,
        pct_sl=2.0, pct_tp=3.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOL/USDT")
    ap.add_argument("--tf", default="3m")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--ref", default="ETH/USDT")
    ap.add_argument("--ref-tf", default="3m")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df = fetch_ohlcv(args.symbol, args.tf, args.days)
    ref = fetch_ohlcv(args.ref, args.ref_tf, args.days)
    print(f"{args.symbol} {args.tf}: {len(df)} mum | ref {args.ref} {args.ref_tf}")

    p = best_params()
    res = simulate_v10(df, args.tf, p, V10Costs(), nft_ref_df=ref)
    m = compute_metrics(res, args.tf)
    print_report(m, args.symbol, args.tf, args.days)

    if args.save:
        safe = args.symbol.replace("/", "")
        trades_to_df(res).to_csv(f"results/{safe}_{args.tf}_v10_trades.csv", index=False)
        res["equity_curve"].to_csv(f"results/{safe}_{args.tf}_v10_equity.csv")
        print(f"  Kaydedildi: results/{safe}_{args.tf}_v10_*.csv")


if __name__ == "__main__":
    main()
