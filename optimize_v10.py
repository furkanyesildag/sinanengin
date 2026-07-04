#!/usr/bin/env python3
"""
V10 parametre optimizasyonu — train/test (out-of-sample) ayrimiyla.

Overfit'e karsi: parametreleri verinin ILK %70'inde (train) tarar, en iyileri
verinin SON %30'unda (test / gormedigi veri) dogrular. Sadece HEM train HEM test'te
iyi olan ayarlar guvenilirdir.

Kullanim:
  python optimize_v10.py --symbol BTC/USDT --tf 3m --days 60
  python optimize_v10.py --symbol BTC/USDT --tf 1m --days 30
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from src.data import fetch_ohlcv
from src.strategy_v10 import V10Params, V10Costs, simulate_v10
from src.metrics import compute_metrics


# --- taranacak parametre uzayi ---
ENTRY_MODES = ["touch", "signal_then_touch"]
SLTP_SCHEMES = [
    {"kind": "band", "atr_sl_buffer": 0.3},
    {"kind": "band", "atr_sl_buffer": 0.6},
    {"kind": "band", "atr_sl_buffer": 1.0},
    {"kind": "pct", "pct_sl": 1.0, "pct_tp": 1.5},
    {"kind": "pct", "pct_sl": 1.0, "pct_tp": 2.0},
    {"kind": "pct", "pct_sl": 1.5, "pct_tp": 3.0},
    {"kind": "pct", "pct_sl": 2.0, "pct_tp": 4.0},
]
STRENGTH = [True, False]
NFT = [True, False]
HTF = [True, False]

MIN_TRADES = 20   # istatistiksel anlam icin train'de en az bu kadar islem


def make_params(entry_mode, sltp, strength, nft, htf) -> V10Params:
    kw = dict(entry_mode=entry_mode, use_strength_filter=strength,
              use_nft_dir_filter=nft, use_htf_filter=htf)
    if sltp["kind"] == "band":
        kw.update(use_pct_sl=False, use_pct_tp=False, atr_sl_buffer=sltp["atr_sl_buffer"])
    else:
        kw.update(use_pct_sl=True, use_pct_tp=True, pct_sl=sltp["pct_sl"], pct_tp=sltp["pct_tp"])
    return V10Params(**kw)


def score(m: dict) -> float:
    """Train siralamasi: min islem sarti + getiri/drawdown dengesi."""
    if m["islem_sayisi"] < MIN_TRADES:
        return -1e9
    dd = abs(m["max_drawdown_%"]) or 1.0
    return m["toplam_getiri_%"] / dd   # basit risk-ayarli skor (Calmar benzeri)


def label(entry_mode, sltp, strength, nft, htf) -> str:
    s = "band%.1f" % sltp["atr_sl_buffer"] if sltp["kind"] == "band" else "pct%.1f/%.1f" % (sltp["pct_sl"], sltp["pct_tp"])
    em = "touch" if entry_mode == "touch" else "sig>touch"
    return f"{em:<10} {s:<11} str={int(strength)} nft={int(nft)} htf={int(htf)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--tf", default="3m")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--ref", default="ETH/USDT")
    ap.add_argument("--ref-tf", default="3m")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    print(f"Veri: {args.symbol} {args.tf} ({args.days}g) | NFT ref: {args.ref} {args.ref_tf}")
    df = fetch_ohlcv(args.symbol, args.tf, args.days)
    ref = fetch_ohlcv(args.ref, args.ref_tf, args.days)
    split = int(len(df) * 0.70)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"  train: {len(train)} mum ({train.index[0]} -> {train.index[-1]})")
    print(f"  test : {len(test)} mum ({test.index[0]} -> {test.index[-1]})")

    costs = V10Costs()
    combos = list(itertools.product(ENTRY_MODES, SLTP_SCHEMES, STRENGTH, NFT, HTF))
    print(f"  {len(combos)} kombinasyon taraniyor...\n")

    rows = []
    for em, sltp, st, nft, htf in combos:
        p = make_params(em, sltp, st, nft, htf)
        r_tr = simulate_v10(train, args.tf, p, costs, nft_ref_df=ref)
        m_tr = compute_metrics(r_tr, args.tf)
        rows.append((score(m_tr), label(em, sltp, st, nft, htf), m_tr, p))

    rows.sort(key=lambda x: x[0], reverse=True)

    hdr = f"{'#':<3}{'ayar':<42}{'islem':<7}{'kazan%':<8}{'getiri%':<9}{'PF':<6}{'DD%':<7}{'skor':<7}"
    print("=== TRAIN'DE EN IYI " + str(args.top) + " ===")
    print(hdr); print("-" * len(hdr))
    top = rows[:args.top]
    for i, (sc, lab, m, p) in enumerate(top, 1):
        print(f"{i:<3}{lab:<42}{m['islem_sayisi']:<7}{m['kazanma_orani_%']:<8}"
              f"{m['toplam_getiri_%']:<9}{str(m['profit_factor']):<6}{m['max_drawdown_%']:<7}{round(sc,2):<7}")

    print("\n=== AYNI AYARLARIN TEST (OUT-OF-SAMPLE) SONUCU ===")
    print(hdr); print("-" * len(hdr))
    for i, (sc, lab, m, p) in enumerate(top, 1):
        r_te = simulate_v10(test, args.tf, p, costs, nft_ref_df=ref)
        m_te = compute_metrics(r_te, args.tf)
        print(f"{i:<3}{lab:<42}{m_te['islem_sayisi']:<7}{m_te['kazanma_orani_%']:<8}"
              f"{m_te['toplam_getiri_%']:<9}{str(m_te['profit_factor']):<6}{m_te['max_drawdown_%']:<7}{'':<7}")
    print("\nNOT: Train'de iyi olup test'te cokenler OVERFIT'tir. Ikisinde de pozitif olan aranir.")


if __name__ == "__main__":
    main()
