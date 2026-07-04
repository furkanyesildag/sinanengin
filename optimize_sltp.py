#!/usr/bin/env python3
"""
SL/TP optimizasyonu — 50 coin evreni uzerinde.

Kullanici ayarlari sabit: band 15m, giris sinyal->temas, guc filtresi acik,
HTF acik, NFT yon ETH 3m acik. Sadece SL/TP semasi taranir.

Saglamlik olcutu: KAC coin'de karli (pct_profitable) + risk-ayarli getiri (Calmar).
Tek coin'e uyduran degil, evren genelinde tutan sema kazanir.
"""
from __future__ import annotations

import json
import numpy as np

from src.data import fetch_ohlcv
from src.strategy_v10 import V10Params, V10Costs, simulate_v10
from src.metrics import compute_metrics

DAYS = 60
BASE = dict(entry_mode="signal_then_touch", band_tf="15m",
            use_strength_filter=True, strength_metric="both",
            deep_os_level=25, deep_ob_level=75,
            use_htf_filter=True, use_nft_dir_filter=True, band_tp_active=False)

# (etiket, use_pct, sl, tp)  -- band_tp kapali, saf %
SCHEMES = [
    ("%1.0/1.0 (1:1)",  1.0, 1.0),
    ("%1.0/1.5",        1.0, 1.5),
    ("%1.0/2.0 (1:2)",  1.0, 2.0),
    ("%1.5/2.0",        1.5, 2.0),
    ("%1.5/3.0 (1:2)",  1.5, 3.0),
    ("%2.0/3.0",        2.0, 3.0),
    ("%2.0/4.0 (1:2)",  2.0, 4.0),
    ("%1.0/3.0 (1:3)",  1.0, 3.0),
]


def run_scheme(coins, ref, sl, tp, days=DAYS, split=None):
    p = V10Params(use_pct_sl=True, use_pct_tp=True, pct_sl=sl, pct_tp=tp, **BASE)
    costs = V10Costs(sizing_mode="risk_pct", risk_per_trade=0.01, initial_equity=1000.0)
    rets, pfs, wins, dds, trs = [], [], [], [], []
    for sym in coins:
        df = fetch_ohlcv(sym, "3m", days)
        if split == "train":
            df = df.iloc[:int(len(df)*0.70)]
        elif split == "test":
            df = df.iloc[int(len(df)*0.70):]
        r = simulate_v10(df, "3m", p, costs, nft_ref_df=ref)
        m = compute_metrics(r, "3m")
        if m["islem_sayisi"] < 5:
            continue
        rets.append(m["toplam_getiri_%"]); wins.append(m["kazanma_orani_%"])
        dds.append(m["max_drawdown_%"]); trs.append(m["islem_sayisi"])
        pf = m["profit_factor"]; pfs.append(pf if pf != "inf" else 5.0)
    n = len(rets)
    if n == 0:
        return None
    rets = np.array(rets)
    return dict(
        coins=n,
        pct_profitable=round((rets > 0).mean()*100, 1),
        med_ret=round(float(np.median(rets)), 2),
        mean_ret=round(float(rets.mean()), 2),
        med_pf=round(float(np.median(pfs)), 2),
        med_win=round(float(np.median(wins)), 1),
        med_dd=round(float(np.median(dds)), 1),
        med_trades=int(np.median(trs)),
        calmar=round(float(np.median(rets))/abs(float(np.median(dds))), 2) if np.median(dds) else 0,
    )


def main():
    coins = json.load(open("results/universe.json"))
    ref = fetch_ohlcv("ETH/USDT", "3m", DAYS)
    print(f"Evren: {len(coins)} coin | NFT ref: ETH 3m | ayarlar: band15m/sig>touch/guc/HTF/NFT\n")

    hdr = f"{'SL/TP sema':<18}{'coin':<6}{'karli%':<8}{'medGetiri':<10}{'medPF':<7}{'medWin':<8}{'medDD':<8}{'Calmar':<7}"
    print("=== TUM EVREN (60g) ==="); print(hdr); print("-"*len(hdr))
    results = []
    for lab, sl, tp in SCHEMES:
        r = run_scheme(coins, ref, sl, tp)
        if not r: continue
        results.append((lab, sl, tp, r))
        print(f"{lab:<18}{r['coins']:<6}{r['pct_profitable']:<8}{r['med_ret']:<10}"
              f"{r['med_pf']:<7}{r['med_win']:<8}{r['med_dd']:<8}{r['calmar']:<7}")

    # skor: once kac coin karli, sonra Calmar
    results.sort(key=lambda x: (x[3]["pct_profitable"], x[3]["calmar"]), reverse=True)
    win_lab, win_sl, win_tp, _ = results[0]
    print(f"\n>>> KAZANAN SEMA: {win_lab}  (SL %{win_sl} / TP %{win_tp})")

    print("\n=== KAZANANIN TRAIN/TEST (OOS) DOGRULAMASI ===")
    for part in ["train", "test"]:
        r = run_scheme(coins, ref, win_sl, win_tp, split=part)
        print(f"  {part:<6}: karli %{r['pct_profitable']} | medGetiri {r['med_ret']} | "
              f"medPF {r['med_pf']} | medWin {r['med_win']}% | Calmar {r['calmar']}")

    json.dump({"scheme": win_lab, "sl": win_sl, "tp": win_tp}, open("results/best_sltp.json", "w"))


if __name__ == "__main__":
    main()
