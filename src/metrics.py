"""Backtest performans metrikleri."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pnls(trades) -> np.ndarray:
    if not trades:
        return np.array([])
    first = trades[0]
    if isinstance(first, dict):
        return np.array([t["pnl"] for t in trades])
    return np.array([t.pnl for t in trades])


def compute_metrics(result: dict, base_tf: str, initial_equity: float | None = None) -> dict:
    trades = result["trades"]
    ec = result["equity_curve"].dropna()
    if "config" in result:
        init = result["config"].initial_equity
    else:
        init = initial_equity if initial_equity is not None else 1000.0
    final = result["final_equity"]

    n = len(trades)
    pnls = _pnls(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown (equity egrisi uzerinden)
    if len(ec):
        roll_max = ec.cummax()
        dd = (ec - roll_max) / roll_max
        max_dd = dd.min() * 100
    else:
        max_dd = 0.0

    # Sharpe (mum getirileri, yillik yaklasik olcek)
    rets = ec.pct_change().dropna()
    bars_per_year = {
        "1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520,
        "1h": 8760, "2h": 4380, "4h": 2190, "1d": 365,
    }.get(base_tf, 8760)
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    return {
        "islem_sayisi": n,
        "kazanan": int((pnls > 0).sum()),
        "kaybeden": int((pnls < 0).sum()),
        "kazanma_orani_%": round(len(wins) / n * 100, 2) if n else 0.0,
        "toplam_getiri_%": round((final / init - 1) * 100, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_drawdown_%": round(max_dd, 2),
        "sharpe": round(float(sharpe), 2),
        "ort_kazanc": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "ort_kayip": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "baslangic_equity": init,
        "son_equity": round(final, 2),
    }


def print_report(metrics: dict, symbol: str, base_tf: str, days: int):
    line = "=" * 52
    print(line)
    print(f"  BACKTEST RAPORU: {symbol} | {base_tf} | son {days} gun")
    print(line)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print(line)


def trades_to_df(result: dict) -> pd.DataFrame:
    trades = result["trades"]
    if trades and isinstance(trades[0], dict):
        return pd.DataFrame(trades)
    return pd.DataFrame([t.__dict__ for t in trades])
