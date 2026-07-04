#!/usr/bin/env python3
"""
V10 Paper Trading Botu — yerel simulasyon (API anahtari gerekmez, sifir risk).

Dogrulanmis kazanan config + %1 risk sizing ile BTC/ETH/SOL/BNB sepetini 3m'de
canli Binance fiyatlariyla simule eder. Kill-switch (gunluk zarar + toplam DD),
kalici durum (results/paper_state.json), islem loglari.

Modlar:
  python paper_bot.py --replay-days 30   # OFFLINE dogrulama: gecmis veriyi bar-bar
                                          #   canli hatti uzerinden gecir, sonucu goster
  python paper_bot.py --once             # bir tik (yeni kapanmis mumlari isle) - scheduler icin
  python paper_bot.py --loop             # her 3m mum kapanisinda otomatik calis
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

from src.data import fetch_recent, fetch_ohlcv
from src.strategy_v10 import V10Params, derived_frame
from src.live_engine import V10LiveEngine
from src.broker import PaperBroker

STATE_PATH = "results/paper_state.json"
LOG_PATH = "results/paper_log.txt"
WARMUP_BARS = 1400   # HTF EMA200 (15m) + EMA200 (3m) icin yeterli gecmis

# 50 coin evreni (results/universe.json). Yoksa kucuk sepete dus.
try:
    SYMBOLS = json.load(open("results/universe.json"))
except Exception:
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
REF_SYMBOL = "ETH/USDT"
TF = "3m"; REF_TF = "3m"

# Risk yonetimi (50 coin portfoy optimizasyonu ile secildi)
MAX_CONCURRENT   = 5      # es zamanli acik pozisyon siniri (korelasyon riskini sinirla)
# Kill-switch limitleri
DAILY_LOSS_LIMIT = 0.03   # gun basi equity'nin %3'u
MAX_TOTAL_DD     = 0.15   # tepe equity'den %15


def winning_params() -> V10Params:
    # 50 coin evreninde SL/TP optimizasyonu + kullanici TV ayarlariyla kilitlendi.
    return V10Params(
        entry_mode="signal_then_touch",
        band_tf="15m",                       # kullanici: 15dk band
        use_strength_filter=True, strength_metric="both",
        deep_os_level=25, deep_ob_level=75,
        use_htf_filter=True,
        use_nft_dir_filter=True,             # NFT yon = ETH 3m (referans)
        nft_ref_symbol="ETH/USDT", nft_dir_tf="3m",
        use_pct_sl=True, use_pct_tp=True, band_tp_active=False,
        pct_sl=2.0, pct_tp=3.0,              # <-- optimize sonucu (50 coin, %70 karli)
    )


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}"
    print(line)
    os.makedirs("results", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class Bot:
    def __init__(self, initial_equity=1000.0, risk=0.0075, persist=True):
        self.p = winning_params()
        self.persist = persist
        self.broker = PaperBroker(initial_equity=initial_equity, risk_per_trade=risk)
        self.engines = {s: V10LiveEngine(self.p) for s in SYMBOLS}
        self.last_processed = {s: None for s in SYMBOLS}
        self.day = None
        self.day_start_equity = initial_equity
        self.peak_equity = initial_equity
        self.halted = False
        if persist and os.path.exists(STATE_PATH):
            self._load()

    # ---- durum kalici ----
    def _save(self):
        if not self.persist:
            return
        state = dict(
            broker=self.broker.to_dict(),
            engines={s: e.to_dict() for s, e in self.engines.items()},
            last_processed={s: (str(t) if t is not None else None) for s, t in self.last_processed.items()},
            day=self.day, day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity, halted=self.halted,
        )
        os.makedirs("results", exist_ok=True)
        json.dump(state, open(STATE_PATH, "w"), indent=2)

    def _load(self):
        s = json.load(open(STATE_PATH))
        self.broker.load_dict(s["broker"])
        for sym, ed in s["engines"].items():
            self.engines[sym].load_dict(ed)
        self.last_processed = {k: (pd.Timestamp(v) if v else None) for k, v in s["last_processed"].items()}
        self.day = s["day"]; self.day_start_equity = s["day_start_equity"]
        self.peak_equity = s["peak_equity"]; self.halted = s["halted"]
        log(f"Durum yuklendi: equity={self.broker.equity:.2f} halted={self.halted}")

    # ---- bir sembol icin yeni kapanmis mumlari isle ----
    def _process_symbol(self, sym, ref_df, act=True):
        df = fetch_recent(sym, TF, WARMUP_BARS)
        d = derived_frame(df, TF, self.p, nft_ref_df=ref_df)
        closed = d.iloc[:-1]   # olusmakta olan son mumu at
        eng = self.engines[sym]
        lp = self.last_processed[sym]

        if lp is None:
            # ilk calisma: motoru gecmisle isit (islem ACMA), son kapanmisa kadar
            for ts, row in closed.iterrows():
                eng.step(row, warmup=True)
            self.last_processed[sym] = closed.index[-1]
            log(f"{sym}: warmup tamam ({len(closed)} mum), son={closed.index[-1]}")
            return

        new = closed[closed.index > lp]
        for ts, row in new.iterrows():
            # 1) SL/TP kontrolu (bu mumun ici)
            tr = self.broker.on_bar(sym, row["high"], row["low"], row["close"], ts)
            if tr:
                eng.notify_closed()
                log(f"CIKIS {sym} {tr['reason']} @ {tr['exit']} pnl={tr['pnl']:+.2f} eq={tr['equity_after']}")
            # 2) giris sinyali
            action = eng.step(row)
            slot_free = len(self.broker.positions) < MAX_CONCURRENT
            if (act and action["action"] != "none" and sym not in self.broker.positions
                    and not self.halted and slot_free):
                side = 1 if action["action"] == "long" else -1
                pos = self.broker.market_open(sym, side, row["close"], action["sl"], action["tp"], ts)
                if pos:
                    eng.notify_entered(side)
                    log(f"GIRIS {sym} {action['action'].upper()} @ {pos['entry']:.4f} "
                        f"SL={pos['sl']:.4f} TP={pos['tp']:.4f} not={pos['notional']:.1f}")
            else:
                self.broker.mark(sym, row["close"])
            self.last_processed[sym] = ts

    def _check_killswitch(self, ts):
        today = str(pd.Timestamp(ts).date())
        if self.day != today:
            self.day = today
            self.day_start_equity = self.broker.equity_now()
            log(f"Yeni gun {today}: gun basi equity={self.day_start_equity:.2f}")
        eq = self.broker.equity_now()
        self.peak_equity = max(self.peak_equity, eq)
        if self.halted:
            return
        if eq <= self.day_start_equity * (1 - DAILY_LOSS_LIMIT):
            self.broker.flatten_all(ts); self.halted = True
            log(f"!! KILL-SWITCH: gunluk zarar limiti (%{DAILY_LOSS_LIMIT*100:.0f}). Pozisyonlar kapatildi, DURDU.")
        elif eq <= self.peak_equity * (1 - MAX_TOTAL_DD):
            self.broker.flatten_all(ts); self.halted = True
            log(f"!! KILL-SWITCH: toplam drawdown limiti (%{MAX_TOTAL_DD*100:.0f}). Pozisyonlar kapatildi, DURDU.")

    def tick(self):
        try:
            ref_df = fetch_recent(REF_SYMBOL, REF_TF, WARMUP_BARS)
        except Exception as e:
            log(f"HATA: NFT ref ({REF_SYMBOL}) cekilemedi, tik atlaniyor: {str(e)[:120]}")
            return
        failed = 0
        for sym in SYMBOLS:
            try:
                self._process_symbol(sym, ref_df, act=True)
            except Exception as e:
                failed += 1
                log(f"ATLA {sym}: veri/isleme hatasi: {str(e)[:100]}")
        if failed:
            log(f"  ({failed}/{len(SYMBOLS)} sembol bu tikte atlandi)")
        self._check_killswitch(pd.Timestamp.utcnow())
        snap = self.broker.snapshot()
        log(f"TIK: equity={snap['equity']} mtm={snap['equity_mtm']} "
            f"acik={snap['open_positions']} islem={snap['total_trades']} halted={self.halted}")
        self._save()

    # ---- OFFLINE dogrulama: gecmis veriyi canli hattan gecir ----
    def replay(self, days: int):
        ref = fetch_ohlcv(REF_SYMBOL, REF_TF, days)
        for sym in SYMBOLS:
            df = fetch_ohlcv(sym, TF, days)
            d = derived_frame(df, TF, self.p, nft_ref_df=ref)
            closed = d.iloc[:-1]
            warm = int(len(closed) * 0.15)   # ilk %15 isinma
            eng = self.engines[sym]
            for k, (ts, row) in enumerate(closed.iterrows()):
                if k < warm:
                    eng.step(row, warmup=True); continue
                tr = self.broker.on_bar(sym, row["high"], row["low"], row["close"], ts)
                if tr:
                    eng.notify_closed()
                action = eng.step(row)
                if action["action"] != "none" and sym not in self.broker.positions:
                    side = 1 if action["action"] == "long" else -1
                    pos = self.broker.market_open(sym, side, row["close"], action["sl"], action["tp"], ts)
                    if pos:
                        eng.notify_entered(side)
                else:
                    self.broker.mark(sym, row["close"])
        return self.broker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-days", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=0.0075)
    ap.add_argument("--reset", action="store_true", help="kayitli durumu sil, sifirdan basla")
    args = ap.parse_args()

    if args.reset and os.path.exists(STATE_PATH):
        os.remove(STATE_PATH); print("Durum sifirlandi.")

    if args.replay_days:
        bot = Bot(initial_equity=args.equity, risk=args.risk, persist=False)
        broker = bot.replay(args.replay_days)
        wins = [t for t in broker.trades if t["pnl"] > 0]
        print(f"\n=== REPLAY DOGRULAMA ({args.replay_days} gun, canli hat) ===")
        print(f"  Islem: {len(broker.trades)} | Kazanan: {len(wins)} "
              f"({len(wins)/max(1,len(broker.trades))*100:.1f}%)")
        print(f"  Baslangic: {broker.start_equity:.0f}  ->  Son: {broker.equity:.2f} "
              f"({(broker.equity/broker.start_equity-1)*100:+.2f}%)")
        for t in broker.trades[-8:]:
            print(f"    {t['exit_time'][:16]} {t['symbol']:<9} {t['side']:<5} {t['reason']:<3} pnl={t['pnl']:+7.2f} eq={t['equity_after']}")
        return

    bot = Bot(initial_equity=args.equity, risk=args.risk)
    if args.once:
        bot.tick(); return
    if args.loop:
        log("LOOP basladi. Her 3m mum kapanisinda calisacak. Ctrl+C ile durdur.")
        while True:
            try:
                bot.tick()
            except Exception as e:
                log(f"HATA: {e}")
            # sonraki 3m sinirina kadar uyu (+8sn tampon)
            now = time.time()
            nxt = (now // 180 + 1) * 180 + 8
            time.sleep(max(5, nxt - now))
    else:
        print("Mod sec: --replay-days N | --once | --loop")


if __name__ == "__main__":
    main()
