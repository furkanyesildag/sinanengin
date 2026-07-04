"""
V10 canli motoru: bar-bar ilerleyen stateful giris sinyali ureticisi.

simulate_v10'un ic dongusundeki GIRIS mantiginin birebir ayni state makinesi,
ama pozisyonu/cikisi disaridan (broker) yonetiliyor. Bot her yeni KAPANMIS
3m mumunda step() cagirir; motor gerekiyorsa {action, sl, tp} dondurur.
"""
from __future__ import annotations

import numpy as np

from .strategy_v10 import V10Params


class V10LiveEngine:
    def __init__(self, params: V10Params):
        self.p = params
        self.waitLB = False
        self.waitUB = False
        self.pendingLong = False
        self.pendingShort = False
        self.pendingLongBar = -1
        self.pendingShortBar = -1
        self.tradeState = 0     # 0 bosta, 1 long, -1 short (broker ile senkron)
        self.bar = 0

    # --- broker senkronizasyonu ---
    def notify_entered(self, side: int):
        self.tradeState = side
        self.waitLB = self.waitUB = False
        self.pendingLong = self.pendingShort = False

    def notify_closed(self):
        self.tradeState = 0
        self.waitLB = self.waitUB = False
        self.pendingLong = self.pendingShort = False

    def _sl_tp(self, side: int, row) -> tuple[float, float | None]:
        p = self.p
        c, lb, ub, atr = row["close"], row["lb"], row["ub"], row["atr_now"]
        if side == 1:
            sl = c * (1 - p.pct_sl / 100.0) if p.use_pct_sl else lb - atr * p.atr_sl_buffer
            tp = c * (1 + p.pct_tp / 100.0) if p.use_pct_tp else None
        else:
            sl = c * (1 + p.pct_sl / 100.0) if p.use_pct_sl else ub + atr * p.atr_sl_buffer
            tp = c * (1 - p.pct_tp / 100.0) if p.use_pct_tp else None
        return sl, tp

    def step(self, row, warmup: bool = False) -> dict:
        """row: derived_frame'in tek satiri (dict benzeri). warmup=True ise
        eylem uretilmez, sadece dahili durum ilerletilir."""
        p = self.p
        do_long = do_short = False

        if self.tradeState == 0:
            base_long = base_short = False
            tl = bool(row["touched_lb"]); tu = bool(row["touched_ub"])
            rb = bool(row["raw_buy"]); rs = bool(row["raw_sell"])

            if p.entry_mode == "touch":
                if tl and not self.waitLB:
                    self.waitLB = True
                if (self.waitLB or tl) and rb:
                    base_long = True
                if tu and not self.waitUB:
                    self.waitUB = True
                if (self.waitUB or tu) and rs:
                    base_short = True
            else:  # signal_then_touch
                if rb:
                    self.waitLB = True
                if self.waitLB and tl:
                    base_long = True
                if rs:
                    self.waitUB = True
                if self.waitUB and tu:
                    base_short = True

            # guc kapisi
            if base_long:
                self.waitLB = False
                if (not p.use_strength_filter) or (not bool(row["strong_down"])):
                    do_long = True
                else:
                    self.pendingLong = True; self.pendingLongBar = self.bar
            if base_short:
                self.waitUB = False
                if (not p.use_strength_filter) or (not bool(row["strong_up"])):
                    do_short = True
                else:
                    self.pendingShort = True; self.pendingShortBar = self.bar

            # bekleyen teyit (reclaim / timeout)
            if self.pendingLong and not do_long:
                if bool(row["reclaim_long"]):
                    do_long = True; self.pendingLong = False
                elif self.bar - self.pendingLongBar >= p.confirm_timeout:
                    self.pendingLong = False
            if self.pendingShort and not do_short:
                if bool(row["reclaim_short"]):
                    do_short = True; self.pendingShort = False
                elif self.bar - self.pendingShortBar >= p.confirm_timeout:
                    self.pendingShort = False

            # NFT yon vetosu
            if p.use_nft_dir_filter:
                ns = int(row["nft_state"])
                if do_long and ns == -1:
                    do_long = False
                if do_short and ns == 1:
                    do_short = False

            if do_long or do_short:
                self.waitLB = self.waitUB = False
                self.pendingLong = self.pendingShort = False

        self.bar += 1

        if warmup or (not do_long and not do_short):
            return {"action": "none"}

        side = 1 if do_long else -1
        sl, tp = self._sl_tp(side, row)
        return {"action": "long" if side == 1 else "short", "sl": sl, "tp": tp,
                "ref_close": float(row["close"])}

    # durumu diske yaz/oku (bot yeniden baslarsa)
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ["waitLB", "waitUB", "pendingLong", "pendingShort",
                 "pendingLongBar", "pendingShortBar", "tradeState", "bar"]}

    def load_dict(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
