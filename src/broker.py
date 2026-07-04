"""
Broker soyutlamasi.

PaperBroker  : yerel simulasyon. API anahtari GEREKMEZ, sifir risk. Canli
               (public) fiyatlarla $ hesabini yerelde simule eder. Su an kullandigimiz.
TestnetBroker: Binance Futures testnet (ccxt sandbox). AYRI testnet anahtari ister
               (testnet.binancefuture.com). Ileride paper->testnet gecisi icin.

Ikisi de ayni arayuzu sunar: reset/market_open/on_bar/mark/flatten_all/snapshot.
Boylece bot kodu degismeden backend degistirilebilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class PaperBroker:
    def __init__(self, initial_equity=1000.0, fee_rate=0.0005, slippage=0.0003,
                 risk_per_trade=0.01, max_leverage=10.0):
        self.fee = fee_rate
        self.slip = slippage
        self.risk = risk_per_trade
        self.max_lev = max_leverage
        self.equity = initial_equity
        self.start_equity = initial_equity
        self.positions: dict = {}     # symbol -> pozisyon dict
        self.trades: list = []
        self.last_price: dict = {}

    # --- pozisyon ac (market, bir sonraki fiyattan ~ ref_price) ---
    def market_open(self, symbol, side, ref_price, sl, tp, ts):
        if symbol in self.positions:
            return None
        fill = ref_price * (1 + self.slip * side)
        stop_dist = abs(fill - sl)
        if stop_dist <= 0:
            return None
        risk_cash = self.equity * self.risk
        notional = min(risk_cash / (stop_dist / fill), self.equity * self.max_lev)
        qty = notional / fill
        self.positions[symbol] = dict(side=side, qty=qty, entry=fill, sl=sl, tp=tp,
                                      notional=notional, entry_time=str(ts))
        return self.positions[symbol]

    # --- yeni mum: SL/TP kontrolu (mum ici) ---
    def on_bar(self, symbol, high, low, close, ts):
        self.last_price[symbol] = close
        pos = self.positions.get(symbol)
        if not pos:
            return None
        side = pos["side"]; sl = pos["sl"]; tp = pos["tp"]
        exit_price = None; reason = None
        if side == 1:
            if low <= sl:
                exit_price, reason = sl * (1 - self.slip), "SL"
            elif tp is not None and high >= tp:
                exit_price, reason = tp, "TP"
        else:
            if high >= sl:
                exit_price, reason = sl * (1 + self.slip), "SL"
            elif tp is not None and low <= tp:
                exit_price, reason = tp, "TP"
        if exit_price is None:
            return None
        return self._close(symbol, exit_price, reason, ts)

    def _close(self, symbol, exit_price, reason, ts):
        pos = self.positions.pop(symbol)
        side = pos["side"]
        gross = (exit_price - pos["entry"]) * pos["qty"] * side
        fees = pos["notional"] * self.fee + (pos["qty"] * exit_price) * self.fee
        pnl = gross - fees
        self.equity += pnl
        trade = dict(symbol=symbol, side="LONG" if side == 1 else "SHORT",
                     entry=round(pos["entry"], 6), exit=round(exit_price, 6),
                     qty=round(pos["qty"], 6), pnl=round(pnl, 2),
                     pnl_pct=round(pnl / pos["notional"] * 100, 3),
                     reason=reason, entry_time=pos["entry_time"], exit_time=str(ts),
                     equity_after=round(self.equity, 2))
        self.trades.append(trade)
        return trade

    def mark(self, symbol, price):
        self.last_price[symbol] = price

    def equity_now(self):
        eq = self.equity
        for sym, pos in self.positions.items():
            px = self.last_price.get(sym, pos["entry"])
            eq += (px - pos["entry"]) * pos["qty"] * pos["side"]
        return eq

    def flatten_all(self, ts):
        out = []
        for sym in list(self.positions):
            px = self.last_price.get(sym, self.positions[sym]["entry"])
            out.append(self._close(sym, px, "FLATTEN", ts))
        return out

    def snapshot(self):
        return dict(equity=round(self.equity, 2), equity_mtm=round(self.equity_now(), 2),
                    open_positions=len(self.positions), total_trades=len(self.trades))

    def to_dict(self):
        return dict(equity=self.equity, start_equity=self.start_equity,
                    positions=self.positions, trades=self.trades, last_price=self.last_price)

    def load_dict(self, d):
        self.equity = d["equity"]; self.start_equity = d["start_equity"]
        self.positions = d["positions"]; self.trades = d["trades"]
        self.last_price = d.get("last_price", {})


class TestnetBroker:
    """Binance Futures testnet (ccxt sandbox). Gercek testnet emirleri gonderir.
    NOT: testnet.binancefuture.com'dan alinan AYRI anahtar ister. Mainnet anahtari
    burada CALISMAZ. Paper asamasi basariyla gecince devreye alinacak.
    """
    def __init__(self, api_key, api_secret, **kw):
        import ccxt
        self.ex = ccxt.binanceusdm({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True, "options": {"defaultType": "future"},
        })
        self.ex.set_sandbox_mode(True)  # <-- testnet
        self.risk = kw.get("risk_per_trade", 0.01)
        self.max_lev = kw.get("max_leverage", 10.0)

    def market_open(self, symbol, side, ref_price, sl, tp, ts):
        bal = self.ex.fetch_balance()["USDT"]["free"]
        stop_dist = abs(ref_price - sl)
        notional = min((bal * self.risk) / (stop_dist / ref_price), bal * self.max_lev)
        qty = self.ex.amount_to_precision(symbol, notional / ref_price)
        order_side = "buy" if side == 1 else "sell"
        self.ex.create_order(symbol, "market", order_side, qty)
        # bracket: reduce-only stop + take-profit
        close_side = "sell" if side == 1 else "buy"
        self.ex.create_order(symbol, "STOP_MARKET", close_side, qty, None,
                             {"stopPrice": self.ex.price_to_precision(symbol, sl), "reduceOnly": True})
        if tp is not None:
            self.ex.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None,
                                 {"stopPrice": self.ex.price_to_precision(symbol, tp), "reduceOnly": True})
        return {"symbol": symbol, "side": side, "qty": qty}

    def positions_open(self):
        return {p["symbol"]: p for p in self.ex.fetch_positions() if abs(float(p["contracts"] or 0)) > 0}

    def flatten_all(self, ts):
        for sym, p in self.positions_open().items():
            side = "sell" if float(p["contracts"]) > 0 else "buy"
            self.ex.create_order(sym, "market", side, abs(float(p["contracts"])), None, {"reduceOnly": True})
        self.ex.cancel_all_orders()
