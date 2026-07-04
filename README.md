# Sinan Engin NFT Sistemi — Trading Bot

TradingView Pine Script stratejisinin ("Sinan Engin NFT Sistemi HTF Filtreli V3")
Python'a taşınmış, backtest edilebilir ve ileride canlı bota dönüştürülebilir hali.

## Durum
- **Faz 0 — Doğrulama (backtest): TAMAMLANDI ✅** ← şu an buradayız
- Faz 1 — Parametre optimizasyonu + walk-forward
- Faz 2 — Paper trading (Binance Futures testnet)
- Faz 3 — Canlı bot (risk limitleri, kill-switch)

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım
```bash
python run_backtest.py --symbol BTC/USDT --tf 1h --days 365 --save
python run_backtest.py --symbol ETH/USDT --tf 4h --days 730 --risk 0.02 --tp-r 3
```
Parametreler: `--risk` (işlem başına risk %), `--sl-atr` (stop = giriş ∓ N·ATR),
`--tp-r` (take-profit R katsayısı), `--no-htf`, `--no-short`.

## Yapı
| Dosya | Görev |
|---|---|
| `src/indicators.py` | HMA, ATR, MFI, SSL Hybrid, AlphaTrend, EMA — Pine'ın birebir portu |
| `src/strategy.py` | HTF filtresi + 3'lü teyit + tradeState → buy/sell sinyalleri |
| `src/data.py` | Binance Futures OHLCV çekme (ccxt, cache'li) |
| `src/backtest.py` | SL/TP/risk yönetimli event-driven backtest (lookahead yok) |
| `src/metrics.py` | Kazanma oranı, profit factor, max drawdown, Sharpe |
| `run_backtest.py` | CLI |

## Pine'a eklenenler (bir bot için şart olan eksikler)
Orijinal Pine bir `indicator` idi — sadece giriş sinyali vardı. Eklenenler:
- Çıkış mantığı (ATR stop-loss, R-katsayılı take-profit, ters sinyalde kapama)
- Pozisyon boyutu (işlem başına sabit risk %)
- Komisyon + slippage modeli
- Performans metrikleri

## Bulgular

### V3 (ilk sistem — src/strategy.py)
Yüksek zaman dilimlerinde (1h/4h) küçük pozitif edge ama yılda ~4–11 işlem
(istatistiksel güven düşük). 1m/3m/5m'de zarar. Teyit penceresi denendi, işe
yaramadı (window=1 en iyi). Sonuç: HTF trend sistemi, scalp'e uygun değil.

### V10 (SE NFT + MTF RSI Band — src/strategy_v10.py)
Girişi banda geri çekilmede yapıyor (mean-reversion) → scalp'e uygun.
Optimizasyon (train/test OOS) sonucu **doğrulanmış kazanan config**:
- `entry_mode = signal_then_touch` (NFT sinyali → banda temasla gir)
- SL/TP = **saf % 2/4 (1:2 R:R), band-TP KAPALI** (band-TP kazançları boğuyordu)
- güç filtresi ON, HTF ON, NFT yön filtresi OFF
- **3m** zaman dilimi (1m ölü — komisyon yiyor)

Sonuç (120 gün, 5 sembol): BTC/ETH/SOL/BNB pozitif (PF 1.25–1.38), XRP zararda.
OOS train/test'te de 4/5 sembol pozitif. **Çalıştır:** `python run_v10.py --symbol SOL/USDT`

**AÇIK RİSK:** Pozisyon boyutu %100 özkaynak (Pine ayarı) → drawdown −13%…−32%.
Canlıya geçmeden işlem başına %1–2 riskle sizing eklenmeli. XRP sepetten çıkarılmalı.

## Paper Trading (Faz 2 — yerel simülasyon, sıfır risk)
`paper_bot.py` doğrulanmış config + %1 risk ile BTC/ETH/SOL/BNB sepetini 3m'de
canlı Binance fiyatlarıyla simüle eder. **API anahtarı gerekmez.** Kill-switch
(günlük −%3, toplam −%15), kalıcı durum, işlem logları.
```bash
python paper_bot.py --replay-days 60   # offline doğrulama (geçmişi canlı hattan geçir)
python paper_bot.py --reset --once      # ilk çalıştırma (warmup + durum kaydı)
python paper_bot.py --loop              # her 3m mum kapanışında otomatik (Ctrl+C ile dur)
```
Durum: `results/paper_state.json` · Log: `results/paper_log.txt`
Testnet'e geçiş: `src/broker.py` → `TestnetBroker` (ayrı testnet anahtarı ister).

## Betikler
| Betik | İş |
|---|---|
| `run_backtest.py` | V3 tek backtest |
| `run_v10.py` | V10 kazanan config ile tek çalıştırma |
| `optimize_v10.py` | V10 parametre taraması (train/test OOS) |
| `paper_bot.py` | Faz 2 paper trading botu (yerel sim) |
| `sweep.py` | Çoklu TF/sembol taraması |
