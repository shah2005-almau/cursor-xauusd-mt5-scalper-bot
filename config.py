"""Настройки скальпинг-бота по золоту (GOLD / XAUUSD)."""

# Торговый инструмент (у FxPro золото называется GOLD)
SYMBOL = "GOLD"

# Scalping: entries on closed M1 candles, direction confirmed on M5.
TIMEFRAME = "M1"
TREND_TIMEFRAME = "M5"

# Signal engine.  "BOTH" trades either a confirmed range breakout or a
# continuation pullback; use one of the other values to test each separately.
STRATEGY_MODE = "BOTH"  # BREAKOUT, PULLBACK, BOTH

# Only trade during the liquid London / New York part of the day.  Hours are
# UTC, so they are independent of the computer and broker-server time zones.
SESSION_START_HOUR = 8
SESSION_END_HOUR = 21

# Объём сделки (лоты)
LOT = 0.01

# Минимальные SL/TP в пунктах (1 пункт GOLD = 0.01)
SL_POINTS = 100
TP_POINTS = 160

STOPS_BUFFER_POINTS = 20
MAX_SPREAD_POINTS = 40

ATR_PERIOD = 14
ATR_SL_MULT = 1.8
ATR_TP_MULT = 2.8
ADX_PERIOD = 14
MIN_ADX = 20.0

# Две EMA, как в большинстве советников по золоту
EMA_FAST = 8
EMA_SLOW = 21

# RSI: не покупаем на перекупе, не продаём на перепроданности
RSI_PERIOD = 14
RSI_BUY_MIN = 48
RSI_BUY_MAX = 72
RSI_SELL_MIN = 28
RSI_SELL_MAX = 52

LOOKBACK_BARS = 12
MIN_CANDLE_BODY_POINTS = 8

# Короткая пауза только после убытка (после профита не ждём)
COOLDOWN_AFTER_LOSS_SEC = 15
COOLDOWN_AFTER_WIN_SEC = 0

# Трейлинг, как в типовых скальперах: безубыток, потом подтягивание SL
TRAIL_BE_POINTS = 120
TRAIL_START_POINTS = 180
TRAIL_STEP_POINTS = 100

# Scalping positions must not turn into long holds.  The bot closes its own
# position after this many completed working-timeframe candles.
MAX_HOLD_BARS = 15

# Circuit breaker.  Set a currency amount appropriate to the account.  Once
# the bot's realised P/L for the broker day reaches this loss, it opens no new
# positions until the next broker day.  Zero disables the breaker.
MAX_DAILY_LOSS = 20.0

# Учимся по последним закрытым сделкам бота
LEARN_DEALS = 8

CHECK_INTERVAL = 2
MAGIC_NUMBER = 123456
CANDLES_COUNT = 250
DEVIATION = 30
