"""Настройки скальпинг-бота по золоту (GOLD / XAUUSD)."""

# Торговый инструмент (у FxPro золото называется GOLD)
SYMBOL = "GOLD"

# M5 — рабочий ТФ; M15 используется как фильтр тренда (как в типичных gold EA)
TIMEFRAME = "M5"
TREND_TIMEFRAME = "M15"

# Объём сделки (лоты)
LOT = 0.01

# Минимальные SL/TP в пунктах (1 пункт GOLD = 0.01)
SL_POINTS = 300
TP_POINTS = 600

STOPS_BUFFER_POINTS = 20
MAX_SPREAD_POINTS = 40

ATR_PERIOD = 14
ATR_SL_MULT = 1.8
ATR_TP_MULT = 2.8

# Две EMA, как в большинстве советников по золоту
EMA_FAST = 8
EMA_SLOW = 21

# RSI: не покупаем на перекупе, не продаём на перепроданности
RSI_PERIOD = 14
RSI_BUY_MIN = 48
RSI_BUY_MAX = 72
RSI_SELL_MIN = 28
RSI_SELL_MAX = 52

LOOKBACK_BARS = 20
MIN_CANDLE_BODY_POINTS = 15

# Короткая пауза только после убытка (после профита не ждём)
COOLDOWN_AFTER_LOSS_SEC = 15
COOLDOWN_AFTER_WIN_SEC = 0

# Трейлинг, как в типовых скальперах: безубыток, потом подтягивание SL
TRAIL_BE_POINTS = 120
TRAIL_START_POINTS = 180
TRAIL_STEP_POINTS = 100

# Учимся по последним закрытым сделкам бота
LEARN_DEALS = 8

CHECK_INTERVAL = 5
MAGIC_NUMBER = 123456
CANDLES_COUNT = 250
DEVIATION = 30
