"""
Скальпинг-бот XAUUSD для MetaTrader 5.

Стратегия: пробой + EMA 8/21 + RSI + тренд M15 + ATR и трейлинг.
Перед запуском должен быть открыт терминал MT5 и выполнен вход в счёт.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config

# Соответствие строкового таймфрейма константам MT5
TIMEFRAME_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


def calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Экспоненциальная скользящая средняя (EMA) через numpy."""
    if period < 1:
        raise ValueError("period EMA должен быть >= 1")
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    alpha = 2.0 / (period + 1.0)
    ema = np.empty(closes.size, dtype=np.float64)
    ema[0] = float(closes[0])
    for i in range(1, closes.size):
        ema[i] = alpha * float(closes[i]) + (1.0 - alpha) * ema[i - 1]
    return ema


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (ATR) — типичный диапазон свечи."""
    n = close.size
    atr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return atr
    tr = np.empty(n, dtype=np.float64)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def calc_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI Уайлдера — фильтр, как в типичных советниках."""
    n = closes.size
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    rsi[period] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_loss == 0.0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi[:period] = rsi[period]
    return rsi


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ADX: trend-strength filter, independent of trend direction."""
    n = close.size
    adx = np.zeros(n, dtype=np.float64)
    if n < period + 2:
        return adx

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = float(high[i] - high[i - 1])
        down = float(low[i - 1] - low[i])
        plus_dm[i] = up if up > down and up > 0.0 else 0.0
        minus_dm[i] = down if down > up and down > 0.0 else 0.0
        tr[i] = max(float(high[i] - low[i]), abs(float(high[i] - close[i - 1])), abs(float(low[i] - close[i - 1])))

    atr = np.zeros(n, dtype=np.float64)
    smooth_plus = np.zeros(n, dtype=np.float64)
    smooth_minus = np.zeros(n, dtype=np.float64)
    atr[period] = float(np.mean(tr[1 : period + 1]))
    smooth_plus[period] = float(np.mean(plus_dm[1 : period + 1]))
    smooth_minus[period] = float(np.mean(minus_dm[1 : period + 1]))
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if i > period:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period
            smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period
        if atr[i] > 0.0:
            plus_di = 100.0 * smooth_plus[i] / atr[i]
            minus_di = 100.0 * smooth_minus[i] / atr[i]
            total = plus_di + minus_di
            dx[i] = 100.0 * abs(plus_di - minus_di) / total if total > 0.0 else 0.0
    start = period * 2 - 1
    if start < n:
        adx[start] = float(np.mean(dx[period : start + 1]))
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        adx[:start] = adx[start]
    return adx


class GoldBot:
    """Торговый бот по золоту: подключение к MT5, сигналы и рыночные ордера."""

    def __init__(self) -> None:
        self.symbol = config.SYMBOL
        self.timeframe_name = config.TIMEFRAME
        self.trend_tf_name = config.TREND_TIMEFRAME
        self.lot = float(config.LOT)
        self.tp_points = int(config.TP_POINTS)
        self.sl_points = int(config.SL_POINTS)
        self.min_risk_reward = float(config.MIN_RISK_REWARD)
        self.ema_fast_period = int(config.EMA_FAST)
        self.ema_slow_period = int(config.EMA_SLOW)
        self.rsi_period = int(config.RSI_PERIOD)
        self.rsi_buy_min = float(config.RSI_BUY_MIN)
        self.rsi_buy_max = float(config.RSI_BUY_MAX)
        self.rsi_sell_min = float(config.RSI_SELL_MIN)
        self.rsi_sell_max = float(config.RSI_SELL_MAX)
        self.lookback = int(config.LOOKBACK_BARS)
        self.check_interval = int(config.CHECK_INTERVAL)
        self.magic = int(config.MAGIC_NUMBER)
        self.candles_count = int(config.CANDLES_COUNT)
        self.deviation = int(config.DEVIATION)
        self.stops_buffer_points = int(config.STOPS_BUFFER_POINTS)
        self.max_spread_points = int(config.MAX_SPREAD_POINTS)
        self.atr_period = int(config.ATR_PERIOD)
        self.atr_sl_mult = float(config.ATR_SL_MULT)
        self.atr_tp_mult = float(config.ATR_TP_MULT)
        self.adx_period = int(config.ADX_PERIOD)
        self.min_adx = float(config.MIN_ADX)
        self.strategy_mode = str(config.STRATEGY_MODE).upper()
        self.session_start_hour = int(config.SESSION_START_HOUR)
        self.session_end_hour = int(config.SESSION_END_HOUR)
        self.max_hold_bars = int(config.MAX_HOLD_BARS)
        self.max_daily_loss = float(config.MAX_DAILY_LOSS)
        self.max_trades_per_day = int(config.MAX_TRADES_PER_DAY)
        self.max_consecutive_losses = int(config.MAX_CONSECUTIVE_LOSSES)
        self.min_tick_volume_ratio = float(config.MIN_TICK_VOLUME_RATIO)
        self.min_body_points = int(config.MIN_CANDLE_BODY_POINTS)
        self.cooldown_loss_sec = int(config.COOLDOWN_AFTER_LOSS_SEC)
        self.cooldown_win_sec = int(config.COOLDOWN_AFTER_WIN_SEC)
        self.trail_be_points = int(config.TRAIL_BE_POINTS)
        self.trail_start_points = int(config.TRAIL_START_POINTS)
        self.trail_step_points = int(config.TRAIL_STEP_POINTS)
        self.learn_deals = int(config.LEARN_DEALS)
        self._connected = False
        self._last_signal_bar: Optional[pd.Timestamp] = None

    def _gold_candidates(self) -> list[str]:
        """Имена золота у разных брокеров (FxPro часто GOLD, не XAUUSD)."""
        seen: set[str] = set()
        names: list[str] = []
        for name in (
            self.symbol,
            "XAUUSD",
            "GOLD",
            "XAUUSD.",
            "XAUUSDm",
            "XAUUSD.a",
            "XAUUSDpro",
            "GOLDm",
            "GOLD.",
        ):
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def _list_gold_symbols(self) -> list[str]:
        """Все символы терминала, похожие на золото."""
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        found: list[str] = []
        for item in symbols:
            upper = item.name.upper()
            if "XAU" in upper or "GOLD" in upper:
                found.append(item.name)
        return found

    def _resolve_symbol(self) -> Optional[str]:
        """Выбрать торговый символ: config, типичные алиасы, поиск в терминале."""
        for name in self._gold_candidates():
            if mt5.symbol_select(name, True):
                info = mt5.symbol_info(name)
                if info is not None:
                    if name != self.symbol:
                        print(
                            f"[OK] Символ {self.symbol!r} недоступен, "
                            f"используем {name!r}. Пропишите его в config.py."
                        )
                    return name

        gold = self._list_gold_symbols()
        if gold:
            print(f"[ИНФО] В терминале найдены символы золота: {', '.join(gold)}")
            for name in gold:
                if mt5.symbol_select(name, True) and mt5.symbol_info(name) is not None:
                    print(f"[OK] Берём символ {name!r}. Запишите его в config.py как SYMBOL.")
                    return name

        err = mt5.last_error()
        print(f"[ОШИБКА] Не удалось выбрать символ {self.symbol}: {err}")
        if gold:
            print("Укажите точное имя из списка выше в config.py → SYMBOL.")
        else:
            print(
                "Python не достучался до списка символов (часто 'Terminal: Call failed'). "
                "Оставьте открытым только один MT5, Python 64-bit как терминал, "
                "перезапустите MT5 и снова python bot.py."
            )
        return None

    def connect(self) -> bool:
        """Подключение к MT5, проверка терминала и наличие символа золота."""
        if not mt5.initialize():
            print(
                f"[ОШИБКА] Не удалось подключиться к MT5: {mt5.last_error()}. "
                "Запустите терминал MetaTrader 5 и повторите попытку."
            )
            return False

        # Даём IPC-каналу время подняться после initialize()
        time.sleep(0.5)

        terminal = mt5.terminal_info()
        if terminal is None:
            print(f"[ОШИБКА] Терминал недоступен: {mt5.last_error()}")
            mt5.shutdown()
            return False

        print(f"[ИНФО] Терминал: {terminal.name}, путь: {terminal.path}")

        if not terminal.connected:
            print("[ОШИБКА] Терминал запущен, но нет связи с торговым сервером.")
            mt5.shutdown()
            return False

        if not terminal.trade_allowed:
            print(
                "[ПРЕДУПРЕЖДЕНИЕ] Автоторговля выключена. "
                "На панели MT5 нажмите кнопку «Algo Trading» / «Алготорг» "
                "(должна стать активной). Галочки в Настройки → Советники недостаточно."
            )

        account = mt5.account_info()
        if account is None:
            print(f"[ОШИБКА] Нет данных счёта: {mt5.last_error()}")
            mt5.shutdown()
            return False

        if self.timeframe_name not in TIMEFRAME_MAP:
            print(
                f"[ОШИБКА] Неизвестный TIMEFRAME={self.timeframe_name!r}. "
                f"Допустимо: {', '.join(TIMEFRAME_MAP)}"
            )
            mt5.shutdown()
            return False
        if self.trend_tf_name not in TIMEFRAME_MAP:
            print(
                f"[ОШИБКА] Неизвестный TREND_TIMEFRAME={self.trend_tf_name!r}. "
                f"Допустимо: {', '.join(TIMEFRAME_MAP)}"
            )
            mt5.shutdown()
            return False
        if self.strategy_mode not in {"BREAKOUT", "PULLBACK", "BOTH"}:
            print("[ОШИБКА] STRATEGY_MODE: BREAKOUT, PULLBACK или BOTH")
            mt5.shutdown()
            return False
        if self.min_risk_reward < 1.0:
            print("[ОШИБКА] MIN_RISK_REWARD должен быть не меньше 1.0")
            mt5.shutdown()
            return False
        if not (0 <= self.session_start_hour <= 23 and 0 <= self.session_end_hour <= 23):
            print("[ОШИБКА] SESSION_START_HOUR и SESSION_END_HOUR должны быть в диапазоне 0..23")
            mt5.shutdown()
            return False

        resolved = self._resolve_symbol()
        if resolved is None:
            mt5.shutdown()
            return False
        self.symbol = resolved

        info = mt5.symbol_info(self.symbol)
        if info is None:
            print(f"[ОШИБКА] Нет информации по символу {self.symbol}: {mt5.last_error()}")
            mt5.shutdown()
            return False

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"[ОШИБКА] Не удалось сделать {self.symbol} видимым в Market Watch.")
                mt5.shutdown()
                return False

        self._connected = True
        tick = mt5.symbol_info_tick(self.symbol)
        spread = (float(tick.ask) - float(tick.bid)) if tick else 0.0
        print(
            f"[OK] Подключено к MT5. Счёт {account.login}, сервер {account.server}, "
            f"символ {self.symbol}, point={info.point}, digits={info.digits}, "
            f"спред={spread:.{info.digits}f}, stops_level={info.trade_stops_level}"
        )
        return True

    def get_data(self) -> Optional[pd.DataFrame]:
        """Последние CANDLES_COUNT свечей выбранного таймфрейма в DataFrame."""
        tf = TIMEFRAME_MAP[self.timeframe_name]
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, self.candles_count)
        if rates is None or len(rates) == 0:
            print(f"[ОШИБКА] Не удалось получить свечи: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        closes = df["close"].to_numpy(dtype=np.float64)
        df["ema_fast"] = calc_ema(closes, self.ema_fast_period)
        df["ema_slow"] = calc_ema(closes, self.ema_slow_period)
        df["rsi"] = calc_rsi(closes, self.rsi_period)
        df["atr"] = calc_atr(
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            closes,
            self.atr_period,
        )
        df["adx"] = calc_adx(
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            closes,
            self.adx_period,
        )
        return df

    def m15_trend(self) -> Optional[str]:
        """Направление старшего ТФ: BUY если EMA8 > EMA21 на закрытой свече."""
        tf = TIMEFRAME_MAP[self.trend_tf_name]
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, 80)
        if rates is None or len(rates) < 30:
            return None
        closes = pd.DataFrame(rates)["close"].to_numpy(dtype=np.float64)
        fast = calc_ema(closes, self.ema_fast_period)
        slow = calc_ema(closes, self.ema_slow_period)
        if fast[-2] > slow[-2]:
            return "BUY"
        if fast[-2] < slow[-2]:
            return "SELL"
        return None

    def in_trading_session(self) -> bool:
        """Return whether current UTC time is inside the configured session."""
        hour = datetime.now(UTC).hour
        if self.session_start_hour == self.session_end_hour:
            return True
        if self.session_start_hour < self.session_end_hour:
            return self.session_start_hour <= hour < self.session_end_hour
        return hour >= self.session_start_hour or hour < self.session_end_hour

    def check_signal(self, df: pd.DataFrame) -> Optional[tuple[str, str]]:
        """
        Закрытая свеча: пробой + быстрая EMA над/под медленной + RSI + тренд M15.
        """
        need = self.lookback + max(self.atr_period, self.rsi_period, self.ema_slow_period) + 3
        if df is None or len(df) < need:
            print(f"[ОШИБКА] Мало свечей для сигнала: нужно минимум {need}")
            return None

        signal_bar = df.iloc[-2]
        prev = df.iloc[-(self.lookback + 2) : -2]
        close = float(signal_bar["close"])
        open_ = float(signal_bar["open"])
        ema_fast = float(signal_bar["ema_fast"])
        ema_slow = float(signal_bar["ema_slow"])
        rsi = float(signal_bar["rsi"])
        adx = float(signal_bar["adx"])
        body = abs(close - open_)
        info = mt5.symbol_info(self.symbol)
        point = float(info.point) if info is not None else 0.01
        min_body = self.min_body_points * point
        mean_tick_volume = float(prev["tick_volume"].mean())
        volume_ratio = float(signal_bar["tick_volume"]) / mean_tick_volume if mean_tick_volume > 0.0 else 0.0

        if body < min_body or adx < self.min_adx or volume_ratio < self.min_tick_volume_ratio:
            return None

        high_max = float(prev["high"].max())
        low_min = float(prev["low"].min())
        trend = self.m15_trend()
        blocked = self.learned_block_direction()

        buy_ok = (
            close > high_max
            and close > open_
            and ema_fast > ema_slow
            and self.rsi_buy_min <= rsi <= self.rsi_buy_max
            and trend == "BUY"
            and blocked != "BUY"
        )
        sell_ok = (
            close < low_min
            and close < open_
            and ema_fast < ema_slow
            and self.rsi_sell_min <= rsi <= self.rsi_sell_max
            and trend == "SELL"
            and blocked != "SELL"
        )
        if self.strategy_mode in ("BREAKOUT", "BOTH"):
            if buy_ok:
                return "BUY", "breakout"
            if sell_ok:
                return "SELL", "breakout"

        # Pullback continuation: a completed candle retests the fast EMA and
        # closes back with the higher-timeframe trend.  This avoids chasing a
        # breakout and is evaluated only after the candle is final.
        pullback_buy = (
            float(signal_bar["low"]) <= ema_fast <= close
            and close > open_ and ema_fast > ema_slow
            and 45.0 <= rsi <= 65.0 and trend == "BUY" and blocked != "BUY"
        )
        pullback_sell = (
            float(signal_bar["high"]) >= ema_fast >= close
            and close < open_ and ema_fast < ema_slow
            and 35.0 <= rsi <= 55.0 and trend == "SELL" and blocked != "SELL"
        )
        if self.strategy_mode in ("PULLBACK", "BOTH"):
            if pullback_buy:
                return "BUY", "pullback"
            if pullback_sell:
                return "SELL", "pullback"
        return None

    def daily_loss_limit_hit(self) -> bool:
        """Protect the account by stopping new entries after the daily loss cap."""
        if self.max_daily_loss <= 0.0:
            return False
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, now)
        if deals is None:
            return True  # Never trade blind if history is unavailable.
        realised = sum(
            float(d.profit) + float(d.commission) + float(d.swap)
            for d in deals
            if d.magic == self.magic and d.symbol == self.symbol and d.entry == mt5.DEAL_ENTRY_OUT
        )
        if realised <= -self.max_daily_loss:
            print(f"[RISK] Daily realised P/L={realised:.2f}; loss cap={self.max_daily_loss:.2f}")
            return True
        return False

    def entry_risk_limit_hit(self) -> bool:
        """Stop new entries after too many trades or consecutive losing exits today."""
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, now)
        if deals is None:
            print("[RISK] Cannot read today's deal history; new entry blocked")
            return True
        closed = sorted(
            (
                d for d in deals
                if d.magic == self.magic and d.symbol == self.symbol and d.entry == mt5.DEAL_ENTRY_OUT
            ),
            key=lambda d: d.time,
        )
        if self.max_trades_per_day > 0 and len(closed) >= self.max_trades_per_day:
            print(f"[RISK] Daily trade limit reached: {len(closed)}/{self.max_trades_per_day}")
            return True
        if self.max_consecutive_losses > 0:
            consecutive_losses = 0
            for deal in reversed(closed):
                net_profit = float(deal.profit) + float(deal.commission) + float(deal.swap)
                if net_profit >= 0.0:
                    break
                consecutive_losses += 1
            if consecutive_losses >= self.max_consecutive_losses:
                print(f"[RISK] Consecutive-loss limit reached: {consecutive_losses}")
                return True
        return False

    def _closed_bar_time(self, df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if df is None or len(df) < 2:
            return None
        return df.iloc[-2]["time"]

    def spread_too_wide(self, tick) -> bool:
        """Не торговать при широком спреде — стоп почти сразу в рынке."""
        info = mt5.symbol_info(self.symbol)
        if tick is None or info is None:
            return True
        spread = abs(float(tick.ask) - float(tick.bid))
        limit = self.max_spread_points * float(info.point)
        if spread > limit:
            print(
                f"[ПРОПУСК] Спред {spread:.{info.digits}f} > лимита "
                f"{limit:.{info.digits}f} ({self.max_spread_points} пунктов)"
            )
            return True
        return False

    def _closed_deals(self) -> list:
        now = int(time.time())
        deals = mt5.history_deals_get(now - 7 * 24 * 3600, now)
        if deals is None:
            return []
        closed = [
            d
            for d in deals
            if d.magic == self.magic
            and d.symbol == self.symbol
            and d.entry == mt5.DEAL_ENTRY_OUT
        ]
        closed.sort(key=lambda d: d.time)
        return closed

    def learned_block_direction(self) -> Optional[str]:
        """Если два последних убытка в одну сторону — не повторяем этот вход."""
        closed = self._closed_deals()
        if len(closed) < 2:
            return None
        last_two = closed[-2:]
        if last_two[0].profit >= 0 or last_two[1].profit >= 0:
            return None
        types = {int(d.type) for d in last_two}
        if types == {mt5.DEAL_TYPE_SELL}:
            print("[ОБУЧЕНИЕ] Два убыточных BUY подряд — следующий BUY пропускаем")
            return "BUY"
        if types == {mt5.DEAL_TYPE_BUY}:
            print("[ОБУЧЕНИЕ] Два убыточных SELL подряд — следующий SELL пропускаем")
            return "SELL"
        return None

    def adaptive_sl_mult(self) -> float:
        """Keep a fixed ATR stop multiplier; losing streaks must not increase risk."""
        return self.atr_sl_mult

    def in_cooldown(self) -> bool:
        """После профита почти сразу; после стопа — короткая пауза 15с."""
        closed = self._closed_deals()
        if not closed:
            return False
        last = closed[-1]
        elapsed = int(time.time()) - int(last.time)
        wait = self.cooldown_win_sec if last.profit >= 0 else self.cooldown_loss_sec
        if elapsed < wait:
            print(
                f"[ПРОПУСК] Пауза {wait}с после сделки ({elapsed}с), "
                f"profit={last.profit:.2f}"
            )
            return True
        return False

    def _filling_type(self, symbol_info: mt5.SymbolInfo) -> int:
        """Режим исполнения, который поддерживает брокер для символа."""
        mode = int(symbol_info.filling_mode)
        # Биты filling_mode: 1 = FOK, 2 = IOC, 4 = RETURN
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _normalize_volume(self, symbol_info: mt5.SymbolInfo, volume: float) -> float:
        """Приводим лот к шагу и лимитам брокера."""
        step = float(symbol_info.volume_step) if symbol_info.volume_step else 0.01
        vmin = float(symbol_info.volume_min)
        vmax = float(symbol_info.volume_max)
        steps = round(volume / step)
        volume = steps * step
        volume = max(vmin, min(vmax, volume))
        # Количество знаков после запятой у шага лота
        step_decimals = max(0, str(step)[::-1].find("."))
        return round(volume, step_decimals)

    def has_open_position(self) -> bool:
        """Есть ли уже позиция по символу с нашим magic."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return False
        return any(pos.magic == self.magic for pos in positions)

    def _bot_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.magic]

    def manage_trailing(self) -> None:
        """Безубыток и трейлинг SL — типичная логика скальперских EA."""
        info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        point = float(info.point)
        digits = int(info.digits)
        min_dist = self._min_stop_distance(info, tick)
        be_trigger = self.trail_be_points * point
        trail_start = self.trail_start_points * point
        trail_step = self.trail_step_points * point
        bid = float(tick.bid)
        ask = float(tick.ask)

        for pos in self._bot_positions():
            new_sl = float(pos.sl) if pos.sl else 0.0
            if pos.type == mt5.POSITION_TYPE_BUY:
                profit = bid - float(pos.price_open)
                if profit >= be_trigger:
                    be_sl = round(float(pos.price_open) + min_dist, digits)
                    if be_sl > new_sl:
                        new_sl = be_sl
                if profit >= trail_start:
                    trail_sl = round(bid - trail_step, digits)
                    if trail_sl > new_sl:
                        new_sl = trail_sl
            elif pos.type == mt5.POSITION_TYPE_SELL:
                profit = float(pos.price_open) - ask
                if profit >= be_trigger:
                    be_sl = round(float(pos.price_open) - min_dist, digits)
                    if new_sl == 0.0 or be_sl < new_sl:
                        new_sl = be_sl
                if profit >= trail_start:
                    trail_sl = round(ask + trail_step, digits)
                    if new_sl == 0.0 or trail_sl < new_sl:
                        new_sl = trail_sl
            else:
                continue

            old_sl = round(float(pos.sl), digits) if pos.sl else 0.0
            new_sl = round(new_sl, digits)
            if new_sl == 0.0 or new_sl == old_sl:
                continue
            result = mt5.order_send(
                {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": float(pos.tp),
                }
            )
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[ТРЕЙЛ] ticket={pos.ticket} SL {old_sl} → {new_sl} profit_dist={profit:.{digits}f}")
            elif result is not None:
                print(f"[ТРЕЙЛ] не обновлён retcode={result.retcode} {result.comment}")

    def close_expired_positions(self) -> None:
        """Close bot positions that outlive the configured scalping time limit."""
        if self.max_hold_bars <= 0:
            return
        max_age = self.max_hold_bars * TIMEFRAME_SECONDS[self.timeframe_name]
        info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        for pos in self._bot_positions():
            age = int(time.time()) - int(pos.time)
            if age < max_age:
                continue
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "position": pos.ticket,
                "volume": float(pos.volume),
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "price": float(tick.bid) if is_buy else float(tick.ask),
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "GoldBot time exit",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_type(info),
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[TIME EXIT] ticket={pos.ticket} age={age}s limit={max_age}s")
            elif result is None:
                print(f"[TIME EXIT] failed: {mt5.last_error()}")
            else:
                print(f"[TIME EXIT] rejected retcode={result.retcode} {result.comment}")

    def _min_stop_distance(self, info: mt5.SymbolInfo, tick: mt5.Tick) -> float:
        """Минимальный отступ SL/TP: брокер + спред + запас (иначе 10016 Invalid stops)."""
        point = float(info.point)
        spread = abs(float(tick.ask) - float(tick.bid))
        broker_min = max(int(info.trade_stops_level), int(info.trade_freeze_level)) * point
        buffer = self.stops_buffer_points * point
        return max(broker_min, spread + buffer, point)

    def open_order(self, direction: str, df: pd.DataFrame, strategy: str) -> bool:
        """Рыночный ордер BUY по ask / SELL по bid со SL и TP в пунктах."""
        tick = mt5.symbol_info_tick(self.symbol)
        info = mt5.symbol_info(self.symbol)
        if tick is None or info is None:
            print(f"[ОШИБКА] Нет тика/инфо по {self.symbol}: {mt5.last_error()}")
            return False

        point = float(info.point)
        digits = int(info.digits)
        volume = self._normalize_volume(info, self.lot)
        min_dist = self._min_stop_distance(info, tick)
        atr = float(df["atr"].iloc[-2]) if len(df) >= 2 else 0.0
        sl_mult = self.adaptive_sl_mult()
        sl_dist = max(self.sl_points * point, atr * sl_mult, min_dist)
        tp_dist = max(
            self.tp_points * point,
            atr * self.atr_tp_mult,
            sl_dist * self.min_risk_reward,
        )
        print(
            f"[УРОВНИ] ATR={atr:.{digits}f} sl_mult={sl_mult:.2f} "
            f"sl_dist={sl_dist:.{digits}f} tp_dist={tp_dist:.{digits}f} "
            f"(риск/прибыль ~ {tp_dist / sl_dist:.2f})"
        )

        bid = float(tick.bid)
        ask = float(tick.ask)

        if direction == "BUY":
            # Закрытие BUY по bid: SL ниже bid, TP выше ask
            order_type = mt5.ORDER_TYPE_BUY
            price = ask
            sl = bid - sl_dist
            tp = ask + tp_dist
        elif direction == "SELL":
            # Закрытие SELL по ask: SL выше ask, TP ниже bid
            order_type = mt5.ORDER_TYPE_SELL
            price = bid
            sl = ask + sl_dist
            tp = bid - tp_dist
        else:
            print(f"[ОШИБКА] Неизвестное направление ордера: {direction}")
            return False

        sl = round(sl, digits)
        tp = round(tp, digits)
        price = round(price, digits)

        last_close = float(df.iloc[-1]["close"])
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"GoldBot {strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(info),
        }

        result = mt5.order_send(request)
        if result is None:
            print(f"[ОШИБКА] order_send вернул None: {mt5.last_error()}")
            return False

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        status = "ИСПОЛНЕН" if ok else "ОТКЛОНЁН"
        print(
            f"[{status}] {direction} {self.symbol} vol={volume} "
            f"price={price} sl={sl} tp={tp} | "
            f"retcode={result.retcode} comment={result.comment} | "
            f"close_свечи={last_close:.5f} deal={result.deal} order={result.order}"
        )
        if not ok:
            print(f"[ДЕТАЛИ] last_error={mt5.last_error()} request={request}")
        return ok

    def run(self) -> None:
        """Главный цикл: каждые CHECK_INTERVAL секунд сигнал и при необходимости ордер."""
        if not self.connect():
            sys.exit(1)

        print(
            f"[СТАРТ] {self.symbol} {self.timeframe_name}+{self.trend_tf_name} | "
            f"EMA {self.ema_fast_period}/{self.ema_slow_period} RSI={self.rsi_period} | "
            f"пауза после стопа {self.cooldown_loss_sec}с"
        )
        print("Остановка: Ctrl+C")

        try:
            while True:
                try:
                    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

                    if self.has_open_position():
                        tick = mt5.symbol_info_tick(self.symbol)
                        price_txt = f"bid={tick.bid} ask={tick.ask}" if tick else "нет тика"
                        print(f"[{now}] {price_txt} | позиция открыта, ведём трейлинг")
                        self.manage_trailing()
                        self.close_expired_positions()
                        time.sleep(self.check_interval)
                        continue

                    if self.in_cooldown():
                        time.sleep(self.check_interval)
                        continue

                    if not self.in_trading_session():
                        print(f"[{now}] [ПРОПУСК] вне сессии {self.session_start_hour}:00-{self.session_end_hour}:00 UTC")
                        time.sleep(self.check_interval)
                        continue

                    if self.daily_loss_limit_hit():
                        time.sleep(self.check_interval)
                        continue

                    if self.entry_risk_limit_hit():
                        time.sleep(self.check_interval)
                        continue

                    df = self.get_data()
                    if df is None:
                        time.sleep(self.check_interval)
                        continue

                    bar_time = self._closed_bar_time(df)
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is not None:
                        price_txt = f"bid={tick.bid} ask={tick.ask}"
                    else:
                        price_txt = f"close={float(df.iloc[-1]['close']):.5f}"

                    signal = self.check_signal(df)
                    signal_text = f"{signal[0]} ({signal[1]})" if signal else None
                    print(f"[{now}] {self.symbol} {price_txt} | сигнал={signal_text}")

                    if signal is None:
                        time.sleep(self.check_interval)
                        continue

                    if bar_time is not None and bar_time == self._last_signal_bar:
                        print("[ПРОПУСК] По этой закрытой свече уже пытались входить")
                        time.sleep(self.check_interval)
                        continue

                    if self.spread_too_wide(tick):
                        time.sleep(self.check_interval)
                        continue

                    self._last_signal_bar = bar_time
                    self.open_order(signal[0], df, signal[1])

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"[ИСКЛЮЧЕНИЕ] {type(exc).__name__}: {exc}")

                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\n[СТОП] Остановка по Ctrl+C")
        finally:
            if self._connected:
                mt5.shutdown()
                self._connected = False
                print("[OK] Соединение с MT5 закрыто")


if __name__ == "__main__":
    bot = GoldBot()
    bot.run()
