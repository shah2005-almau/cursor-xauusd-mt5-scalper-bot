"""
Скальпинг-бот XAUUSD для MetaTrader 5.

Стратегия: пробой закрытой свечи + EMA + ATR-стопы.
Перед запуском должен быть открыт терминал MT5 и выполнен вход в счёт.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
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


class GoldBot:
    """Торговый бот по золоту: подключение к MT5, сигналы и рыночные ордера."""

    def __init__(self) -> None:
        self.symbol = config.SYMBOL
        self.timeframe_name = config.TIMEFRAME
        self.lot = float(config.LOT)
        self.tp_points = int(config.TP_POINTS)
        self.sl_points = int(config.SL_POINTS)
        self.ema_period = int(config.EMA_PERIOD)
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
        self.min_body_points = int(config.MIN_CANDLE_BODY_POINTS)
        self.cooldown_sec = int(config.COOLDOWN_AFTER_TRADE_SEC)
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
        df["ema"] = calc_ema(closes, self.ema_period)
        df["atr"] = calc_atr(
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            closes,
            self.atr_period,
        )
        return df

    def check_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Сигнал только по ПОСЛЕДНЕЙ ЗАКРЫТОЙ свече (не по текущей формирующейся).
        BUY: пробой максимума, закрытие выше EMA, EMA растёт, бычья свеча.
        SELL: пробой минимума, закрытие ниже EMA, EMA падает, медвежья свеча.
        """
        # -1 — текущая незакрытая, -2 — последняя закрытая
        need = self.lookback + self.atr_period + 3
        if df is None or len(df) < need:
            print(f"[ОШИБКА] Мало свечей для сигнала: нужно минимум {need}")
            return None

        signal_bar = df.iloc[-2]
        prev = df.iloc[-(self.lookback + 2) : -2]
        ema_now = float(signal_bar["ema"])
        ema_prev = float(df.iloc[-3]["ema"])
        close = float(signal_bar["close"])
        open_ = float(signal_bar["open"])
        body = abs(close - open_)
        info = mt5.symbol_info(self.symbol)
        point = float(info.point) if info is not None else 0.01
        min_body = self.min_body_points * point

        if body < min_body:
            return None

        high_max = float(prev["high"].max())
        low_min = float(prev["low"].min())
        ema_up = ema_now > ema_prev
        ema_down = ema_now < ema_prev

        if close > high_max and close > ema_now and ema_up and close > open_:
            return "BUY"
        if close < low_min and close < ema_now and ema_down and close < open_:
            return "SELL"
        return None

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

    def in_cooldown(self) -> bool:
        """Пауза после закрытия позиции, чтобы не ловить серию стопов подряд."""
        now = int(time.time())
        deals = mt5.history_deals_get(now - 24 * 3600, now)
        if deals is None:
            return False
        closed = [
            d
            for d in deals
            if d.magic == self.magic
            and d.symbol == self.symbol
            and d.entry == mt5.DEAL_ENTRY_OUT
        ]
        if not closed:
            return False
        last = max(closed, key=lambda d: d.time)
        elapsed = now - int(last.time)
        if elapsed < self.cooldown_sec:
            left = self.cooldown_sec - elapsed
            print(
                f"[ПРОПУСК] Пауза после сделки ({elapsed}с), осталось {left}с. "
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

    def _min_stop_distance(self, info: mt5.SymbolInfo, tick: mt5.Tick) -> float:
        """Минимальный отступ SL/TP: брокер + спред + запас (иначе 10016 Invalid stops)."""
        point = float(info.point)
        spread = abs(float(tick.ask) - float(tick.bid))
        broker_min = max(int(info.trade_stops_level), int(info.trade_freeze_level)) * point
        buffer = self.stops_buffer_points * point
        return max(broker_min, spread + buffer, point)

    def open_order(self, direction: str, df: pd.DataFrame) -> bool:
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
        sl_dist = max(self.sl_points * point, atr * self.atr_sl_mult, min_dist)
        tp_dist = max(self.tp_points * point, atr * self.atr_tp_mult, sl_dist * 1.6)
        print(
            f"[УРОВНИ] ATR={atr:.{digits}f} sl_dist={sl_dist:.{digits}f} "
            f"tp_dist={tp_dist:.{digits}f} (риск/прибыль ~ {tp_dist / sl_dist:.2f})"
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
            "comment": "GoldBot scalp",
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
            f"[СТАРТ] Цикл {self.check_interval}с | {self.symbol} {self.timeframe_name} | "
            f"lot={self.lot} minSL={self.sl_points} minTP={self.tp_points} "
            f"EMA={self.ema_period} ATR x{self.atr_sl_mult}/{self.atr_tp_mult}"
        )
        print("Остановка: Ctrl+C")

        try:
            while True:
                try:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if self.has_open_position():
                        tick = mt5.symbol_info_tick(self.symbol)
                        price_txt = f"bid={tick.bid} ask={tick.ask}" if tick else "нет тика"
                        print(
                            f"[{now}] {price_txt} | позиция уже открыта, сигнал не ищем"
                        )
                        time.sleep(self.check_interval)
                        continue

                    if self.in_cooldown():
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
                    print(f"[{now}] {self.symbol} {price_txt} | сигнал={signal}")

                    if signal not in ("BUY", "SELL"):
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
                    self.open_order(signal, df)

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
