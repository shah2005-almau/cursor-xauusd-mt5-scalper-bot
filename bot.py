"""
Скальпинг-бот XAUUSD для MetaTrader 5.

Стратегия: пробой экстремума последних LOOKBACK_BARS свечей
с фильтром по EMA. Перед запуском должен быть открыт терминал MT5
и выполнен вход в торговый счёт.
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
        self._connected = False

    def connect(self) -> bool:
        """Подключение к MT5, проверка терминала и наличие символа XAUUSD."""
        if not mt5.initialize():
            print(
                f"[ОШИБКА] Не удалось подключиться к MT5: {mt5.last_error()}. "
                "Запустите терминал MetaTrader 5 и повторите попытку."
            )
            return False

        terminal = mt5.terminal_info()
        if terminal is None:
            print(f"[ОШИБКА] Терминал недоступен: {mt5.last_error()}")
            mt5.shutdown()
            return False

        if not terminal.connected:
            print("[ОШИБКА] Терминал запущен, но нет связи с торговым сервером.")
            mt5.shutdown()
            return False

        if not terminal.trade_allowed:
            print(
                "[ПРЕДУПРЕЖДЕНИЕ] Автоторговля в терминале выключена "
                "(Сервис → Настройки → Советники → разрешить алготорговлю)."
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

        # Включаем символ в «Обзоре рынка», иначе котировки могут не приходить
        if not mt5.symbol_select(self.symbol, True):
            print(
                f"[ОШИБКА] Символ {self.symbol} не найден или не выбран. "
                f"{mt5.last_error()}"
            )
            mt5.shutdown()
            return False

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
        print(
            f"[OK] Подключено к MT5. Счёт {account.login}, сервер {account.server}, "
            f"символ {self.symbol}, point={info.point}, digits={info.digits}"
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
        df["ema"] = calc_ema(df["close"].to_numpy(dtype=np.float64), self.ema_period)
        return df

    def check_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        BUY: close последней свечи выше max(high) предыдущих LOOKBACK свечей и выше EMA.
        SELL: close последней свечи ниже min(low) предыдущих LOOKBACK свечей и ниже EMA.
        Иначе None.
        """
        need = self.lookback + 1
        if df is None or len(df) < need:
            print(f"[ОШИБКА] Мало свечей для сигнала: нужно минимум {need}")
            return None

        last = df.iloc[-1]
        prev = df.iloc[-(self.lookback + 1) : -1]
        close = float(last["close"])
        ema_now = float(last["ema"])
        high_max = float(prev["high"].max())
        low_min = float(prev["low"].min())

        if close > high_max and close > ema_now:
            return "BUY"
        if close < low_min and close < ema_now:
            return "SELL"
        return None

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

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
            sl = price - self.sl_points * point
            tp = price + self.tp_points * point
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
            sl = price + self.sl_points * point
            tp = price - self.tp_points * point
        else:
            print(f"[ОШИБКА] Неизвестное направление ордера: {direction}")
            return False

        # Минимальная дистанция стопов у брокера
        stops_level = int(info.trade_stops_level)
        freeze_level = int(info.trade_freeze_level)
        min_distance = max(stops_level, freeze_level) * point
        if min_distance > 0:
            if direction == "BUY":
                if price - sl < min_distance:
                    sl = price - min_distance
                if tp - price < min_distance:
                    tp = price + min_distance
            else:
                if sl - price < min_distance:
                    sl = price + min_distance
                if price - tp < min_distance:
                    tp = price - min_distance

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
            f"lot={self.lot} SL={self.sl_points} TP={self.tp_points} EMA={self.ema_period}"
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

                    df = self.get_data()
                    if df is None:
                        time.sleep(self.check_interval)
                        continue

                    signal = self.check_signal(df)
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is not None:
                        price_txt = f"bid={tick.bid} ask={tick.ask}"
                    else:
                        price_txt = f"close={float(df.iloc[-1]['close']):.5f}"

                    print(f"[{now}] {self.symbol} {price_txt} | сигнал={signal}")

                    if signal in ("BUY", "SELL"):
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
