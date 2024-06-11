import pandas as pd
from typing import Callable, Tuple, List, Union
import numpy as np

def bigger(*s: pd.Series) -> pd.Series:
    '''
    s2 是否有大於 s1
    '''
    s1, s2 = s[0], s[1]
    return s1 < s2


def smaller(*s: pd.Series) -> pd.Series:
    '''
    s2 是否有小於 s1
    '''
    s1, s2 = s[0], s[1]
    return s1 > s2

def add(*s: pd.Series) -> pd.Series:
    '''
    return s1 + s2  
    '''
    s1, s2 = s[0], s[1]
    return s1 + s2

def multi(*s: pd.Series) -> pd.Series:
    '''
    return s1 + s2  
    '''
    s1, s2 = s[0], s[1]
    return s1 * s2


def crossover(*s: pd.Series) -> pd.Series:
    '''
    s2 是否有向上 crossover s1
    '''
    s1, s2 = s[0], s[1]
    # 找到兩個序列的交集時間區段
    common_dates = s1.index.intersection(s2.index)
    # 將 s1 和 s2 限制在共有的時間區段
    s1_common = s1.loc[common_dates]
    s2_common = s2.loc[common_dates]
    diff = s2_common - s1_common
    crossover_points = (diff.shift(1) < 0) & (diff > 0)
    # 檢查交叉：查找符號改變的位置
    return crossover_points

def ema(data: pd.Series, period: int) -> pd.Series:
    """
    計算指數移動平均線（EMA）

    :param data: 序列數據
    :param period: 計算週期
    :return: EMA 序列
    """
    return data.ewm(span=period, adjust=False).mean()

def atr(data: pd.DataFrame, period: int) -> pd.Series:
    """
    計算平均真實範圍（ATR）

    :param high: 高價序列
    :param low: 低價序列
    :param close: 收盤價序列
    :param period: 計算週期
    :return: ATR 序列
    """
    high, low, close = data.High, data.Low, data.Close
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def keltner_channel(data: pd.DataFrame, period: int, multiplier: float):
    """
    計算凱特上通道（Keltner Channel）

    :param high: 高價序列
    :param low: 低價序列
    :param close: 收盤價序列
    :param period: 計算週期，默認為 20
    :param multiplier: ATR 乘數，默認為 2.0
    :return: 上通道、下通道、中線序列
    """
    mid = ema(data.Close, period)
    atr_value = atr(data.High, data.Low, data.Close, period)
    upper = mid + (multiplier * atr_value)
    
    return upper


def k_value(data: pd.DataFrame, period: int) -> pd.Series:
    """
    計算 KD 指標的 K 值

    :param high: 高價序列
    :param low: 低價序列
    :param close: 收盤價序列
    :param period: 計算週期
    :return: K 值序列
    """
    low_min = data.Low.rolling(window=period).min()
    high_max = data.High.rolling(window=period).max()
    rsv = (data.Close - low_min) / (high_max - low_min) * 100

    k = rsv.ewm(com=2).mean()

    return k


def d_value(data: pd.DataFrame, period: int = 3) -> pd.Series:
    low_min = data.Low.rolling(window=period).min()
    high_max = data.High.rolling(window=period).max()
    rsv = (data.Close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2).mean()
    # 計算 D 值
    d = k.ewm(span=period, adjust=False).mean()
    return d

def zscore(data: pd.Series, period: int) -> pd.Series:
    """
    計算滾動 Z 分數

    :param data: 序列數據
    :param period: 計算週期
    :return: 滾動 Z 分數序列
    """
    rolling_mean = data.rolling(window=period).mean()
    rolling_std = data.rolling(window=period).std()
    z_scores = (data - rolling_mean) / rolling_std
    return z_scores

def ts_max(data: pd.Series, period: int) -> pd.Series:
    """
    計算滾動最大值

    :param data: 序列數據
    :param period: 計算週期
    :return: 滾動最大值序列
    """
    return data.rolling(window=period).max()

def ts_min(data: pd.Series, period: int) -> pd.Series:
    """
    計算滾動最小值

    :param data: 序列數據
    :param period: 計算週期
    :return: 滾動最小值序列
    """
    return data.rolling(window=period).min()

def moving_average(*data) -> pd.Series:
    s1, l = data[0], data[1]
    return s1.rolling(l).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    計算相對強弱指數（RSI）

    :param close: 收盤價序列
    :param period: 計算週期
    :return: RSI 序列
    """
    period = max(period, 2)
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.fillna(100)  # 當 avg_loss 為 0 時，rs 無窮大，RSI 設為 100

    return rsi


def count(*data) -> pd.Series:
    '''
    count bigger than 0
    '''
    s1, l = data[0], data[1]
    return (s1 > 0).rolling(l).sum()


def shift(*data) -> pd.Series:
    '''
    將 s1 shift l 個單位
    '''
    s1, l = data[0], data[1]
    return s1.shift(l)

def macd(data: pd.Series, short_period: int, long_period: int, signal_period: int) -> pd.DataFrame:
    """
    計算 MACD 線、信號線和 MACD 柱狀圖

    :param data: 序列數據
    :param short_period: 短期 EMA 的週期
    :param long_period: 長期 EMA 的週期
    :param signal_period: 信號線的週期
    :return: 包含 MACD 線、信號線和 MACD 柱狀圖的 DataFrame
    """
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_histogram