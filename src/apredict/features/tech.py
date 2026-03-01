import pandas as pd


def _atr(df: pd.DataFrame, n: int = 20) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n).mean().iloc[-1]


def _calc_pct_chg_series(df: pd.DataFrame) -> pd.Series:
    """
    统一得到 pct_chg(%) 序列：
    - 若原始数据有 pct_chg 且有效 -> 用它
    - 否则用 close.pct_change()*100 兜底
    """
    if "pct_chg" in df.columns and df["pct_chg"].notna().sum() > 0:
        s = df["pct_chg"].astype(float)
    else:
        s = df["close"].pct_change() * 100.0
    return s


def compute_features(hist: pd.DataFrame, asof: str, window: int = 60) -> dict:
    """
    输入：单只股票历史K线（至少包含 trade_date/open/high/low/close/amount）
    输出：当日(asof)特征字典
    """
    df = hist[hist["trade_date"] <= asof].copy()
    df = df.tail(window)

    if len(df) < 30:
        raise ValueError("历史数据不足（<30）")

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    amount = df["amount"].astype(float)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    last_amount = float(amount.iloc[-1])

    ma20 = float(close.rolling(20).mean().iloc[-1])
    hh20 = float(close.rolling(20).max().iloc[-1])
    ll20 = float(close.rolling(20).min().iloc[-1])

    amount_ma20 = float(amount.rolling(20).mean().iloc[-1])
    amount_ratio = last_amount / max(amount_ma20, 1e-9)

    # 收益：5/10/20
    ret_5 = float(last_close / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    ret_10 = float(last_close / close.iloc[-11] - 1.0) if len(close) >= 11 else 0.0
    ret20 = float(last_close / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0

    # 突破：是否创20日新高（当日收盘>=过去20日最高）
    breakout_20 = 1.0 if last_close >= hh20 - 1e-12 else 0.0

    # 位置：距离 20 日高点（越小越好）
    dist_to_hh20 = (hh20 - last_close) / max(hh20, 1e-9)

    # ATR 与 ATR%
    atr20 = float(_atr(df, 20))
    atr_pct = atr20 / max(last_close, 1e-9)

    # 统一 pct_chg 序列（%）
    pct_series = _calc_pct_chg_series(df)

    # 涨停阈值（粗判）：9.5
    LIMIT_UP_TH = 9.5

    # 近20日涨停次数
    limit_ups_20 = int((pct_series.tail(20) >= LIMIT_UP_TH).sum())

    # 近5日涨停次数（首板过滤用）
    limit_ups_5 = int((pct_series.tail(5) >= LIMIT_UP_TH).sum())

    # 分歧/冲高回落：上影线比例 + 收盘强度
    rng = max(last_high - last_low, 1e-9)
    upper_shadow = last_high - max(last_close, last_open)
    upper_shadow_ratio = float(upper_shadow / rng)

    close_strength = float((last_close - last_low) / rng)  # 越接近1越强

    # 当天是否涨停（asof 当天是否涨停）：用于“昨日涨停”判断
    is_limit_up_today = 1.0 if float(pct_series.iloc[-1]) >= LIMIT_UP_TH else 0.0

    # 昨日是否涨停（这里等价于 is_limit_up_today：因为 asof=当天）
    is_limit_up_yday = is_limit_up_today

    # 当日涨跌幅（%）——很多地方会用到，给一个明确字段
    pct_chg = float(pct_series.iloc[-1]) if pd.notna(pct_series.iloc[-1]) else 0.0

    # 昨日是否涨停（用于避免追板）
    if "pct_chg" in df.columns and df["pct_chg"].notna().sum() >= 2:
        yday_pct = float(df["pct_chg"].iloc[-2])
        is_limit_up_yday = 1.0 if yday_pct >= 9.5 else 0.0
    else:
        is_limit_up_yday = 0.0

    return {
        "ma20": ma20,
        "hh20": hh20,
        "ll20": ll20,
        "atr20": atr20,
        "atr_pct": atr_pct,
        "amount_ratio": float(amount_ratio),

        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret20": ret20,

        "breakout_20": breakout_20,
        "dist_to_hh20": float(dist_to_hh20),

        "pct_chg": pct_chg,

        "limit_ups_20": limit_ups_20,
        "limit_ups_5": limit_ups_5,


        "upper_shadow_ratio": upper_shadow_ratio,
        "close_strength": close_strength,

        "is_limit_up_today": is_limit_up_today,
        "is_limit_up_yday": is_limit_up_yday,


        "hist_len": int(len(df)),
    }