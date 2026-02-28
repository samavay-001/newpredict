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


def compute_features(hist: pd.DataFrame, asof: str, window: int = 60) -> dict:
    df = hist[hist["trade_date"] <= asof].copy()
    df = df.tail(window)

    if len(df) < 30:
        raise ValueError("历史数据不足（<30）")

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    amount = df["amount"]

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

    # 近20日涨停次数
    if "pct_chg" in df.columns and df["pct_chg"].notna().sum() > 0:
        limit_ups_20 = int((df["pct_chg"].tail(20) >= 9.5).sum())
    else:
        pct = close.pct_change() * 100
        limit_ups_20 = int((pct.tail(20) >= 9.5).sum())

    # 分歧/冲高回落：上影线比例 + 收盘强度
    rng = max(last_high - last_low, 1e-9)
    upper_shadow = last_high - max(last_close, last_open)
    upper_shadow_ratio = float(upper_shadow / rng)

    close_strength = float((last_close - last_low) / rng)  # 越接近1越强

    # 当天是否涨停（粗判）：pct_chg >= 9.5
    if "pct_chg" in df.columns and pd.notna(df["pct_chg"].iloc[-1]):
        is_limit_up_today = 1.0 if float(df["pct_chg"].iloc[-1]) >= 9.5 else 0.0
    else:
        is_limit_up_today = 0.0

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
        "limit_ups_20": limit_ups_20,
        "upper_shadow_ratio": upper_shadow_ratio,
        "close_strength": close_strength,
        "is_limit_up_today": is_limit_up_today,
        "hist_len": int(len(df)),
    }