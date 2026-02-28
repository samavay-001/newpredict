import pandas as pd


def phase_a_filter(snapshot: pd.DataFrame,
                   min_amount: float = 2e8,
                   min_close: float = 2.0,
                   max_close: float = 200.0,
                   min_turnover: float | None = None) -> pd.DataFrame:
    """
    Phase A（离线）
    - 硬过滤：成交额、价格区间、可选换手率
    - 强势K线：收盘位置、实体强度、上影线风险
    """
    df0 = snapshot.copy()

    # N0
    n0 = len(df0)

    # N1：基础清洗（缺失值）
    df = df0.dropna(subset=["code", "close", "amount"])
    n1 = len(df)

    # N2：流动性过滤
    df = df[df["amount"] >= min_amount]
    n2 = len(df)

    # N3：价格过滤
    df = df[(df["close"] >= min_close) & (df["close"] <= max_close)]
    n3 = len(df)

    # N4：换手率（可选）
    if min_turnover is not None and "turnover_rate" in df.columns:
        df = df[df["turnover_rate"] >= min_turnover]
    n4 = len(df)

    # 强势K线条件
    rng = (df["high"] - df["low"]).replace(0, pd.NA)
    close_pos = (df["close"] - df["low"]) / rng               # 收盘在区间位置
    body = (df["close"] - df["open"]).abs()                   # 实体
    upper_shadow = (df["high"] - df[["close", "open"]].max(axis=1))  # 上影线

    df = df.assign(rng=rng, close_pos=close_pos, body=body, upper_shadow=upper_shadow)
    df = df.dropna(subset=["rng", "close_pos"])

    # 生产级阈值（比诊断版更严格）
    # - close_pos >= 0.75：收盘靠近高位
    # - body >= 0.30 * rng：实体足够强
    # - upper_shadow <= 0.25 * rng：避免冲高回落
    df = df[
        (df["close_pos"] >= 0.75) &
        (df["body"] >= 0.30 * df["rng"]) &
        (df["upper_shadow"] <= 0.25 * df["rng"])
    ]
    n5 = len(df)

    print(f"[PhaseA] N0={n0} -> N1={n1} -> N2={n2} -> N3={n3} -> N4={n4} -> N5(strong_candle)={n5}")

    keep = [c for c in ["trade_date", "code", "open", "close", "high", "low", "vol", "amount",
                        "pct_chg", "turnover_rate"] if c in df.columns]
    return df[keep].sort_values("amount", ascending=False)