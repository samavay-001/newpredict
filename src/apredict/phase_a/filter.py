import pandas as pd


def _norm_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code" in out.columns:
        out["code"] = (
            out["code"]
            .astype(str)
            .str.strip()
            .str.split(".").str[0]
            .str.zfill(6)
        )
    return out


def phase_a_filter(snapshot: pd.DataFrame,
                   min_amount: float = 2e8,
                   min_close: float = 2.0,
                   max_close: float = 200.0,
                   min_turnover: float | None = None,
                   exclude_st: bool = True) -> pd.DataFrame:
    """
    Phase A（生产版）
    - 硬过滤：成交额、价格区间、可选换手率
    - 质量过滤：ST过滤（如果有 name）、强势K线（收盘位置/实体/上影线）
    """
    df0 = snapshot.copy()
    n0 = len(df0)

    # N1：基础清洗（缺失值）
    df = df0.dropna(subset=[c for c in ["code", "open", "high", "low", "close", "amount"] if c in df0.columns])
    df = _norm_code(df)
    n1 = len(df)

    # N1.5：ST过滤（依赖 name）
    if exclude_st and "name" in df.columns:
        name_u = df["name"].astype(str).str.upper()
        df = df[~name_u.str.contains("ST", regex=False)]
    n1_5 = len(df)

    # N2：流动性过滤
    if "amount" in df.columns:
        df = df[df["amount"] >= min_amount]
    n2 = len(df)

    # N3：价格过滤
    if "close" in df.columns:
        df = df[(df["close"] >= min_close) & (df["close"] <= max_close)]
    n3 = len(df)

    # N4：换手率（可选）
    if min_turnover is not None and "turnover_rate" in df.columns:
        df = df[df["turnover_rate"] >= min_turnover]
    n4 = len(df)

    # 强势K线条件
    if all(c in df.columns for c in ["high", "low", "open", "close"]):
        rng = (df["high"] - df["low"]).replace(0, pd.NA)
        close_pos = (df["close"] - df["low"]) / rng
        body = (df["close"] - df["open"]).abs()
        upper_shadow = (df["high"] - df[["close", "open"]].max(axis=1))

        df = df.assign(rng=rng, close_pos=close_pos, body=body, upper_shadow=upper_shadow)
        df = df.dropna(subset=["rng", "close_pos"])

        # 生产级阈值
        df = df[
            (df["close_pos"] >= 0.75) &
            (df["body"] >= 0.30 * df["rng"]) &
            (df["upper_shadow"] <= 0.25 * df["rng"])
        ]
    n5 = len(df)

    print(f"[PhaseA] N0={n0} -> N1={n1} -> N1.5(ST)={n1_5} -> N2={n2} -> N3={n3} -> N4={n4} -> N5(strong)={n5}")

    keep = [c for c in [
        "trade_date", "ts_code", "code", "name",
        "open", "close", "high", "low", "vol", "amount",
        "pct_chg", "turnover_rate"
    ] if c in df.columns]
    if keep:
        df = df[keep]

    if "amount" in df.columns:
        return df.sort_values("amount", ascending=False)
    return df


def phase_a_firstboard_snapshot(snapshot: pd.DataFrame,
                                min_amount: float = 1e8,
                                min_close: float = 2.0,
                                max_close: float = 200.0,
                                exclude_st: bool = True,
                                verbose: bool = True) -> pd.DataFrame:
    """
    ✅ 首板 PhaseA（快照版，不依赖历史滚动特征）
    作用：先把全市场缩成“可能首板”的规模，减少 PhaseB 拉历史的成本。

    后续会在 PhaseB 算完 features 后再次做严格首板过滤（改造A的第二段）。
    """
    df0 = snapshot.copy()
    n0 = len(df0)

    df = df0.dropna(subset=[c for c in ["code", "open", "high", "low", "close", "amount"] if c in df0.columns]).copy()
    df = _norm_code(df)
    n1 = len(df)

    # ST过滤（name）
    if exclude_st and "name" in df.columns:
        name_u = df["name"].astype(str).str.upper()
        df = df[~name_u.str.contains(r"\bst\b", regex=True)]
        df = df[~name_u.str.contains(r"\*ST", regex=True)]
    n2 = len(df)

    # 流动性 + 价格
    if "amount" in df.columns:
        df = df[df["amount"] >= min_amount]
    if "close" in df.columns:
        df = df[(df["close"] >= min_close) & (df["close"] <= max_close)]
    n3 = len(df)

    # 快照级 K线质量（首板偏好：收盘靠上、上影短、实体不小）
    rng = (df["high"] - df["low"]).replace(0, pd.NA)
    close_pos = (df["close"] - df["low"]) / rng
    body = (df["close"] - df["open"]).abs()
    upper_shadow = (df["high"] - df[["close", "open"]].max(axis=1))

    df = df.assign(rng=rng, close_pos=close_pos, body=body, upper_shadow=upper_shadow)
    df = df.dropna(subset=["rng", "close_pos"])

    df = df[
        (df["close_pos"] >= 0.60) &
        (df["body"] >= 0.20 * df["rng"]) &
        (df["upper_shadow"] <= 0.40 * df["rng"])
    ]
    n4 = len(df)

    if verbose:
        print(f"[PhaseA-FirstBoard-Snapshot] N0={n0} -> N1={n1} -> N2(ST)={n2} -> N3(liq/price)={n3} -> N4(kline)={n4}")

    keep = [c for c in [
        "trade_date", "ts_code", "code", "name",
        "open", "close", "high", "low", "vol", "amount",
        "pct_chg", "turnover_rate"
    ] if c in df.columns]
    out = df[keep] if keep else df

    if "amount" in out.columns:
        out = out.sort_values("amount", ascending=False)
    return out