import numpy as np
import pandas as pd

def _safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return a / b

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    需要列：open, high, low, close, pct_chg
    """
    out = df.copy()

    # 振幅
    out["range_pct"] = _safe_div(out["high"] - out["low"], out["close"]).fillna(0.0)

    # 收盘强度（越接近1越强）
    denom = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_strength"] = ((out["close"] - out["low"]) / denom).fillna(0.0).clip(0, 1)

    # 上影线比例
    upper = (out["high"] - np.maximum(out["open"], out["close"]))
    out["upper_shadow"] = _safe_div(upper, out["close"]).fillna(0.0).clip(0, 1)

    return out

def add_volume_ratio(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    需要列：code, vol
    """
    out = df.copy()
    out = out.sort_values(["code", "date"])
    g = out.groupby("code", group_keys=False)

    for w in windows:
        ma = g["vol"].rolling(w).mean().reset_index(level=0, drop=True)
        out[f"vol_ma_{w}"] = ma
        out[f"vol_ratio_{w}"] = (out["vol"] / ma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def add_price_position(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """
    需要列：code, high, low, close
    """
    out = df.copy()
    out = out.sort_values(["code", "date"])
    g = out.groupby("code", group_keys=False)

    hh = g["high"].rolling(window).max().reset_index(level=0, drop=True)
    ll = g["low"].rolling(window).min().reset_index(level=0, drop=True)
    out[f"hh{window}"] = hh
    out[f"ll{window}"] = ll

    denom = (hh - ll).replace(0, np.nan)
    out[f"pos_{window}"] = ((out["close"] - ll) / denom).fillna(0.5).clip(0, 1)

    out[f"dist_to_hh{window}"] = ((hh - out["close"]) / out["close"].replace(0, np.nan)).fillna(0.0).clip(0, 10)
    out[f"dist_to_ll{window}"] = ((out["close"] - ll) / out["close"].replace(0, np.nan)).fillna(0.0).clip(0, 10)

    return out

def add_limitup_flags(df: pd.DataFrame, limit10=9.8, limit20=19.8, board_col: str | None = None) -> pd.DataFrame:
    """
    生成 is_limit_up_today / is_limit_up_yday
    - 如果你没有 board 信息，就先按 9.8% 近似（会混入20%板块噪声，但仍能跑通）
    - 若有 board_col（如 market/board），可以在此按板块阈值区分
    """
    out = df.copy()

    if board_col and board_col in out.columns:
        # 你可以在这里根据 board_col 的取值改规则
        thr = np.where(out[board_col].astype(str).str.contains("20"), limit20, limit10)
        out["is_limit_up_today"] = (out["pct_chg"] >= thr).astype(int)
    else:
        out["is_limit_up_today"] = (out["pct_chg"] >= limit10).astype(int)

    out = out.sort_values(["code", "date"])
    out["is_limit_up_yday"] = out.groupby("code")["is_limit_up_today"].shift(1).fillna(0).astype(int)
    return out

def add_limitup_history(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    需要列：code, is_limit_up_today
    """
    out = df.copy()
    out = out.sort_values(["code", "date"])
    g = out.groupby("code", group_keys=False)

    for w in windows:
        out[f"limit_ups_{w}"] = g["is_limit_up_today"].rolling(w).sum().reset_index(level=0, drop=True).fillna(0).astype(int)

    # 距离上次涨停天数
    def _days_since_last_limitup(x: pd.Series) -> pd.Series:
        # x 是 0/1
        last = -10_000
        res = []
        for i, v in enumerate(x.values):
            if v == 1:
                last = i
                res.append(0)
            else:
                res.append(i - last if last >= 0 else 10_000)
        return pd.Series(res, index=x.index)

    out["days_since_limit_up"] = g["is_limit_up_today"].apply(_days_since_last_limitup).clip(0, 10_000)
    return out

def make_label_nextday_limitup(df: pd.DataFrame) -> pd.DataFrame:
    """
    label = 次日是否涨停（使用 is_limit_up_today shift(-1)）
    """
    out = df.copy()
    out = out.sort_values(["code", "date"])
    out["label_next_limitup"] = out.groupby("code")["is_limit_up_today"].shift(-1).fillna(0).astype(int)
    return out