# src/apredict/features/feature_set.py
from __future__ import annotations

import numpy as np
import pandas as pd

DATE_COL = "trade_date"
CODE_COL = "code"


# =========================
# helpers
# =========================
def norm_trade_date(s: pd.Series) -> pd.Series:
    """
    统一日期为 YYYYMMDD 字符串，避免 2026-02-27 / 20260227 混用导致排序或对齐问题。
    """
    out = s.astype("string")
    out = out.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    dt = pd.to_datetime(out, errors="coerce", format="%Y%m%d")
    m = dt.notna()
    out2 = out.copy()
    out2[m] = dt[m].dt.strftime("%Y%m%d")
    return out2.astype(str)


def _safe_div(a, b, fill=0.0):
    """
    安全除法：自动处理 0 / inf / nan
    - a,b 可以是 Series 或标量
    """
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
        out = a / b
        out = out.replace([np.inf, -np.inf], np.nan).fillna(fill)
        return out
    else:
        bb = np.nan if b == 0 else b
        out = a / bb
        if isinstance(out, (float, np.floating)) and (not np.isfinite(out)):
            return fill
        return out


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature_set 需要列缺失: {missing}")


def _infer_is_20pct_board(df: pd.DataFrame) -> pd.Series:
    """
    识别是否 20% 涨跌幅板：优先 market_type，其次 ts_code + code 前缀。
    返回 bool Series。
    """
    if "market_type" in df.columns:
        mt = df["market_type"].astype(str)
        return mt.str.contains("创业板") | mt.str.contains("科创板")

    code = df.get(CODE_COL, pd.Series([""] * len(df))).astype(str).str.split(".").str[0].str.zfill(6)
    ts_code = df.get("ts_code", pd.Series([""] * len(df))).astype(str)

    is_20 = (
        (ts_code.str.contains("SZ", na=False) & code.str.startswith(("300", "301")))
        | (ts_code.str.contains("SH", na=False) & code.str.startswith("688"))
    )
    return is_20


def _days_since_last_one(x: pd.Series) -> pd.Series:
    """
    x 为 0/1 序列，返回距离上次 1 的天数（当天为0）。
    """
    last = None
    res = []
    for i, v in enumerate(x.values):
        if v == 1:
            last = i
            res.append(0)
        else:
            res.append(i - last if last is not None else 10000)
    return pd.Series(res, index=x.index)


def _calc_atr_panel(out: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    稳定版面板 ATR：
    - out 必须已按 [code, date] 排序
    - 返回与 out 同 index 的 atr(n) Series
    """
    g = out.groupby(CODE_COL, sort=False, group_keys=False)

    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    prev_close = g["close"].shift(1).astype(float)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # rolling mean of TR
    atr = (
        tr.groupby(out[CODE_COL], sort=False)
        .rolling(n, min_periods=max(2, n // 2))
        .mean()
        .reset_index(level=0, drop=True)
    )
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================
# online (single stock hist) feature
# =========================
def compute_features_for_hist(hist: pd.DataFrame, asof: str, window: int = 60) -> dict:
    """
    单股在线特征：与 add_features_panel 产出的列名保持一致。
    依赖列：trade_date/open/high/low/close/amount
    可选：vol, pct_chg, ts_code, code, market_type
    """
    _ensure_cols(hist, [DATE_COL, "open", "high", "low", "close", "amount"])

    df = hist.copy()

    # ✅ 必须有 code，否则 add_features_panel 会报错
    if CODE_COL not in df.columns:
        df[CODE_COL] = "000000"

    df[DATE_COL] = norm_trade_date(df[DATE_COL])
    asof = str(asof).replace("-", "").replace("/", "")

    df = df[df[DATE_COL] <= asof].sort_values(DATE_COL).reset_index(drop=True)
    df = df.tail(window).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("历史数据不足（<30）")

    df2 = add_features_panel(df)
    row = df2.iloc[-1]

    out = {}
    for c in FEATURE_COLS_DEFAULT:
        if c in df2.columns:
            v = row[c]
            if isinstance(v, (np.floating, float, int, np.integer)):
                out[c] = float(v)
            else:
                out[c] = v

    out["hist_len"] = int(len(df))
    return out


# =========================
# offline training panel feature
# =========================
def add_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    线下训练/批量特征：输入面板数据（多 code 多 trade_date），输出新增特征列。

    要求 df 至少包含：
      trade_date, code, open, high, low, close, amount
    可选：
      vol, pct_chg, ts_code, market_type
    """
    _ensure_cols(df, [DATE_COL, CODE_COL, "open", "high", "low", "close", "amount"])

    out = df.copy()
    out[DATE_COL] = norm_trade_date(out[DATE_COL])
    out[CODE_COL] = out[CODE_COL].astype(str).str.split(".").str[0].str.zfill(6)

    out = out.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    g = out.groupby(CODE_COL, group_keys=False, sort=False)

    # ---------- returns ----------
    out["ret_3"] = g["close"].pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret_5"] = g["close"].pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret_10"] = g["close"].pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret20"] = g["close"].pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["momentum_accel"] = (out["ret_5"] - out["ret_10"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---------- MA / slopes ----------
    ma5 = g["close"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    ma10 = g["close"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
    ma20 = g["close"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    out["ma5"] = ma5.ffill().fillna(0.0)
    out["ma10"] = ma10.ffill().fillna(0.0)
    out["ma20"] = ma20.ffill().fillna(0.0)

    out["ma5_slope"] = _safe_div(ma5 - ma5.shift(1), ma5.shift(1), fill=0.0)
    out["ma10_slope"] = _safe_div(ma10 - ma10.shift(1), ma10.shift(1), fill=0.0)
    out["ma20_slope"] = _safe_div(ma20 - ma20.shift(1), ma20.shift(1), fill=0.0)

    out["trend_stack_5_10_20"] = ((ma5 > ma10) & (ma10 > ma20)).astype(int)
    out["trend_strength"] = (out["ma5_slope"] + out["ma10_slope"] + out["ma20_slope"]).clip(-1.0, 1.0)

    # ---------- price position / hh ll ----------
    # ✅ 建议用 high/low 口径：更符合“阻力/突破”的语义
    hh20 = g["high"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    ll20 = g["low"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)

    out["hh20"] = hh20.ffill().fillna(0.0)
    out["ll20"] = ll20.ffill().fillna(0.0)

    out["breakout_20"] = (out["close"].astype(float) >= (hh20 - 1e-12)).astype(int)
    out["dist_to_hh20"] = _safe_div(hh20 - out["close"], hh20, fill=0.0).clip(0, 10)
    out["pos_20"] = _safe_div(out["close"] - ll20, (hh20 - ll20), fill=0.5).clip(0, 1)
    out["dist_to_ll20"] = _safe_div(out["close"] - ll20, out["close"], fill=0.0).clip(0, 10)

    hh60 = g["high"].rolling(60, min_periods=30).max().reset_index(level=0, drop=True)
    out["hh60"] = hh60.ffill().fillna(0.0)
    out["dist_to_hh60"] = _safe_div(hh60 - out["close"], hh60, fill=0.0).clip(0, 10)

    # ---------- amount ratios / accel ----------
    ama3 = g["amount"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    ama5 = g["amount"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    ama10 = g["amount"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
    ama20 = g["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    out["amount_ratio_3"] = _safe_div(out["amount"], ama3, fill=0.0)
    out["amount_ratio_5"] = _safe_div(out["amount"], ama5, fill=0.0)
    out["amount_ratio_10"] = _safe_div(out["amount"], ama10, fill=0.0)
    out["amount_ratio_20"] = _safe_div(out["amount"], ama20, fill=0.0)

    # 兼容你原来的 amount_ratio（默认用 20日口径）
    out["amount_ratio"] = out["amount_ratio_20"]

    out["amount_accel_3_10"] = _safe_div(out["amount_ratio_3"], out["amount_ratio_10"].replace(0, np.nan), fill=0.0)
    out["amount_accel_5_20"] = _safe_div(out["amount_ratio_5"], out["amount_ratio_20"].replace(0, np.nan), fill=0.0)

    # ---------- vol ratios / stability ----------
    if "vol" in out.columns:
        vma10 = g["vol"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
        out["vol_ratio_10"] = _safe_div(out["vol"], vma10, fill=0.0)
        out["vol_stability"] = (
            g["vol_ratio_10"]
            .rolling(5, min_periods=2)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["vol_ratio_10"] = 1.0
        out["vol_stability"] = 0.0

    # ---------- candle quality ----------
    rng = (out["high"] - out["low"]).astype(float)
    rng2 = rng.replace(0, np.nan)

    out["range_pct"] = _safe_div(rng, out["close"].astype(float), fill=0.0).clip(0, 10)
    out["close_strength"] = _safe_div((out["close"] - out["low"]).astype(float), rng2, fill=0.0).clip(0, 1)

    upper_shadow = (out["high"] - out[["open", "close"]].max(axis=1)).astype(float)
    out["upper_shadow_ratio"] = _safe_div(upper_shadow, rng2, fill=0.0).clip(0, 1)

    body = (out["close"] - out["open"]).abs().astype(float)
    out["body_ratio"] = _safe_div(body, rng2, fill=0.0).clip(0, 10)

    # ---------- pct_chg ----------
    if "pct_chg" not in out.columns or out["pct_chg"].isna().all():
        out["pct_chg"] = g["close"].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    else:
        out["pct_chg"] = out["pct_chg"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---------- limit up flags / history ----------
    is_20 = _infer_is_20pct_board(out)
    thr = np.where(is_20, 19.8, 9.8).astype(float)
    out["thr_limit"] = thr

    out["is_limit_up_today"] = (out["pct_chg"].values >= thr).astype(int)
    out["is_limit_up_yday"] = g["is_limit_up_today"].shift(1).fillna(0).astype(int)

    for w in (5, 10, 20):
        out[f"limit_ups_{w}"] = (
            g["is_limit_up_today"]
            .rolling(w, min_periods=max(2, w // 2))
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
            .astype(int)
        )

    out["days_since_limit_up"] = g["is_limit_up_today"].apply(_days_since_last_one).clip(0, 10000)

    # ---------- ATR / ATR% ----------
    out["atr20"] = _calc_atr_panel(out, n=20)
    out["atr_pct"] = _safe_div(out["atr20"], out["close"].astype(float), fill=0.0).clip(0, 10)

    return out


# =========================
# unified feature list (recommended)
# =========================
FEATURE_COLS_DEFAULT = [
    # price / basic
    "open", "close", "high", "low", "amount", "pct_chg",

    # returns
    "ret_3", "ret_5", "ret_10", "ret20", "momentum_accel",

    # MA / trend
    "ma5", "ma10", "ma20",
    "ma5_slope", "ma10_slope", "ma20_slope",
    "trend_stack_5_10_20", "trend_strength",

    # position / breakout
    "hh20", "ll20", "pos_20",
    "breakout_20", "dist_to_hh20", "dist_to_ll20",
    "hh60", "dist_to_hh60",

    # volatility
    "atr20", "atr_pct",

    # liquidity / volume
    "amount_ratio", "amount_ratio_3", "amount_ratio_5", "amount_ratio_10", "amount_ratio_20",
    "amount_accel_3_10", "amount_accel_5_20",
    "vol_ratio_10", "vol_stability",

    # candle quality
    "range_pct", "upper_shadow_ratio", "close_strength", "body_ratio",

    # behavior / limit-up
    "is_limit_up_today", "is_limit_up_yday",
    "limit_ups_5", "limit_ups_10", "limit_ups_20",
    "days_since_limit_up",
]