import os
import re
import glob
import pandas as pd

from .schema import CN_TO_EN, REQUIRED_SNAPSHOT_COLS


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    rename = {c: CN_TO_EN[c] for c in df.columns if c in CN_TO_EN}
    df = df.rename(columns=rename)

    if "code" in df.columns:
        df["code"] = (
            df["code"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )

    return df


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    """
    支持：
    1991/4/3
    2026-02-27
    20260227
    """
    s = series.astype(str).str.strip()

    dt = pd.to_datetime(s, errors="coerce")

    mask = dt.isna() & s.str.match(r"^\d{8}$")
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s[mask], format="%Y%m%d", errors="coerce")

    if dt.isna().any():
        bad = s[dt.isna()].head(5).tolist()
        raise ValueError(f"无法解析日期: {bad}")

    return dt.dt.strftime("%Y%m%d")


def _trade_date_from_filename(path: str):
    name = os.path.basename(path)

    m = re.search(r"daily_(\d{4})-(\d{2})-(\d{2})", name)

    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    return None


def load_daily_snapshot(path: str):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, encoding="utf-8-sig")

    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_SNAPSHOT_COLS if c not in df.columns]

    if missing:
        raise Exception(f"缺少字段 {missing}")

    trade_date = _trade_date_from_filename(path)

    if trade_date:
        df["trade_date"] = trade_date
    else:
        df["trade_date"] = _normalize_trade_date(df["trade_date"])

    for c in ["open", "close", "high", "low", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["code", "close", "amount"])

    return df


def load_history_for_code(history_dir: str, code: str):

    code6 = str(code).zfill(6)

    candidates = [
        os.path.join(history_dir, f"{code6}_daily.csv"),
        os.path.join(history_dir, f"T{code6}_daily.csv"),
        os.path.join(history_dir, f"T{code6[-5:]}_daily.csv"),
    ]

    path = None

    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        hits = glob.glob(os.path.join(history_dir, f"*{code6}*_daily.csv"))
        if hits:
            path = hits[0]

    if path is None:
        raise FileNotFoundError(f"找不到历史文件: {code6}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    df = _normalize_columns(df)

    df["trade_date"] = _normalize_trade_date(df["trade_date"])

    for c in ["open", "close", "high", "low", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["trade_date", "close", "amount"])

    df = df.sort_values("trade_date")

    return df