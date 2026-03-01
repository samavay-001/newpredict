import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

HISTORY_DIR = Path("data/history")
OUT_PATH = Path("data/training/training.parquet")

# ========= 1) 中文列名 → 英文列名 映射 =========
RENAME_MAP = {
    "日期": "trade_date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover_rate",
}

# ========= 2) 涨停阈值（先按主板10%处理；后续可按 300/688 做 20%）=========
MAIN_LIMIT = 9.5   # 主板涨停粗阈值
GEM_LIMIT = 19.5   # 创业板/科创板粗阈值（后续启用）

def limit_threshold(code: str) -> float:
    """根据股票代码判断涨停阈值（简化版）"""
    code = str(code).strip()
    if code.startswith("300") or code.startswith("688"):
        return GEM_LIMIT
    return MAIN_LIMIT

# ========= 3) ATR计算 =========
def atr_pct(df: pd.DataFrame, n: int = 20) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(n).mean()
    return atr / close.replace(0, np.nan)

# ========= 4) 特征工程：尽量与你现有系统一致（面向“次日涨停”） =========
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()

    # 基础滚动
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    amount = df["amount"].astype(float)

    df["ma20"] = close.rolling(20).mean()
    df["hh20"] = close.rolling(20).max()
    df["ll20"] = close.rolling(20).min()

    # 放量倍数（成交额相对20日均值）
    df["amount_ma20"] = amount.rolling(20).mean()
    df["amount_ratio"] = amount / df["amount_ma20"].replace(0, np.nan)

    # 收益
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret20"] = close.pct_change(20)

    # 突破
    df["breakout_20"] = (close >= df["hh20"]).astype(int)

    # 位置：距离20日高点
    df["dist_to_hh20"] = (df["hh20"] - close) / df["hh20"].replace(0, np.nan)

    # ATR%
    df["atr_pct"] = atr_pct(df, 20)

    # K线强度：上影线比例 & 收盘强度
    open_ = df["open"].astype(float)
    rng = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    df["upper_shadow_ratio"] = (upper_shadow / rng).clip(0, 1)
    df["close_strength"] = ((close - low) / rng).clip(0, 1)

    # 近20日涨停次数（使用 pct_chg）
    df["limit_up_flag"] = df.apply(lambda r: 1 if float(r["pct_chg"]) >= limit_threshold(r["code"]) else 0, axis=1)
    df["limit_ups_20"] = df["limit_up_flag"].rolling(20).sum()

    return df

# ========= 5) 标签：次日是否涨停 =========
def create_label(df: pd.DataFrame) -> pd.DataFrame:
    # 次日涨跌幅（你数据里已有“涨跌幅”，但我们用 close 也能算；这里优先用 pct_chg 的 shift(-1)
    next_pct = df["pct_chg"].shift(-1).astype(float)
    # 每行按该行 code 确定阈值（同股恒定）
    thr = df["code"].astype(str).apply(limit_threshold)
    df["label"] = (next_pct >= thr).astype(int)
    return df

# ========= 6) 单文件处理 =========
def process_one_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        # 统一列名
        df = df.rename(columns=RENAME_MAP)

        required = ["trade_date", "code", "open", "close", "high", "low", "vol", "amount", "pct_chg"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺列: {missing}")

        # 日期统一：1991/4/3 -> 19910403
        # pandas.to_datetime 对这种格式很稳
        dt = pd.to_datetime(df["trade_date"], errors="coerce")
        df["trade_date"] = dt.dt.strftime("%Y%m%d")
        df = df.dropna(subset=["trade_date"])

        # 统一 code 为6位
        df["code"] = df["code"].astype(str).str.strip().str.split(".").str[0].str.zfill(6)

        # 转数值列
        for c in ["open", "close", "high", "low", "vol", "amount", "pct_chg"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close", "high", "low", "amount", "pct_chg"])

        if len(df) < 80:  # 太短的股票直接丢弃
            return None

        df = compute_features(df)
        df = create_label(df)

        # 清理：去掉滚动窗口产生的 NA & 最后一天无label
        df = df.dropna(subset=[
            "ma20", "hh20", "ll20", "amount_ratio", "ret_5", "ret_10", "ret20",
            "dist_to_hh20", "atr_pct", "upper_shadow_ratio", "close_strength", "limit_ups_20", "label"
        ])

        # 只保留训练需要列（避免文件巨大）
        keep = [
            "trade_date", "code",
            "open", "close", "high", "low",
            "vol", "amount", "pct_chg",
            "amount_ratio", "ret_5", "ret_10", "ret20",
            "breakout_20", "dist_to_hh20", "atr_pct",
            "limit_ups_20", "upper_shadow_ratio", "close_strength",
            "label"
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()

    except Exception as e:
        print(f"[skip] {path.name}: {type(e).__name__} - {e}")
        return None

def main():
    files = sorted(HISTORY_DIR.glob("*_daily.csv"))
    print("股票数量:", len(files))

    chunks = []
    skipped = 0

    for f in tqdm(files):
        d = process_one_file(f)
        if d is None or d.empty:
            skipped += 1
            continue
        chunks.append(d)

    if not chunks:
        raise SystemExit("没有生成任何训练数据，请检查列名映射/文件编码/数据质量")

    final = pd.concat(chunks, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)

    print(f"完成: {OUT_PATH}  rows={len(final)}  skipped_files={skipped}")

if __name__ == "__main__":
    main()