# scripts/run_daily.py
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# 让 Python 能找到 src/apredict（src-layout）
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from apredict.io.loader import load_daily_snapshot, load_history_for_code
from apredict.io.stock_meta import load_stock_meta, attach_stock_meta
from apredict.phase_a.filter import phase_a_firstboard_snapshot
from apredict.phase_b.rank import rank_candidates
from apredict.tracking.tracker import append_tracking

# ✅ live_tracking.xlsx 自动回填 + 追加
from apredict.tracking.live_tracker import (
    LiveTrackingConfig,
    update_realized_for_target_date,
    append_predictions,
)


# -----------------------------
# 交易日处理：严格下一个交易日（优先从 parquet 日历），失败则自然日+1跳周末
# -----------------------------
def _try_next_trade_date_from_parquet(trade_date: str, parquet_path: Path) -> Optional[str]:
    if not parquet_path.exists():
        return None
    try:
        import pyarrow.dataset as ds
    except Exception:
        return None

    try:
        dataset = ds.dataset(str(parquet_path), format="parquet")
        table = dataset.to_table(columns=["trade_date"])
        s = table.column("trade_date").to_pandas()
        cal = pd.Series(s).dropna().astype(int).drop_duplicates().sort_values().tolist()
        td = int(trade_date)
        for d in cal:
            if d > td:
                return f"{d:08d}"
        return None
    except Exception:
        return None


def get_next_trade_date(trade_date: str) -> str:
    parquet_path = ROOT / "data" / "cache_history" / "market_daily.parquet"
    nxt = _try_next_trade_date_from_parquet(trade_date, parquet_path)
    if nxt:
        return nxt

    dt = datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=1)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y%m%d")


# -----------------------------
# 展示增强：概率校准 + 星级
# -----------------------------
def calibrate_prob_by_rank(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty or score_col not in out.columns:
        out["prob_calibrated"] = np.nan
        return out

    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    anchors = [
        (1, 0.30),
        (5, 0.18),
        (10, 0.12),
        (20, 0.08),
        (50, 0.04),
        (100, 0.025),
        (200, 0.015),
        (500, 0.010),
    ]

    ranks = out["rank"].to_numpy(dtype=float)
    anchor_r = np.array([a[0] for a in anchors], dtype=float)
    anchor_p = np.array([a[1] for a in anchors], dtype=float)

    p = np.interp(np.minimum(ranks, anchor_r[-1]), anchor_r, anchor_p)
    tail = ranks > anchor_r[-1]
    if np.any(tail):
        r0, p0 = anchor_r[-1], anchor_p[-1]
        r1 = max(r0 + 1, float(len(out)))
        p1 = 0.005
        p[tail] = p0 + (p1 - p0) * ((ranks[tail] - r0) / (r1 - r0))

    out["prob_calibrated"] = np.clip(p, 0.0, 1.0)
    return out


def add_star_rating(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["star"] = ""
        return out

    if "rank" not in out.columns:
        if key_col in out.columns:
            out = out.sort_values(key_col, ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)

    def stars(r: int) -> str:
        if r <= 5:
            return "★★★★★"
        if r <= 10:
            return "★★★★☆"
        if r <= 20:
            return "★★★☆☆"
        if r <= 50:
            return "★★☆☆☆"
        return "★☆☆☆☆"

    out["star"] = out["rank"].astype(int).apply(stars)
    return out


def _pick_sort_key(df: pd.DataFrame) -> str:
    for c in ["ml_prob", "rank_score", "prob_calibrated", "prob_raw"]:
        if c in df.columns:
            return c
    return "rank"


# -----------------------------
# FirstBoard-Strict（二段严格池）
# -----------------------------
def firstboard_strict_filter(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if "atr_pct" in df.columns:
        mask &= df["atr_pct"].fillna(1.0) <= 0.08
    if "dist_to_hh20" in df.columns:
        mask &= df["dist_to_hh20"].fillna(0.0) >= 0.01
    if "amount_ratio" in df.columns:
        mask &= df["amount_ratio"].fillna(0.0) >= 1.10

    if "upper_shadow_ratio" in df.columns:
        mask &= df["upper_shadow_ratio"].fillna(1.0) <= 0.45
    if "close_strength" in df.columns:
        mask &= df["close_strength"].fillna(0.0) >= 0.45

    if "vol_ratio_10" in df.columns:
        mask &= df["vol_ratio_10"].fillna(0.0) >= 1.05
    if "pos_20" in df.columns:
        mask &= df["pos_20"].fillna(0.0).between(0.15, 0.95)

    if "is_limit_up_today" in df.columns:
        mask &= df["is_limit_up_today"].fillna(0).astype(int) == 0

    return df[mask].copy()


# -----------------------------
# 输出列映射（简版）
# -----------------------------
def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "rank": "排名",
        "code": "股票代码",
        "name": "股票名称",
        "star": "推荐星级",
        "rank_score": "综合评分",
        "ml_prob": "AI预测概率",
        "prob_calibrated": "预测概率",
        "close": "收盘价",
        "pct_chg": "涨跌幅%",
        "amount_ratio": "放量倍数",
        "breakout_20": "突破20日新高",
        "ret_5": "5日涨幅%",
        "ret_10": "10日涨幅%",
        "ret20": "20日涨幅%",
        "atr_pct": "ATR波动率%",
        "limit_ups_20": "20日涨停次数",
        "upper_shadow_ratio": "上影线比例",
        "close_strength": "收盘强度",
        "trade_date": "交易日期",
        "ts_code": "TS代码",
    }
    out = df.copy()
    cols = [c for c in out.columns if c in mapping]
    return out[cols].rename(columns={c: mapping[c] for c in cols})


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-path", required=True, help="data/raw/daily_YYYY-MM-DD.csv 或 data/raw/daily_YYYYMMDD.csv")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    daily_path = Path(args.daily_path)
    if not daily_path.exists():
        raise FileNotFoundError(f"daily_path not found: {daily_path}")

    # 从文件名解析 trade_date（优先）
    stem = daily_path.stem
    trade_date = None
    if "daily_" in stem:
        s = stem.split("daily_", 1)[1].replace("-", "")
        if len(s) == 8 and s.isdigit():
            trade_date = s
    if trade_date is None:
        tmp = pd.read_csv(daily_path, nrows=1)
        if "trade_date" in tmp.columns:
            trade_date = str(int(tmp.loc[0, "trade_date"])).zfill(8)
        else:
            raise ValueError("无法解析 trade_date：文件名和内容都不包含 trade_date")

    # 1) 当日快照
    snapshot = load_daily_snapshot(str(daily_path))

    # 2) 股票 meta（补 name/ts_code 等）
    meta = load_stock_meta(str(ROOT / "data" / "meta" / "stock_basic.csv"))
    snapshot = attach_stock_meta(snapshot, meta)

    # ✅ LiveTracking：先回填 target_date==today
    live_cfg = LiveTrackingConfig(xlsx_path=Path("data/output/live_tracking.xlsx"))
    try:
        update_realized_for_target_date(snapshot_today=snapshot, today_trade_date=trade_date, cfg=live_cfg, verbose=True)
    except Exception as e:
        print(f"[LiveTracking] 回填异常（不中断主流程）：{type(e).__name__}: {e}")

    # 3) Phase A：首板快照过滤
    universe = phase_a_firstboard_snapshot(snapshot, min_amount=1e8, verbose=True)

    # 4) Phase B/C：特征+排序（注意：你的本地 rank_candidates 返回顺序是 features, pred）
    history_dir = str(ROOT / "data" / "history")
    features, pred = rank_candidates(
        universe,
        lambda code: load_history_for_code(history_dir, code),
        trade_date,
        args.topk,
    )

    # 5) FirstBoard-Strict（二段严格池）
    if features is not None and not features.empty:
        strict_pool = firstboard_strict_filter(features)
        if not strict_pool.empty:
            sort_key = _pick_sort_key(strict_pool)
            pred = strict_pool.sort_values(sort_key, ascending=False).head(args.topk).copy()
            pred["rank"] = np.arange(1, len(pred) + 1)
            print(f"[FirstBoard-Strict] pool_after_features={len(strict_pool)}  top{args.topk} by {sort_key}")
        else:
            print("[FirstBoard-Strict] pool_after_features=0  skip")

    if pred is None or pred.empty:
        print("没有生成预测结果 pred（请检查 PhaseB/PhaseC）")
        return

    # 6) prob_calibrated（按排序键校准）
    sort_key = _pick_sort_key(pred)
    pred = calibrate_prob_by_rank(pred, score_col=sort_key)
    print(f"[Calibrate] prob_calibrated by {sort_key}")

    # 7) 星级
    pred = add_star_rating(pred, key_col=sort_key)

    # 8) 输出目录
    out_dir = ROOT / "data" / "processed" / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_path = out_dir / "universe.csv"
    features_path = out_dir / "features.csv"
    pred_path = out_dir / "predictions.csv"
    pred_cn_path = out_dir / "predictions_中文.csv"

    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
    if features is not None and not features.empty:
        features.to_csv(features_path, index=False, encoding="utf-8-sig")
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")

    pred_cn = to_chinese_columns(pred)
    pred_cn.to_csv(pred_cn_path, index=False, encoding="utf-8-sig")

    # 9) tracking.csv（你现有逻辑）
    append_tracking(pred, "output/tracking.csv")

    # ✅ LiveTracking：追加今天预测（目标日 = 下一个交易日）
    target_date = get_next_trade_date(trade_date)
    try:
        append_predictions(preds_topk=pred, predict_date=trade_date, target_date=target_date, cfg=live_cfg, verbose=True)
    except Exception as e:
        print(f"[LiveTracking] 追加异常（不中断主流程）：{type(e).__name__}: {e}")

    # 10) 总结
    print(f"完成：{trade_date}")
    print(f"universe: {len(universe)} -> {universe_path}")
    if features is not None and not features.empty:
        print(f"features: {len(features)} -> {features_path}")
    print(f"predictions(top{args.topk}): {len(pred)} -> {pred_path}")
    print(f"predictions_中文(top{args.topk}): {pred_cn_path}")
    print("tracking: output/tracking.csv")
    print(f"live_tracking: {live_cfg.xlsx_path} (sheet={live_cfg.sheet_name})")


if __name__ == "__main__":
    main()