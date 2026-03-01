import sys
import os
import argparse
import time
import numpy as np
import pandas as pd

def calibrate_prob_by_rank(df: pd.DataFrame,
                           score_col: str,
                           top_anchors=None) -> pd.DataFrame:
    """
    截面校准：不改变排序，只把“分数/概率”映射成更像真实概率的数值，便于展示/沟通。
    - 输入 df：至少有 score_col
    - 输出 df：新增 prob_calibrated（0~1）

    默认锚点（经验值，可根据你长期回测再调）：
      Top1  -> 0.30
      Top5  -> 0.18
      Top10 -> 0.12
      Top20 -> 0.08
      Top50 -> 0.04
      其它  -> 0.01~0.03 左右（随排名递减）
    """
    if top_anchors is None:
        top_anchors = [
            (1, 0.30),
            (5, 0.18),
            (10, 0.12),
            (20, 0.08),
            (50, 0.04),
        ]

    out = df.copy()
    if out.empty or score_col not in out.columns:
        out["prob_calibrated"] = np.nan
        return out

    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(out)
    ranks = np.arange(1, n + 1)

    # rank->prob 分段插值：在锚点之间线性插值，尾部衰减到 floor
    ks = np.array([k for k, _ in top_anchors], dtype=float)
    ps = np.array([p for _, p in top_anchors], dtype=float)

    # 超过最大锚点的尾部概率地板（不要变成0）
    floor = 0.01

    prob = np.empty(n, dtype=float)

    for i, r0 in enumerate(ranks):
        r = int(r0)  # ✅ numpy -> python int，比较/max 都不再报黄

        if r <= int(ks[0]):
            prob[i] = float(ps[0])

        elif r >= int(ks[-1]):
            denom = max(int(n) - int(ks[-1]), 1)  # ✅ 全是 int
            t = (r - int(ks[-1])) / denom
            prob[i] = float(ps[-1]) * (1 - t) + float(floor) * t

        else:
            j = int(np.searchsorted(ks, r) - 1)

            k1, p1 = float(ks[j]), float(ps[j])
            k2, p2 = float(ks[j + 1]), float(ps[j + 1])

            denom = max(k2 - k1, 1.0)  # ✅ 全是 float
            t = (r - k1) / denom
            prob[i] = p1 * (1 - t) + p2 * t

    out["prob_calibrated"] = np.clip(prob, 0.0, 1.0)
    return out


def pick_sort_col(df: pd.DataFrame) -> str | None:
    """
    选择用于“展示概率校准”的排序字段（不改变你交易排序逻辑，只用于映射）
    """
    for c in ["rank_score", "ml_prob", "prob_raw"]:
        if c in df.columns:
            return c
    return None
# 让 Python 能找到 src/apredict（src-layout）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from apredict.io.loader import load_daily_snapshot, load_history_for_code
from apredict.io.stock_meta import load_stock_meta, attach_stock_meta
from apredict.phase_a.filter import phase_a_firstboard_snapshot  # ✅ 改造A
from apredict.phase_b.rank import rank_candidates
from apredict.tracking.tracker import append_tracking


def safe_to_csv(df, path, **kwargs):
    base, ext = os.path.splitext(path)
    for i in range(0, 30):
        p = path if i == 0 else f"{base}_{i}{ext}"
        try:
            df.to_csv(p, **kwargs)
            return p
        except PermissionError:
            time.sleep(0.2)
    raise PermissionError(f"文件被占用且多次重试失败：{path}（请关闭Excel/WPS/预览窗格）")


def stars_by_rank(rank: int, k: int) -> str:
    """
    分位星级：按TopK排名分星，适合交易展示
    - Top1: ★★★★★
    - Top2: ★★★★☆
    - Top3: ★★★☆☆
    - Top4: ★★☆☆☆
    - Top5+: ★☆☆☆☆
    如果 k>5，会按分位比例分配。
    """
    if k <= 1:
        return "★★★★★"

    # 先按比例分段
    q = rank / k  # 1/k ~ 1
    if q <= 0.20:
        return "★★★★★"
    if q <= 0.40:
        return "★★★★☆"
    if q <= 0.60:
        return "★★★☆☆"
    if q <= 0.80:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def to_chinese_columns_with_stars(df):
    mapping = {
        "trade_date": "交易日期",
        "ml_prob": "AI预测概率",
        "rank": "排名",
        "code": "股票代码",
        "rank_score": "综合评分",
        "prob_calibrated": "预测概率",
        "prob_raw": "原始概率",
        "ts_code": "TS代码",
        "name": "股票名称",

        "open": "开盘价",
        "close": "收盘价",
        "high": "最高价",
        "low": "最低价",
        "pct_chg": "涨跌幅%",
        "amount": "成交额",

        "amount_ratio": "放量倍数",
        "breakout_20": "突破20日新高",

        "ret_5": "5日涨幅%",
        "ret_10": "10日涨幅%",
        "ret20": "20日涨幅%",

        "atr_pct": "ATR波动率%",

        "limit_ups_20": "20日涨停次数",
        "upper_shadow_ratio": "上影线比例",
        "close_strength": "收盘强度",

        "hh20": "20日最高价",
        "ma20": "20日均线",
    }

    df_cn = df.rename(columns=mapping).copy()

    # ===== 推荐星级：改为“分位星级”（按排名/TopK）=====
    # 先确保有“排名”列（没有就根据综合评分/概率生成）
    if "排名" not in df_cn.columns:
        # 尝试用综合评分/AI概率生成排名
        if "综合评分" in df_cn.columns:
            df_cn["排名"] = df_cn["综合评分"].rank(ascending=False, method="first").astype(int)
        elif "AI预测概率" in df_cn.columns:
            df_cn["排名"] = df_cn["AI预测概率"].rank(ascending=False, method="first").astype(int)
        elif "预测概率" in df_cn.columns:
            df_cn["排名"] = df_cn["预测概率"].rank(ascending=False, method="first").astype(int)
        else:
            df_cn["排名"] = list(range(1, len(df_cn) + 1))

    k = int(df_cn["排名"].max()) if len(df_cn) else 5
    df_cn["推荐星级"] = df_cn["排名"].apply(lambda r: stars_by_rank(int(r), k))

    order = [
        "排名", "股票代码", "股票名称", "推荐星级",
        "综合评分", "预测概率", "AI预测概率",
        "收盘价", "涨跌幅%", "成交额",
        "放量倍数", "突破20日新高",
        "5日涨幅%", "10日涨幅%", "20日涨幅%",
        "ATR波动率%", "20日涨停次数",
        "上影线比例", "收盘强度",
    ]
    order = [c for c in order if c in df_cn.columns]
    rest = [c for c in df_cn.columns if c not in order]
    return df_cn[order + rest]


def strict_firstboard_filter_after_features(features_df, vol_ratio_min=1.3, pos20_max=0.80, shadow_max=0.06):
    """
    ✅ 改造A第二段：在 PhaseB 已算出滚动特征后，再做严格首板过滤。
    只在列存在时生效，保证兼容不同版本 features。
    """
    fb = features_df.copy()

    # 首板核心：近20日无涨停 + 今日/昨日非涨停
    if "limit_ups_20" in fb.columns:
        fb = fb[fb["limit_ups_20"] == 0]
    if "is_limit_up_today" in fb.columns:
        fb = fb[fb["is_limit_up_today"] == 0]
    if "is_limit_up_yday" in fb.columns:
        fb = fb[fb["is_limit_up_yday"] == 0]

    # 放量
    if "vol_ratio_10" in fb.columns:
        fb = fb[fb["vol_ratio_10"] >= vol_ratio_min]

    # 位置不能太高
    if "pos_20" in fb.columns:
        fb = fb[fb["pos_20"] <= pos20_max]

    # 冲高回落过滤
    if "upper_shadow_ratio" in fb.columns:
        fb = fb[fb["upper_shadow_ratio"] <= shadow_max]

    return fb


def main(daily_path: str, topk: int):
    # 1) 读取当日快照
    snapshot = load_daily_snapshot(daily_path)
    trade_date = str(snapshot["trade_date"].iloc[0])

    # ✅ 加载股票列表并补齐名称
    meta = load_stock_meta("data/meta/stock_basic.csv")
    snapshot = attach_stock_meta(snapshot, meta)

    # 2) Phase A（改造A第一段）：先快照首板过滤，缩小 universe
    universe = phase_a_firstboard_snapshot(snapshot, min_amount=1e8, verbose=True)

    # 3) Phase B：计算特征 + 排序（先按原逻辑跑）
    history_dir = "data/history"
    features, pred = rank_candidates(
        universe,
        lambda code: load_history_for_code(history_dir, code),
        trade_date,
        topk
    )

    # 3.5) 改造A第二段：features 算完后做严格首板过滤，再TopK覆盖 pred
    if features is not None and not features.empty:
        fb = strict_firstboard_filter_after_features(
            features,
            vol_ratio_min=1.3,
            pos20_max=0.80,
            shadow_max=0.06
        )

        # 重新TopK（优先 rank_score，其次 ml_prob，其次 prob_calibrated）
        sort_col = None
        for c in ["rank_score", "ml_prob", "prob_calibrated", "prob_raw"]:
            if c in fb.columns:
                sort_col = c
                break

        if sort_col is not None and not fb.empty:
            fb_top = fb.sort_values(sort_col, ascending=False).head(topk).copy()
            pred = fb_top
            print(f"[FirstBoard-Strict] pool_after_features={len(fb):,}  top{topk} by {sort_col}")
        else:
            print("[FirstBoard-Strict] skip (no sort_col or empty pool)")

    # 4) 输出目录
    out_dir = os.path.join("data", "processed", trade_date)
    os.makedirs(out_dir, exist_ok=True)

    # 5) 写出 universe / features / predictions（英文版：给系统用）
    universe_path = safe_to_csv(universe, os.path.join(out_dir, "universe.csv"),
                                index=False, encoding="utf-8-sig")

    if features is None or features.empty:
        print("没有生成 features（请检查历史数据是否缺失/字段是否异常）")
        print(f"已输出 universe：{universe_path}")
        return

    features_path = safe_to_csv(features, os.path.join(out_dir, "features.csv"),
                                index=False, encoding="utf-8-sig")

    if pred is None or pred.empty:
        # ====== 截面校准：新增 prob_calibrated（展示用，不改变排序）======
        sort_col_for_cal = pick_sort_col(pred)
        if sort_col_for_cal:
            pred = calibrate_prob_by_rank(pred, score_col=sort_col_for_cal)
            print(f"[Calibrate] prob_calibrated by {sort_col_for_cal}")
        else:
            pred["prob_calibrated"] = None
            print("[Calibrate] skip (no rank_score/ml_prob/prob_raw)")
        print("没有生成预测结果 pred（请检查 PhaseB 排序逻辑/TopK 设置）")
        print(f"已输出 features：{features_path}")
        return

    pred_path = safe_to_csv(pred, os.path.join(out_dir, "predictions.csv"),
                            index=False, encoding="utf-8-sig")

    # 6) 写出中文版本（含推荐星级）
    pred_cn = to_chinese_columns_with_stars(pred)
    pred_cn_path = safe_to_csv(pred_cn, os.path.join(out_dir, "predictions_中文.csv"),
                               index=False, encoding="utf-8-sig")

    # 7) tracking
    append_tracking(pred, "output/tracking.csv")

    # 8) 控制台输出摘要
    print(f"完成：{trade_date}")
    print(f"universe: {len(universe)} -> {universe_path}")
    print(f"features: {len(features)} -> {features_path}")
    print(f"predictions(top{topk}): {len(pred)} -> {pred_path}")
    print(f"predictions_中文(top{topk}): {pred_cn_path}")
    print("tracking: output/tracking.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-path", required=True, help=r"data/raw/daily_2026-02-27.csv")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    main(args.daily_path, args.topk)