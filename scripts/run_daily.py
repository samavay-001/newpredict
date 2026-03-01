import sys
import os
import argparse
import time
from apredict.io.stock_meta import load_stock_meta, attach_stock_meta

# 让 Python 能找到 src/apredict（src-layout）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from apredict.io.loader import load_daily_snapshot, load_history_for_code
from apredict.phase_a.filter import phase_a_filter
from apredict.phase_b.rank import rank_candidates
from apredict.tracking.tracker import append_tracking


def safe_to_csv(df, path, **kwargs):
    """
    Windows 下如果文件被 Excel/WPS/预览占用会 PermissionError。
    这里自动尝试写入带序号的新文件名，保证程序不中断。
    """
    base, ext = os.path.splitext(path)
    for i in range(0, 30):
        p = path if i == 0 else f"{base}_{i}{ext}"
        try:
            df.to_csv(p, **kwargs)
            return p
        except PermissionError:
            time.sleep(0.2)
    raise PermissionError(f"文件被占用且多次重试失败：{path}（请关闭Excel/WPS/预览窗格）")


def score_to_stars(score: float) -> str:
    """
    推荐星级：只用于展示
    """
    if score >= 1.35:
        return "★★★★★"
    if score >= 1.25:
        return "★★★★☆"
    if score >= 1.15:
        return "★★★☆☆"
    if score >= 1.05:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def to_chinese_columns_with_stars(df):
    """
    只用于“人工查看”的中文版本输出（含推荐星级），不影响模型/回测用的英文版本。
    """
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

    # 增加推荐星级（根据综合评分）
    if "综合评分" in df_cn.columns:
        df_cn["推荐星级"] = df_cn["综合评分"].apply(lambda x: score_to_stars(float(x)))
    else:
        df_cn["推荐星级"] = "★☆☆☆☆"

    # 关键字段优先排序
    order = [
        "排名",
        "股票代码",
        "股票名称",
        "推荐星级",
        "综合评分",
        "预测概率",
        "AI预测概率",

        "收盘价",
        "涨跌幅%",
        "成交额",

        "放量倍数",
        "突破20日新高",

        "5日涨幅%",
        "10日涨幅%",
        "20日涨幅%",

        "ATR波动率%",
        "20日涨停次数",

        "上影线比例",
        "收盘强度",
    ]
    order = [c for c in order if c in df_cn.columns]

    # 其余字段放后面（你想更精简的话也可以直接 return df_cn[order]）
    rest = [c for c in df_cn.columns if c not in order]
    return df_cn[order + rest]


def main(daily_path: str, topk: int):

    # 1) 读取当日快照
    snapshot = load_daily_snapshot(daily_path)
    trade_date = str(snapshot["trade_date"].iloc[0])

    # ✅ 加载股票列表并补齐名称
    meta = load_stock_meta("data/meta/stock_basic.csv")
    snapshot = attach_stock_meta(snapshot, meta)

    # 2) Phase A：过滤候选池
    universe = phase_a_filter(snapshot)

    # 3) Phase B：计算特征 + 排序
    history_dir = "data/history"
    features, pred = rank_candidates(
        universe,
        lambda code: load_history_for_code(history_dir, code),
        trade_date,
        topk
    )

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
        print("没有生成预测结果 pred（请检查 PhaseB 排序逻辑/TopK 设置）")
        print(f"已输出 features：{features_path}")
        return

    pred_path = safe_to_csv(pred, os.path.join(out_dir, "predictions.csv"),
                            index=False, encoding="utf-8-sig")

    # 6) 写出中文版本（含推荐星级）
    pred_cn = to_chinese_columns_with_stars(pred)
    pred_cn_path = safe_to_csv(pred_cn, os.path.join(out_dir, "predictions_中文.csv"),
                               index=False, encoding="utf-8-sig")

    # 7) tracking（保存英文 pred，便于后续自动打标/回测/校准）
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

    parser.add_argument(
        "--daily-path",
        required=True,
        help=r"data/raw/daily_2026-02-27.csv"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5
    )

    args = parser.parse_args()
    main(args.daily_path, args.topk)