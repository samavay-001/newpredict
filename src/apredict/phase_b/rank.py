import pandas as pd
import numpy as np
from apredict.features.tech import compute_features


def rank_candidates(universe: pd.DataFrame, history_loader, asof: str, topk: int = 5):
    rows = []
    skipped = 0

    for _, r in universe.iterrows():
        code = r["code"]
        try:
            hist = history_loader(code)
            feat = compute_features(hist, asof=asof, window=60)
            row = {**r.to_dict(), **feat}
            rows.append(row)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"[PhaseB] 跳过 {code}: {type(e).__name__} - {e}")

    print(f"[PhaseB] universe={len(universe)} 生成features={len(rows)} 跳过={skipped}")

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        return features_df, features_df

    # 使用当前已经存在的字段计算 rank_score（稳定版）
    # 字段来自 tech.compute_features():
    # - amount_ratio, ret20, dist_to_hh20, atr_pct, limit_ups_20
    df = features_df.copy()

    # 放量压缩：sqrt（避免7倍放量一票封神）
    vol_boost = np.sqrt(df["amount_ratio"].clip(0, 9))  # 0~3

    # 活跃度压缩：涨停次数 sqrt（最多算5次）
    act_boost = np.sqrt(df["limit_ups_20"].clip(0, 5) / 5.0)  # 0~1

    # 分歧惩罚：上影线比例 × (1-收盘强度)
    div_penalty = df["upper_shadow_ratio"].clip(0, 1) * (1 - df["close_strength"].clip(0, 1))

    # 风险惩罚：ATR%过大降低分数
    risk_penalty = (df["atr_pct"].clip(0, 0.25) / 0.25)

    # 动量：5/10日（截断）
    mom = 0.6 * df["ret_5"].clip(-0.15, 0.25) + 0.4 * df["ret_10"].clip(-0.25, 0.40)

    # 突破与位置
    breakout = df["breakout_20"]  # 0/1
    pos = (1 - df["dist_to_hh20"].clip(0, 1))

    # 最终分数（稳定版 v2）
    df["rank_score"] = (
            0.35 * vol_boost +
            0.25 * mom +
            0.15 * pos +
            0.10 * breakout +
            0.10 * act_boost +
            0.05 * df["is_limit_up_today"] -
            0.20 * div_penalty -
            0.10 * risk_penalty
    )

    df = df.sort_values("rank_score", ascending=False)
    pred_df = df.head(topk).copy()

    return df, pred_df