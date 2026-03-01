import pandas as pd
import numpy as np
from apredict.features.tech import compute_features


def _ensure_cols(df: pd.DataFrame, cols_defaults: dict):
    """
    确保df存在某些列；不存在则按默认值补齐，避免 KeyError。
    """
    for c, d in cols_defaults.items():
        if c not in df.columns:
            df[c] = d
    return df


def rank_candidates(
    universe: pd.DataFrame,
    history_loader,
    asof: str,
    topk: int = 5,
    # ===== 新增：生产稳定性参数（可调）=====
    min_dist_to_hh20: float = 0.01,  # 避免贴着20日新高追高；0.01=离高点至少1%
    max_atr_pct: float = 0.08,       # 极端波动护栏；0.08=ATR%>8%剔除
):
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

    df = features_df.copy()

    # 兜底：避免字段缺失导致崩溃
    df = _ensure_cols(df, {
        "amount_ratio": 1.0,
        "limit_ups_20": 0.0,
        "limit_ups_5": 0.0,
        "upper_shadow_ratio": 0.0,
        "close_strength": 0.5,
        "atr_pct": 0.0,
        "ret_5": 0.0,
        "ret_10": 0.0,
        "breakout_20": 0.0,
        "dist_to_hh20": 1.0,
        "is_limit_up_today": 0.0,
        "is_limit_up_yday": 0.0,
        "pct_chg": 0.0,
    })

    # ============================================================
    # 0) 生产护栏：极端波动剔除 + 避免贴新高追高
    #    放在首板过滤之前，先把“最不稳定/最像追高”的样本拿掉
    # ============================================================
    # ATR 护栏
    before = len(df)
    df = df[df["atr_pct"].clip(lower=0) <= max_atr_pct].copy()
    print(f"[PhaseB] atr_guard: {before} -> {len(df)} (max_atr_pct={max_atr_pct})")

    if df.empty:
        return df, df

    # 避免贴着 20 日高点追高（dist_to_hh20 太小说明“几乎就是新高附近”）
    before = len(df)
    df = df[df["dist_to_hh20"].clip(lower=0) >= min_dist_to_hh20].copy()
    print(f"[PhaseB] avoid_buying_at_high: {before} -> {len(df)} (min_dist_to_hh20={min_dist_to_hh20})")

    if df.empty:
        return df, df

    # ============================================================
    # 1) 先计算 rank_score（你原有：首板启动稳定版 v3）
    # ============================================================

    # 放量压缩：sqrt（避免极端放量一票封神）
    vol_boost = np.sqrt(df["amount_ratio"].clip(0, 9))  # 0~3

    # 动量：5/10日（截断）
    mom = 0.6 * df["ret_5"].clip(-0.15, 0.25) + 0.4 * df["ret_10"].clip(-0.25, 0.40)

    # 突破与位置
    breakout = df["breakout_20"].clip(0, 1)            # 0/1
    pos = (1 - df["dist_to_hh20"].clip(0, 1))          # 越接近20日高点越好（但已被上面护栏限制了“过近”）

    # 分歧惩罚：上影线比例 × (1-收盘强度)
    div_penalty = df["upper_shadow_ratio"].clip(0, 1) * (1 - df["close_strength"].clip(0, 1))

    # 风险惩罚：ATR%过大降低分数
    risk_penalty = (df["atr_pct"].clip(0, 0.25) / 0.25)

    # 过去涨停次数：对首板预测而言是“风险项”（妖/接力属性），做惩罚
    act_risk = np.sqrt(df["limit_ups_20"].clip(0, 5) / 5.0)  # 0~1

    # 追板项：当天涨停属于“已涨停”，通常不作为首板启动候选
    chase_penalty = df["is_limit_up_today"].clip(0, 1)

    # 最终分数（首板启动 v3）
    df["rank_score"] = (
        0.38 * vol_boost +
        0.28 * mom +
        0.16 * pos +
        0.10 * breakout +
        0.08 * df["close_strength"].clip(0, 1) -   # 强收盘加分
        0.20 * div_penalty -
        0.10 * risk_penalty -
        0.10 * act_risk -                          # 抑制妖/接力
        0.20 * chase_penalty                       # 抑制追板（兜底）
    )

    # ============================================================
    # 2) 硬过滤：昨日涨停 + 近5日有涨停（不追板/不做接力）
    # ============================================================
    before = len(df)

    # 昨日涨停
    df = df[df["is_limit_up_yday"] < 0.5]
    drop1 = before - len(df)

    # 近5日出现过涨停（接力/连板/妖股）
    before2 = len(df)
    df = df[df["limit_ups_5"] == 0]
    drop2 = before2 - len(df)

    print(f"[PhaseB] 首板过滤：剔除昨日涨停={drop1}  剔除近5日有涨停={drop2}  剩余={len(df)}")

    if df.empty:
        return df, df

    # ============================================================
    # 3) 尝试生成 ml_prob（失败则回退）
    # ============================================================
    sort_key = "rank_score"
    try:
        from apredict.ml.infer import predict_prob

        df["ml_prob"] = predict_prob(df)

        sort_key = "ml_prob"
        print("[PhaseC] 使用 ML 概率排序：ml_prob")
    except Exception as e:
        df["ml_prob"] = 0.0
        print(f"[PhaseC] ML模型不可用，回退到rank_score。原因: {type(e).__name__} - {e}")

    # ============================================================
    # 4) 排序 & TopK
    # ============================================================
    df = df.sort_values(sort_key, ascending=False)
    pred_df = df.head(topk).copy()

    return df, pred_df