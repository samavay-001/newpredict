import pandas as pd
import numpy as np
from apredict.features.feature_set import compute_features_for_hist


def _ensure_cols(df: pd.DataFrame, cols_defaults: dict):
    for c, d in cols_defaults.items():
        if c not in df.columns:
            df[c] = d
    return df


def rank_candidates(
    universe: pd.DataFrame,
    history_loader,
    asof: str,
    topk: int = 5,
    min_dist_to_hh20: float = 0.01,
    max_atr_pct: float = 0.08,
):
    rows = []
    skipped = 0

    for _, r in universe.iterrows():
        code = r["code"]
        try:
            hist = history_loader(code)
            feat = compute_features_for_hist(hist, asof=asof, window=60)
            rows.append({**r.to_dict(), **feat})
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"[PhaseB] 跳过 {code}: {type(e).__name__} - {e}")

    print(f"[PhaseB] universe={len(universe)} 生成features={len(rows)} 跳过={skipped}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    # ✅ 统一兜底列（修正 limit_ups_20）
    df = _ensure_cols(df, {
        "amount_ratio": 1.0,
        "limit_ups_20": 0,
        "limit_ups_10": 0,
        "limit_ups_5": 0,
        "upper_shadow_ratio": 0.0,
        "close_strength": 0.5,
        "atr_pct": 0.0,
        "ret_5": 0.0,
        "ret_10": 0.0,
        "breakout_20": 0,
        "dist_to_hh20": 1.0,
        "is_limit_up_today": 0,
        "is_limit_up_yday": 0,
        "pct_chg": 0.0,
    })

    # ====== 生产护栏 ======
    before = len(df)
    df = df[df["atr_pct"].clip(lower=0) <= max_atr_pct].copy()
    print(f"[PhaseB] atr_guard: {before} -> {len(df)} (max_atr_pct={max_atr_pct})")
    if df.empty:
        return df, df

    # ✅ 更合理的“避免追高”：只在“突破成功且贴新高”时剔除
    #   否则你会把真正 breakout 的票砍光
    before = len(df)
    mask_chase_breakout = (df["breakout_20"].astype(int) == 1) & (df["dist_to_hh20"].clip(lower=0) < min_dist_to_hh20)
    df = df[~mask_chase_breakout].copy()
    print(f"[PhaseB] avoid_breakout_chasing: {before} -> {len(df)} (min_dist_to_hh20={min_dist_to_hh20})")
    if df.empty:
        return df, df

    # ====== rank_score ======
    vol_boost = np.sqrt(df["amount_ratio"].clip(0, 9))
    mom = 0.6 * df["ret_5"].clip(-0.15, 0.25) + 0.4 * df["ret_10"].clip(-0.25, 0.40)

    breakout = df["breakout_20"].clip(0, 1)
    pos = (1 - df["dist_to_hh20"].clip(0, 1))

    div_penalty = df["upper_shadow_ratio"].clip(0, 1) * (1 - df["close_strength"].clip(0, 1))
    risk_penalty = (df["atr_pct"].clip(0, 0.25) / 0.25)
    act_risk = np.sqrt(df["limit_ups_20"].clip(0, 5) / 5.0)
    chase_penalty = df["is_limit_up_today"].clip(0, 1)

    df["rank_score"] = (
        0.38 * vol_boost +
        0.28 * mom +
        0.16 * pos +
        0.10 * breakout +
        0.08 * df["close_strength"].clip(0, 1) -
        0.20 * div_penalty -
        0.10 * risk_penalty -
        0.10 * act_risk -
        0.20 * chase_penalty
    )

    # ====== 首板硬过滤 ======
    before = len(df)
    df = df[df["is_limit_up_yday"] < 0.5]
    drop1 = before - len(df)

    before2 = len(df)
    df = df[df["limit_ups_5"] == 0]
    drop2 = before2 - len(df)

    print(f"[PhaseB] 首板过滤：剔除昨日涨停={drop1}  剔除近5日有涨停={drop2}  剩余={len(df)}")
    if df.empty:
        return df, df

    # ====== ML 概率 ======
    sort_key = "rank_score"
    try:
        from apredict.ml.infer import predict_prob

        # ✅ 如果 infer 里有 feature_cols meta，就让 infer 自己 reindex+fill
        df["ml_prob"] = predict_prob(df)
        # ===== 新增：截面分位概率（核心优化）=====
        df["ml_prob_rank"] = df["ml_prob"].rank(pct=True)

        # 融合概率（更稳定）
        df["ml_prob_fused"] = (
                0.7 * df["ml_prob_rank"] +
                0.3 * df["ml_prob"]
        )

        sort_key = "ml_prob"
        sort_key = "ml_prob"
        print("[PhaseC] 使用 ML 概率排序：ml_prob")
    except Exception as e:
        df["ml_prob"] = 0.0
        print(f"[PhaseC] ML模型不可用，回退到rank_score。原因: {type(e).__name__} - {e}")

    df = df.sort_values(sort_key, ascending=False)
    pred_df = df.head(topk).copy()
    return df, pred_df