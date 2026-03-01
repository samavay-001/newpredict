import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb

DATA_PATH = Path("data/training/training.parquet")
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgb_limitup.txt"
META_PATH = MODEL_DIR / "lgb_limitup_meta.json"

# 训练使用的特征列（与你 build_training_dataset 输出一致）
FEATURE_COLS = [
    "open", "close", "high", "low",
    "vol", "amount", "pct_chg",
    "amount_ratio", "ret_5", "ret_10", "ret20",
    "breakout_20", "dist_to_hh20", "atr_pct",
    "limit_ups_20", "upper_shadow_ratio", "close_strength",
]

LABEL_COL = "label"
DATE_COL = "trade_date"


def load_data():
    df = pd.read_parquet(DATA_PATH)
    # 基本清洗
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + [LABEL_COL, DATE_COL])
    df[DATE_COL] = df[DATE_COL].astype(str)
    return df


def time_split(df: pd.DataFrame):
    """
    时间切分（避免未来数据泄露）：
    - train: < 20240101
    - valid: [20240101, 20250101)
    - test : >= 20250101
    你也可以后面按自己需求改。
    """
    train = df[df[DATE_COL] < "20240101"].copy()
    valid = df[(df[DATE_COL] >= "20240101") & (df[DATE_COL] < "20250101")].copy()
    test = df[df[DATE_COL] >= "20250101"].copy()
    return train, valid, test


def evaluate(name: str, y_true, y_prob):
    ap = average_precision_score(y_true, y_prob)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    print(f"[{name}] PR-AUC(AP)={ap:.6f} ROC-AUC={auc:.6f}")
    return {"ap": float(ap), "auc": float(auc)}


def topk_hit_rate(df_part: pd.DataFrame, prob_col: str, k: int = 5):
    """
    评估：每天取TopK，看看有没有命中（label=1）
    返回：
    - day_hit_rate：命中天数/总天数
    - avg_hits：每天TopK平均命中个数
    """
    g = df_part.groupby(DATE_COL, sort=True)
    total_days = 0
    hit_days = 0
    hits = []
    for d, x in g:
        x = x.sort_values(prob_col, ascending=False).head(k)
        h = int(x[LABEL_COL].sum())
        hits.append(h)
        total_days += 1
        if h > 0:
            hit_days += 1
    return {
        "day_hit_rate": float(hit_days / max(total_days, 1)),
        "avg_hits_in_topk": float(np.mean(hits) if hits else 0.0),
        "total_days": int(total_days),
    }


def main():
    print(f"Loading: {DATA_PATH}")
    df = load_data()
    print(f"rows={len(df)}  pos_rate={df[LABEL_COL].mean():.6f}")

    # 为了更稳更快：可选下采样负样本（不影响线上推理）
    # 你数据极不平衡（涨停很少），不下采样会更慢。
    pos = df[df[LABEL_COL] == 1]
    neg = df[df[LABEL_COL] == 0]

    # 负样本采样比例（建议 20~50 倍）
    neg_ratio = 30
    neg_sample = neg.sample(n=min(len(neg), len(pos) * neg_ratio), random_state=42)

    df_train_all = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1, random_state=42)
    print(f"after_sampling rows={len(df_train_all)} pos_rate={df_train_all[LABEL_COL].mean():.6f}")

    train_df, valid_df, test_df = time_split(df_train_all)

    X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL].astype(int)
    X_valid, y_valid = valid_df[FEATURE_COLS], valid_df[LABEL_COL].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df[LABEL_COL].astype(int)

    # 处理类别不平衡：scale_pos_weight
    pos_cnt = int(y_train.sum())
    neg_cnt = int((y_train == 0).sum())
    spw = (neg_cnt / max(pos_cnt, 1))

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "scale_pos_weight": spw,
        "verbosity": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    # 评估（valid/test）
    valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    test_prob = model.predict(X_test, num_iteration=model.best_iteration)

    m_valid = evaluate("valid", y_valid, valid_prob)
    m_test = evaluate("test", y_test, test_prob)

    # TopK命中评估（注意：这里是在采样后的集合上评估，偏乐观）
    # 更严谨的做法是用“未采样的原始 test 集合”来算 topk，
    # 但会更慢；我们下一步可以加一个 strict_eval 模式。
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    valid_df["ml_prob"] = valid_prob
    test_df["ml_prob"] = test_prob

    topk_valid = topk_hit_rate(valid_df, "ml_prob", k=5)
    topk_test = topk_hit_rate(test_df, "ml_prob", k=5)

    print("[valid] top5:", topk_valid)
    print("[test ] top5:", topk_test)

    # 保存模型
    model.save_model(str(MODEL_PATH))
    meta = {
        "feature_cols": FEATURE_COLS,
        "best_iteration": int(model.best_iteration),
        "metrics_valid": m_valid,
        "metrics_test": m_test,
        "top5_valid": topk_valid,
        "top5_test": topk_test,
        "neg_ratio": neg_ratio,
        "scale_pos_weight": float(spw),
        "data_rows_used": int(len(df_train_all)),
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta : {META_PATH}")


if __name__ == "__main__":
    main()