import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb

DATA_PATH = Path("data/training/training.parquet")
MODEL_PATH = Path("data/models/lgb_limitup.txt")
META_PATH = Path("data/models/lgb_limitup_meta.json")
OUT_SAMPLE = Path("data/output/top5_strict_samples.csv")
OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)

DATE_COL = "trade_date"
LABEL_COL = "label"

def topk_hit_rate(df: pd.DataFrame, prob_col: str, k: int = 5):
    g = df.groupby(DATE_COL, sort=True)
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
    if not MODEL_PATH.exists():
        raise SystemExit("模型不存在，请先运行 scripts/train_lightgbm.py")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    best_iter = meta.get("best_iteration", None)

    print(f"Loading data: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df[DATE_COL] = df[DATE_COL].astype(str)

    # 严格评估：不采样，直接用全量 test
    test = df[df[DATE_COL] >= "20250101"].copy()
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [LABEL_COL, DATE_COL])
    print(f"strict test rows={len(test)} pos_rate={test[LABEL_COL].mean():.6f}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    prob = model.predict(test[feat_cols], num_iteration=best_iter)
    test["ml_prob"] = prob

    ap = average_precision_score(test[LABEL_COL].astype(int), prob)
    try:
        auc = roc_auc_score(test[LABEL_COL].astype(int), prob)
    except Exception:
        auc = float("nan")

    print(f"[STRICT test] PR-AUC(AP)={ap:.6f} ROC-AUC={auc:.6f}")

    top5 = topk_hit_rate(test, "ml_prob", k=5)
    print(f"[STRICT test] top5: {top5}")

    # 输出最近10天 Top5 方便人工核查
    last_days = sorted(test[DATE_COL].unique())[-10:]
    sample = test[test[DATE_COL].isin(last_days)].copy()
    sample = sample.sort_values([DATE_COL, "ml_prob"], ascending=[True, False])
    sample = sample.groupby(DATE_COL).head(5)
    sample.to_csv(OUT_SAMPLE, index=False, encoding="utf-8-sig")
    print(f"Saved sample: {OUT_SAMPLE}")

if __name__ == "__main__":
    main()