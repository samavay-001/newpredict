from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb

# ✅ 必须与训练脚本一致
MODEL_PATH = Path("data/models/lgb_first_limitup_meta.txt")
META_PATH = Path("data/models/lgb_first_limitup_meta.json")

_model = None
_feature_cols = None
_best_iter = None

def _load():
    global _model, _feature_cols, _best_iter

    if _model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    _feature_cols = meta["feature_cols"]
    _best_iter = meta.get("best_iteration", None)

    _model = lgb.Booster(model_file=str(MODEL_PATH))


def predict_prob(df: pd.DataFrame) -> pd.Series:
    _load()

    missing = [c for c in _feature_cols if c not in df.columns]
    if missing:
        # 只打印前若干个，避免刷屏
        print(f"[ML] missing feature {len(missing)}/{len(_feature_cols)} e.g. {missing[:12]}")

    # 自动适配特征列（关键）
    X = df.reindex(columns=_feature_cols, fill_value=0)

    # 处理 inf/NaN
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    # 可选：如果缺失太多，给一个强提醒（不影响运行）
    if len(missing) / max(len(_feature_cols), 1) >= 0.30:
        print("[ML][WARN] too many missing features -> prediction quality may collapse")

    return pd.Series(
        _model.predict(X, num_iteration=_best_iter),
        index=df.index,
        name="ml_prob"
    )