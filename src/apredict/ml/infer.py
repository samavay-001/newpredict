from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb

MODEL_PATH = Path("data/models/lgb_limitup.txt")
META_PATH = Path("data/models/lgb_limitup_meta.json")

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
    X = df[_feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    return pd.Series(_model.predict(X, num_iteration=_best_iter), index=df.index, name="ml_prob")