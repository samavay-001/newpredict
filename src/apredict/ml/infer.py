# src/apredict/ml/infer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LGBModelBundle:
    model_path: Path
    meta_path: Path
    feature_cols: List[str]
    best_iteration: Optional[int] = None
    model_version: Optional[str] = None


def _resolve_model_paths(models_dir: Path) -> Optional[Tuple[Path, Path]]:
    """
    优先加载 pro 模型；不存在则回退普通模型。
    返回 (model_path, meta_path)，都不存在则 None
    """
    pro_model = models_dir / "lgb_limitup_pro.txt"
    pro_meta = models_dir / "lgb_limitup_pro_meta.json"

    normal_model = models_dir / "lgb_limitup.txt"
    normal_meta = models_dir / "lgb_limitup_meta.json"

    if pro_model.exists() and pro_meta.exists():
        return pro_model, pro_meta
    if normal_model.exists() and normal_meta.exists():
        return normal_model, normal_meta
    return None


def load_lgb_bundle(models_dir: str | Path = "data/models", verbose: bool = True) -> Optional[LGBModelBundle]:
    models_dir = Path(models_dir)
    pair = _resolve_model_paths(models_dir)
    if pair is None:
        if verbose:
            print("[PhaseC] 未找到可用模型文件（pro/normal 都不存在）")
        return None

    model_path, meta_path = pair

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        if verbose:
            print(f"[PhaseC] 读取 meta 失败：{meta_path} ({type(e).__name__}: {e})")
        return None

    feature_cols = meta.get("feature_cols") or meta.get("features") or meta.get("feature_columns")
    if not feature_cols or not isinstance(feature_cols, list):
        if verbose:
            print(f"[PhaseC] meta 缺少 feature_cols：{meta_path}")
        return None

    best_iteration = meta.get("best_iteration", None)
    model_version = meta.get("model_version", None)

    if verbose:
        which = "PRO" if model_path.name.endswith("_pro.txt") else "NORMAL"
        print(f"[PhaseC] 加载 {which} 模型：{model_path.name}")

    return LGBModelBundle(
        model_path=model_path,
        meta_path=meta_path,
        feature_cols=list(feature_cols),
        best_iteration=best_iteration if isinstance(best_iteration, int) else None,
        model_version=model_version if isinstance(model_version, str) else None,
    )


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    x = df.copy()

    missing = [c for c in feature_cols if c not in x.columns]
    for c in missing:
        x[c] = 0.0

    x = x[feature_cols]
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in feature_cols:
        if not np.issubdtype(x[c].dtype, np.number):
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    return x


def infer_ml_prob(
    df_features: pd.DataFrame,
    models_dir: str | Path = "data/models",
    out_col: str = "ml_prob",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    生成 ml_prob 概率列（0~1）。模型不可用则原样返回（不加列）。
    """
    bundle = load_lgb_bundle(models_dir=models_dir, verbose=verbose)
    if bundle is None:
        return df_features

    try:
        import lightgbm as lgb
    except Exception as e:
        if verbose:
            print(f"[PhaseC] 无法导入 lightgbm ({type(e).__name__}: {e})")
        return df_features

    try:
        booster = lgb.Booster(model_file=str(bundle.model_path))
    except Exception as e:
        if verbose:
            print(f"[PhaseC] 加载模型失败：{bundle.model_path} ({type(e).__name__}: {e})")
        return df_features

    x = _prepare_feature_matrix(df_features, bundle.feature_cols)

    try:
        pred = booster.predict(x, num_iteration=bundle.best_iteration or booster.best_iteration)
        pred = np.asarray(pred, dtype=float)
        pred = np.clip(pred, 0.0, 1.0)
    except Exception as e:
        if verbose:
            print(f"[PhaseC] 推理失败 ({type(e).__name__}: {e})")
        return df_features

    out = df_features.copy()
    out[out_col] = pred
    return out


# ----------------------------
# ✅ 老接口兼容：rank.py 里可能在 import 这个
# ----------------------------
def predict_prob(
    df_features: pd.DataFrame,
    models_dir: str | Path = "data/models",
    verbose: bool = False,
) -> np.ndarray:
    """
    兼容旧版接口：
    - 输入 features DF
    - 返回 概率 ndarray
    """
    out = infer_ml_prob(df_features, models_dir=models_dir, out_col="ml_prob", verbose=verbose)
    if "ml_prob" not in out.columns:
        # 模型不可用时返回 NaN，交给上层回退
        return np.full(len(df_features), np.nan, dtype=float)
    return out["ml_prob"].to_numpy(dtype=float)