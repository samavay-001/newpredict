# src/apredict/features/tech.py
from __future__ import annotations

"""
线上特征入口（兼容旧 import）：

旧代码通常写：
    from apredict.features.tech import compute_features

我们保留这个入口，但内部实现改为统一口径：
    apredict.features.feature_set.compute_features_for_hist

这样：
- rank_candidates 不用改 import（最稳）
- 训练与线上特征完全一致（避免“训练一套、线上一套”的漂移）
"""

from apredict.features.feature_set import (
    compute_features_for_hist as compute_features,
    FEATURE_COLS_DEFAULT,
)

__all__ = ["compute_features", "FEATURE_COLS_DEFAULT"]