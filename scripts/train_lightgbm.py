import os
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb


# =========================
# Config
# =========================
DATA_PATH = Path("data/training/training.parquet")
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgb_limitup_pro.txt"
META_PATH = MODEL_DIR / "lgb_limitup_pro_meta.json"

# 必备列（训练数据必须有）
DATE_COL = "trade_date"
CODE_COL = "code"

# 如果你没有板块字段，就留空（None）
# 若你有例如 'board'/'market' 可在 detect_limit_threshold 里自定义逻辑
BOARD_COL = None  # e.g. "board"

# 涨停阈值（近似）
LIMIT10 = 9.8
LIMIT20 = 19.8

# 时间切分（可按需改）
TRAIN_END = "20240101"
VALID_END = "20250101"

# 训练负样本采样比例（只影响训练速度，不影响严格评估）
NEG_RATIO = 30
SEED = 42


# =========================
# Helpers
# =========================
def _safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return a / b


def _as_str_date(x: pd.Series) -> pd.Series:
    """
    统一把日期转成 'YYYYMMDD' 字符串，兼容：
    - pandas StringDtype
    - object
    - int (20260227)
    - datetime
    - 'YYYY-MM-DD'
    """

    # 转字符串
    s = x.astype("string")

    # 尝试解析为 datetime
    dt = pd.to_datetime(s, errors="coerce")

    # 能解析的转 yyyyMMdd
    out = s.copy()
    mask = dt.notna()
    out[mask] = dt[mask].dt.strftime("%Y%m%d")

    # 清理常见格式
    out = out.str.replace("-", "", regex=False)
    out = out.str.replace("/", "", regex=False)

    return out.astype(str)


def detect_limit_threshold(df: pd.DataFrame) -> np.ndarray:
    """
    生成每行对应的涨停阈值（10% or 20%）。
    - 如果你没有板块字段，默认全部按 10% 阈值（9.8）处理
      这会把20%板块当成“非涨停”，会引入噪声，但能先跑通。
    - 如果你有 BOARD_COL，可在这里改成更精准的规则。
    """
    if BOARD_COL and BOARD_COL in df.columns:
        s = df[BOARD_COL].astype(str)
        # 示例：包含 "20" 就视作 20% 板块（你按实际字段改）
        thr = np.where(s.str.contains("20"), LIMIT20, LIMIT10)
        return thr.astype(float)

    return np.full(len(df), LIMIT10, dtype=float)


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 振幅
    out["range_pct"] = _safe_div(out["high"] - out["low"], out["close"]).fillna(0.0)

    # 收盘强度 0~1
    denom = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_strength"] = ((out["close"] - out["low"]) / denom).fillna(0.0).clip(0, 1)

    # 上影线比例
    upper = out["high"] - np.maximum(out["open"], out["close"])
    out["upper_shadow"] = _safe_div(upper, out["close"]).fillna(0.0).clip(0, 1)

    return out


def add_volume_ratio(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    g = out.groupby(CODE_COL, group_keys=False)

    for w in windows:
        ma = g["vol"].rolling(w).mean().reset_index(level=0, drop=True)
        out[f"vol_ma_{w}"] = ma
        out[f"vol_ratio_{w}"] = (out["vol"] / ma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ama = g["amount"].rolling(w).mean().reset_index(level=0, drop=True)
        out[f"amount_ma_{w}"] = ama
        out[f"amount_ratio_{w}"] = (out["amount"] / ama.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def add_price_position(df: pd.DataFrame, window=20) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    g = out.groupby(CODE_COL, group_keys=False)

    hh = g["high"].rolling(window).max().reset_index(level=0, drop=True)
    ll = g["low"].rolling(window).min().reset_index(level=0, drop=True)

    out[f"hh{window}"] = hh
    out[f"ll{window}"] = ll

    denom = (hh - ll).replace(0, np.nan)
    out[f"pos_{window}"] = ((out["close"] - ll) / denom).fillna(0.5).clip(0, 1)

    out[f"dist_to_hh{window}"] = ((hh - out["close"]) / out["close"].replace(0, np.nan)).fillna(0.0).clip(0, 10)
    out[f"dist_to_ll{window}"] = ((out["close"] - ll) / out["close"].replace(0, np.nan)).fillna(0.0).clip(0, 10)

    # 近window最大回撤（从滚动最高到当前）
    out[f"drawdown_{window}"] = (out["close"] / hh.replace(0, np.nan) - 1.0).fillna(0.0).clip(-1, 1)

    return out


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果你的 parquet 已经有 ret_5/ret_10/ret20，可保留；
    没有的话这里补一套（用 close 计算）。
    """
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    g = out.groupby(CODE_COL, group_keys=False)

    if "ret_5" not in out.columns:
        out["ret_5"] = g["close"].pct_change(5).fillna(0.0)
    if "ret_10" not in out.columns:
        out["ret_10"] = g["close"].pct_change(10).fillna(0.0)
    if "ret20" not in out.columns:
        out["ret20"] = g["close"].pct_change(20).fillna(0.0)

    # 连涨天数（启动阶段常见）
    up = (g["close"].diff() > 0).astype(int)
    out["up_days_streak"] = up.groupby(out[CODE_COL]).apply(lambda s: s.groupby((s == 0).cumsum()).cumsum()).reset_index(level=0, drop=True)
    out["up_days_streak"] = out["up_days_streak"].fillna(0).clip(0, 20)

    return out


def add_limitup_flags_and_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    thr = detect_limit_threshold(out)

    out["is_limit_up_today"] = (out["pct_chg"].values >= thr).astype(int)
    out["is_limit_up_yday"] = out.groupby(CODE_COL)["is_limit_up_today"].shift(1).fillna(0).astype(int)

    g = out.groupby(CODE_COL, group_keys=False)

    # 近N日涨停次数（行为记忆）
    for w in (5, 10, 20):
        out[f"limit_ups_{w}"] = g["is_limit_up_today"].rolling(w).sum().reset_index(level=0, drop=True).fillna(0).astype(int)

    # 距离上次涨停天数
    def _days_since_last_limitup(x: pd.Series) -> pd.Series:
        last = None
        res = []
        for i, v in enumerate(x.values):
            if v == 1:
                last = i
                res.append(0)
            else:
                res.append(i - last if last is not None else 10000)
        return pd.Series(res, index=x.index)

    out["days_since_limit_up"] = g["is_limit_up_today"].apply(_days_since_last_limitup).clip(0, 10000)

    # 连板高度（近似：连续涨停天数）
    def _consecutive_limitups(x: pd.Series) -> pd.Series:
        res = []
        streak = 0
        for v in x.values:
            if v == 1:
                streak += 1
            else:
                streak = 0
            res.append(streak)
        return pd.Series(res, index=x.index)

    out["limitup_streak"] = g["is_limit_up_today"].apply(_consecutive_limitups).fillna(0).clip(0, 10)

    return out


def make_label_nextday_limitup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    out["label_next_limitup"] = out.groupby(CODE_COL)["is_limit_up_today"].shift(-1).fillna(0).astype(int)
    return out


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    need = [DATE_COL, CODE_COL, "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"training.parquet 缺少必要字段: {miss}")

    out = df.copy()
    out[DATE_COL] = _as_str_date(out[DATE_COL])
    bad = out[DATE_COL].str.len() != 8
    if bad.any():
        print(f"[WARN] trade_date 有 {int(bad.sum())} 行不是YYYYMMDD，将被丢弃")
        out = out[~bad].copy()

    # 基础清洗：去除无效数
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=need)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_and_validate(df)
    df = add_candle_features(df)
    df = add_volume_ratio(df, windows=(5, 10, 20))
    df = add_price_position(df, window=20)
    df = add_return_features(df)
    df = add_limitup_flags_and_history(df)
    df = make_label_nextday_limitup(df)
    return df


def time_split(df: pd.DataFrame):
    train = df[df[DATE_COL] < TRAIN_END].copy()
    valid = df[(df[DATE_COL] >= TRAIN_END) & (df[DATE_COL] < VALID_END)].copy()
    test = df[df[DATE_COL] >= VALID_END].copy()
    return train, valid, test


def evaluate_probs(name: str, y_true, y_prob):
    ap = average_precision_score(y_true, y_prob)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    print(f"[{name}] PR-AUC(AP)={ap:.6f} ROC-AUC={auc:.6f}")
    return {"ap": float(ap), "auc": float(auc)}


def topk_metrics_strict(df_part: pd.DataFrame, prob_col: str, k: int = 5):
    """
    严格TopK：按“每天全市场/全样本”取TopK
    输出：
    - hit_rate: 每天TopK至少命中1只涨停的比例
    - avg_hits: 每天TopK平均命中数
    - precision_at_k: 平均(命中数/k)
    - lift_at_k: 相对随机选k只的提升倍数
    """
    g = df_part.groupby(DATE_COL, sort=True)
    total_days = 0
    hit_days = 0
    hits = []
    lifts = []
    for d, x in g:
        total_days += 1
        x = x.sort_values(prob_col, ascending=False)
        top = x.head(k)
        h = int(top["label_next_limitup"].sum())
        hits.append(h)
        if h > 0:
            hit_days += 1

        # 随机基准：当日正样本率 * k
        base_rate = float(x["label_next_limitup"].mean())
        expected_random_hits = base_rate * k
        lift = (h / expected_random_hits) if expected_random_hits > 1e-12 else (0.0 if h == 0 else 999.0)
        lifts.append(lift)

    avg_hits = float(np.mean(hits) if hits else 0.0)
    precision = float(avg_hits / k) if k > 0 else 0.0
    return {
        "k": int(k),
        "hit_rate": float(hit_days / max(total_days, 1)),
        "avg_hits": avg_hits,
        "precision_at_k": precision,
        "avg_lift": float(np.mean(lifts) if lifts else 0.0),
        "total_days": int(total_days),
    }


def sample_for_training(train_df: pd.DataFrame, neg_ratio: int, seed: int):
    """
    仅用于训练提速：保留全部正样本 + 采样部分负样本
    """
    pos = train_df[train_df["label_next_limitup"] == 1]
    neg = train_df[train_df["label_next_limitup"] == 0]

    if len(pos) == 0:
        raise RuntimeError("训练集没有正样本（label=1），请检查标签或时间切分！")

    neg_sample = neg.sample(n=min(len(neg), len(pos) * neg_ratio), random_state=seed)
    out = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1, random_state=seed)
    return out


def main():
    print(f"Loading raw data: {DATA_PATH}")
    raw = pd.read_parquet(DATA_PATH)
    print(f"raw rows={len(raw):,} cols={len(raw.columns)}")

    print("Building professional limit-up features + next-day label...")
    df = build_features(raw)

    # 清理：最后一天 label shift(-1) 无意义
    df = df.dropna(subset=["label_next_limitup"])
    df = df.replace([np.inf, -np.inf], np.nan)

    # 训练前：确保关键列不缺失
    # 你也可以把 FEATURE_COLS 写到 meta 里，推理时直接读取
    FEATURE_COLS = [
        # 基础价量
        "open", "close", "high", "low", "vol", "amount", "pct_chg",

        # 动量
        "ret_5", "ret_10", "ret20",

        # 量能爆发
        "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
        "amount_ratio_5", "amount_ratio_10", "amount_ratio_20",

        # 位置/回撤
        "pos_20", "dist_to_hh20", "dist_to_ll20", "drawdown_20",

        # K线行为
        "range_pct", "close_strength", "upper_shadow",

        # 涨停行为历史（最关键）
        "is_limit_up_yday",
        "limit_ups_5", "limit_ups_10", "limit_ups_20",
        "days_since_limit_up",
        "limitup_streak",

        # 启动行为
        "up_days_streak",
    ]

    # dropna
    df = df.dropna(subset=FEATURE_COLS + [DATE_COL, CODE_COL, "label_next_limitup"])

    pos_rate = float(df["label_next_limitup"].mean())
    print(f"all rows={len(df):,}  pos_rate={pos_rate:.6f}")

    # 时间切分（严格）
    train_all, valid_all, test_all = time_split(df)
    print(f"split: train={len(train_all):,} valid={len(valid_all):,} test={len(test_all):,}")

    # 训练采样（仅对 train）
    train_df = sample_for_training(train_all, neg_ratio=NEG_RATIO, seed=SEED)
    print(f"train sampled rows={len(train_df):,} pos_rate={train_df['label_next_limitup'].mean():.6f}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label_next_limitup"].astype(int)

    X_valid = valid_all[FEATURE_COLS]
    y_valid = valid_all["label_next_limitup"].astype(int)

    X_test = test_all[FEATURE_COLS]
    y_test = test_all["label_next_limitup"].astype(int)

    # 类别不平衡（用采样后的 train 来算也可以；更稳是用原 train_all）
    pos_cnt = int(train_all["label_next_limitup"].sum())
    neg_cnt = int((train_all["label_next_limitup"] == 0).sum())
    spw = (neg_cnt / max(pos_cnt, 1))

    params = {
        "objective": "binary",
        "metric": "auc",

        # 学习率/树结构
        "learning_rate": 0.05,
        "num_leaves": 63,  # 从127降到63，更稳
        "max_depth": 8,  # 加深度限制，防过拟合
        "min_data_in_leaf": 300,  # 200 -> 300 更稳（你数据大，叶子可以更大）
        "min_gain_to_split": 0.01,  # 分裂要有“真收益”，抑制噪声分裂

        # 随机化增强泛化
        "feature_fraction": 0.8,  # 0.85 -> 0.8
        "bagging_fraction": 0.8,  # 0.85 -> 0.8
        "bagging_freq": 1,

        # 正则
        "lambda_l2": 10.0,  # 5 -> 10 更稳
        "lambda_l1": 1.0,  # 加L1，防止少数特征“独大”

        # 不平衡
        "scale_pos_weight": spw,

        # 稳定性
        "verbosity": -1,
        "seed": SEED,
        "feature_pre_filter": False,  # 防止特征被误过滤（可选但推荐）
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=FEATURE_COLS)

    print("Training LightGBM (professional limit-up model)...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train_sampled", "valid_strict"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=100),
        ],
    )

    # 概率输出（严格评估用原始 valid/test）
    valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    test_prob = model.predict(X_test, num_iteration=model.best_iteration)

    m_valid = evaluate_probs("valid_strict", y_valid, valid_prob)
    m_test = evaluate_probs("test_strict", y_test, test_prob)

    # TopK严格评估：用原始 valid/test（不采样）
    valid_eval = valid_all[[DATE_COL, CODE_COL, "label_next_limitup"]].copy()
    test_eval = test_all[[DATE_COL, CODE_COL, "label_next_limitup"]].copy()
    valid_eval["ml_prob"] = valid_prob
    test_eval["ml_prob"] = test_prob

    topk_valid_5 = topk_metrics_strict(valid_eval, "ml_prob", k=5)
    topk_valid_10 = topk_metrics_strict(valid_eval, "ml_prob", k=10)
    topk_valid_20 = topk_metrics_strict(valid_eval, "ml_prob", k=20)

    topk_test_5 = topk_metrics_strict(test_eval, "ml_prob", k=5)
    topk_test_10 = topk_metrics_strict(test_eval, "ml_prob", k=10)
    topk_test_20 = topk_metrics_strict(test_eval, "ml_prob", k=20)

    print("[valid] topk:", topk_valid_5, topk_valid_10, topk_valid_20)
    print("[test ] topk:", topk_test_5, topk_test_10, topk_test_20)

    # feature importance（可选，帮助你做因子审计）
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    # 保存模型
    model.save_model(str(MODEL_PATH))
    meta = {
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "feature_cols": FEATURE_COLS,
        "best_iteration": int(model.best_iteration),
        "train_end": TRAIN_END,
        "valid_end": VALID_END,
        "neg_ratio_train": int(NEG_RATIO),
        "scale_pos_weight": float(spw),
        "metrics_valid": m_valid,
        "metrics_test": m_test,
        "topk_valid": {"k5": topk_valid_5, "k10": topk_valid_10, "k20": topk_valid_20},
        "topk_test": {"k5": topk_test_5, "k10": topk_test_10, "k20": topk_test_20},
        "pos_rate_all": float(pos_rate),
        "train_rows_all": int(len(train_all)),
        "valid_rows_all": int(len(valid_all)),
        "test_rows_all": int(len(test_all)),
        "limit10": LIMIT10,
        "limit20": LIMIT20,
        "board_col": BOARD_COL,
        "feature_importance_top20": imp.head(20).to_dict(orient="records"),
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta : {META_PATH}")


if __name__ == "__main__":
    main()