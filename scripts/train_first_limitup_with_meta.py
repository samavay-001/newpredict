import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_PATH = Path("data/training/training.parquet")
META_PATH = Path("data/meta/stock_basic.csv")

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgb_first_limitup_meta.txt"
MODEL_META_PATH = MODEL_DIR / "lgb_first_limitup_meta.json"

DATE_COL = "trade_date"
CODE_COL = "code"

# 近似涨停阈值
LIMIT10 = 9.8
LIMIT20 = 19.8

TRAIN_END = "20240101"
VALID_END = "20250101"

NEG_RATIO = 30
SEED = 42


def _as_str_date(x: pd.Series) -> pd.Series:
    s = x.astype("string")
    s = s.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    dt = pd.to_datetime(s, errors="coerce", format="%Y%m%d")
    out = s.copy()
    m = dt.notna()
    out[m] = dt[m].dt.strftime("%Y%m%d")
    return out.astype(str)


def _safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return a / b


def load_stock_basic() -> pd.DataFrame:
    """
    读取 data/meta/stock_basic.csv
    你给的示例看起来是“制表符分隔”，这里做了兼容：
    - 自动尝试 sep='\t' 或 ','
    - 自动尝试 utf-8-sig / gbk
    """
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    seps = ["\t", ","]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(META_PATH, encoding=enc, sep=sep)
                # 必须包含这几列
                need = ["TS代码", "股票代码", "股票名称", "市场类型", "交易所代码", "上市日期"]
                if all(c in df.columns for c in need):
                    # 规范字段
                    out = df[need].copy()
                    out.rename(columns={
                        "TS代码": "ts_code",
                        "股票代码": "code",
                        "股票名称": "name",
                        "市场类型": "market_type",
                        "交易所代码": "exchange",
                        "上市日期": "list_date",
                    }, inplace=True)
                    out["code"] = out["code"].astype(str).str.zfill(6)
                    out["list_date"] = _as_str_date(out["list_date"])
                    out["is_st_name"] = out["name"].astype(str).str.contains("ST", regex=False).astype(int)
                    return out
            except Exception as e:
                last_err = e

    raise RuntimeError(f"无法读取 stock_basic.csv，请检查分隔符/编码。最后错误: {last_err}")


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    need = [DATE_COL, CODE_COL, "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"training.parquet 缺少必要字段: {miss}")

    out = df.copy()
    out[DATE_COL] = _as_str_date(out[DATE_COL])
    out[CODE_COL] = out[CODE_COL].astype(str).str.zfill(6)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=need)
    return out


def merge_meta(df: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(basic, on="code", how="left")

    # 缺失元信息的票（理论上很少），先标记为 unknown
    out["market_type"] = out["market_type"].fillna("未知")
    out["exchange"] = out["exchange"].fillna("UNK")
    out["list_date"] = out["list_date"].fillna("00000000")
    out["is_st_name"] = out["is_st_name"].fillna(0).astype(int)

    # 上市天数特征（对剔除新股/次新很有用）
    td = pd.to_datetime(out[DATE_COL], format="%Y%m%d", errors="coerce")
    ld = pd.to_datetime(out["list_date"], format="%Y%m%d", errors="coerce")
    out["days_listed"] = (td - ld).dt.days
    out["days_listed"] = out["days_listed"].fillna(99999).clip(0, 99999)

    return out


def limit_threshold_by_market(df: pd.DataFrame) -> np.ndarray:
    """
    根据 market_type 自动区分 10% / 20%
    你这个 stock_basic.csv 的“市场类型”可能是：主板 / 创业板 / 科创板 等
    """
    mt = df["market_type"].astype(str)
    is_20 = mt.str.contains("创业板") | mt.str.contains("科创板")
    thr = np.where(is_20, LIMIT20, LIMIT10)
    return thr.astype(float)


def add_rolling_ratios(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    g = out.groupby(CODE_COL, group_keys=False)

    for w in windows:
        vma = g["vol"].rolling(w).mean().reset_index(level=0, drop=True)
        ama = g["amount"].rolling(w).mean().reset_index(level=0, drop=True)
        out[f"vol_ratio_{w}"] = (out["vol"] / vma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"amount_ratio_{w}"] = (out["amount"] / ama.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def add_price_position(df: pd.DataFrame, window=20) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    g = out.groupby(CODE_COL, group_keys=False)

    hh = g["high"].rolling(window).max().reset_index(level=0, drop=True)
    ll = g["low"].rolling(window).min().reset_index(level=0, drop=True)

    denom = (hh - ll).replace(0, np.nan)
    out[f"pos_{window}"] = ((out["close"] - ll) / denom).fillna(0.5).clip(0, 1)
    out[f"dist_to_ll{window}"] = ((out["close"] - ll) / out["close"].replace(0, np.nan)).fillna(0.0).clip(0, 10)
    # 你已有 dist_to_hh20，保留即可
    return out


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range_pct"] = _safe_div(out["high"] - out["low"], out["close"]).fillna(0.0)
    denom = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_strength2"] = ((out["close"] - out["low"]) / denom).fillna(0.0).clip(0, 1)
    upper = out["high"] - np.maximum(out["open"], out["close"])
    out["upper_shadow"] = _safe_div(upper, out["close"]).fillna(0.0).clip(0, 1)
    return out


def add_limitup_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    thr = limit_threshold_by_market(out)
    out["thr_limit"] = thr

    out["is_limit_up_today"] = (out["pct_chg"].values >= thr).astype(int)
    out["is_limit_up_yday"] = out.groupby(CODE_COL)["is_limit_up_today"].shift(1).fillna(0).astype(int)

    g = out.groupby(CODE_COL, group_keys=False)
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
    return out


def make_firstboard_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    首板标签（更严谨）：
    label_first = 1 当且仅当：
      - 次日涨停（next_is_limit_up == 1）
      - 今日不是涨停
      - 过去20日没有涨停（limit_ups_20 == 0）
      - 非ST（仅基于名称包含ST识别）
      - 排除新股窗口（例如上市<60天）
    """
    out = df.sort_values([CODE_COL, DATE_COL]).copy()
    out["next_is_limit_up"] = out.groupby(CODE_COL)["is_limit_up_today"].shift(-1).fillna(0).astype(int)

    # 新股过滤：上市天数<60 直接排除（你可改 30/90）
    ok_listed = out["days_listed"] >= 60

    out["label_first"] = (
        (out["next_is_limit_up"] == 1)
        & (out["is_limit_up_today"] == 0)
        & (out["limit_ups_20"] == 0)
        & (out["is_st_name"] == 0)
        & ok_listed
    ).astype(int)

    return out


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
    g = df_part.groupby(DATE_COL, sort=True)
    total_days = 0
    hit_days = 0
    hits = []
    lifts = []
    for d, x in g:
        total_days += 1
        x = x.sort_values(prob_col, ascending=False)
        top = x.head(k)
        h = int(top["label_first"].sum())
        hits.append(h)
        if h > 0:
            hit_days += 1

        base_rate = float(x["label_first"].mean())
        expected = base_rate * k
        lift = (h / expected) if expected > 1e-12 else (0.0 if h == 0 else 999.0)
        lifts.append(lift)

    avg_hits = float(np.mean(hits) if hits else 0.0)
    return {
        "k": int(k),
        "hit_rate": float(hit_days / max(total_days, 1)),
        "avg_hits": avg_hits,
        "precision_at_k": float(avg_hits / k),
        "avg_lift": float(np.mean(lifts) if lifts else 0.0),
        "total_days": int(total_days),
    }


def sample_for_training(train_df: pd.DataFrame, neg_ratio: int, seed: int):
    pos = train_df[train_df["label_first"] == 1]
    neg = train_df[train_df["label_first"] == 0]
    if len(pos) == 0:
        raise RuntimeError("训练集没有正样本(label_first=1)，请检查标签或过滤条件。")
    neg_sample = neg.sample(n=min(len(neg), len(pos) * neg_ratio), random_state=seed)
    out = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1, random_state=seed)
    return out


def main():
    print(f"Loading raw data: {DATA_PATH}")
    raw = pd.read_parquet(DATA_PATH)
    print(f"raw rows={len(raw):,} cols={len(raw.columns)}")

    print(f"Loading stock meta: {META_PATH}")
    basic = load_stock_basic()
    print(f"meta rows={len(basic):,} cols={len(basic.columns)}")

    print("Cleaning + merging meta...")
    df = clean_and_validate(raw)
    df = merge_meta(df, basic)

    print("Building FIRST-BOARD features + label_first (with 10%/20% thresholds)...")
    # 构建特征
    df = add_rolling_ratios(df, windows=(5, 10, 20))
    df = add_price_position(df, window=20)
    df = add_candle_features(df)
    df = add_limitup_flags(df)
    df = make_firstboard_label(df)

    # 特征列（含 meta 特征）
    FEATURE_COLS = [
        # 基础价量
        "open", "close", "high", "low", "vol", "amount", "pct_chg",
        # 你已有因子
        "amount_ratio", "ret_5", "ret_10", "ret20",
        "breakout_20", "dist_to_hh20", "atr_pct",
        "upper_shadow_ratio", "close_strength",
        # 新增量能/位置/形态
        "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
        "amount_ratio_5", "amount_ratio_10", "amount_ratio_20",
        "pos_20", "dist_to_ll20",
        "range_pct", "close_strength2", "upper_shadow",
        # 行为
        "is_limit_up_yday", "limit_ups_5", "limit_ups_10", "limit_ups_20",
        "days_since_limit_up",
        # 元信息（基金级很常用）
        "days_listed", "is_st_name",
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + [DATE_COL, CODE_COL, "label_first"])

    pos_rate = float(df["label_first"].mean())
    print(f"all rows={len(df):,}  firstboard_pos_rate={pos_rate:.6f}")

    train_all, valid_all, test_all = time_split(df)
    print(f"split: train={len(train_all):,} valid={len(valid_all):,} test={len(test_all):,}")

    train_df = sample_for_training(train_all, neg_ratio=NEG_RATIO, seed=SEED)
    print(f"train sampled rows={len(train_df):,} pos_rate={train_df['label_first'].mean():.6f}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label_first"].astype(int)

    X_valid = valid_all[FEATURE_COLS]
    y_valid = valid_all["label_first"].astype(int)

    X_test = test_all[FEATURE_COLS]
    y_test = test_all["label_first"].astype(int)

    pos_cnt = int(train_all["label_first"].sum())
    neg_cnt = int((train_all["label_first"] == 0).sum())
    spw = (neg_cnt / max(pos_cnt, 1))

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 300,
        "min_gain_to_split": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "lambda_l1": 1.0,
        "scale_pos_weight": spw,
        "verbosity": -1,
        "seed": SEED,
        "feature_pre_filter": False,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=FEATURE_COLS)

    print("Training LightGBM (FIRST-BOARD + META model)...")
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

    valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    test_prob = model.predict(X_test, num_iteration=model.best_iteration)

    m_valid = evaluate_probs("valid_strict", y_valid, valid_prob)
    m_test = evaluate_probs("test_strict", y_test, test_prob)

    valid_eval = valid_all[[DATE_COL, CODE_COL, "label_first"]].copy()
    test_eval = test_all[[DATE_COL, CODE_COL, "label_first"]].copy()
    valid_eval["ml_prob"] = valid_prob
    test_eval["ml_prob"] = test_prob

    topk_valid_5 = topk_metrics_strict(valid_eval, "ml_prob", k=5)
    topk_test_5 = topk_metrics_strict(test_eval, "ml_prob", k=5)
    topk_test_10 = topk_metrics_strict(test_eval, "ml_prob", k=10)

    print("[valid] top5:", topk_valid_5)
    print("[test ] top5:", topk_test_5)
    print("[test ] top10:", topk_test_10)

    model.save_model(str(MODEL_PATH))

    meta = {
        "model_path": str(MODEL_PATH),
        "best_iteration": int(model.best_iteration),
        "feature_cols": FEATURE_COLS,
        "limit10": LIMIT10,
        "limit20": LIMIT20,
        "train_end": TRAIN_END,
        "valid_end": VALID_END,
        "neg_ratio_train": int(NEG_RATIO),
        "scale_pos_weight": float(spw),
        "metrics_valid": m_valid,
        "metrics_test": m_test,
        "top5_valid": topk_valid_5,
        "top5_test": topk_test_5,
        "top10_test": topk_test_10,
        "firstboard_pos_rate_all": float(pos_rate),
        "notes": "Label=next day limit-up & no limit-up in past 20 days & today not limit-up & non-ST by name & listed>=60 days. Threshold=10/20 by market_type.",
    }
    MODEL_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta : {MODEL_META_PATH}")


if __name__ == "__main__":
    main()