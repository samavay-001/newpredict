from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterable, Dict, Any

import pandas as pd


@dataclass
class LiveTrackingConfig:
    xlsx_path: Path = Path("data/output/live_tracking.xlsx")
    sheet_name: str = "tracking"


def _normalize_code(x: Any) -> str:
    """
    统一成 6 位数字代码（000001 / 300750 等）
    兼容：000001.SZ / 000001.SH / 000001
    """
    if x is None:
        return ""
    s = str(x).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(6) if s else ""


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_tracking_df(cfg: LiveTrackingConfig) -> pd.DataFrame:
    if not cfg.xlsx_path.exists():
        return pd.DataFrame(columns=[
            "predict_date", "target_date", "rank",
            "code", "name",
            "rank_score", "ml_prob", "prob_calibrated", "star",
            "close_pred", "pct_chg_pred",
            "close_real", "pct_chg_real", "is_limit_up_real", "hit",
            "updated_at",
        ])

    try:
        df = pd.read_excel(cfg.xlsx_path, sheet_name=cfg.sheet_name, engine="openpyxl")
    except Exception:
        # 兜底：sheet 不存在或格式异常时，按空表处理
        df = pd.DataFrame()

    if df is None or df.empty:
        df = pd.DataFrame(columns=[
            "predict_date", "target_date", "rank",
            "code", "name",
            "rank_score", "ml_prob", "prob_calibrated", "star",
            "close_pred", "pct_chg_pred",
            "close_real", "pct_chg_real", "is_limit_up_real", "hit",
            "updated_at",
        ])

    # 规范列
    for c in ["code", "predict_date", "target_date"]:
        if c not in df.columns:
            df[c] = ""
    df["code"] = df["code"].apply(_normalize_code)
    return df


def _write_tracking_df(cfg: LiveTrackingConfig, df: pd.DataFrame) -> None:
    _ensure_parent_dir(cfg.xlsx_path)
    with pd.ExcelWriter(cfg.xlsx_path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, sheet_name=cfg.sheet_name, index=False)


def _compute_is_limit_up_from_snapshot(row: pd.Series) -> Optional[int]:
    """
    优先使用快照内的 is_limit_up_today；
    没有就用 pct_chg 粗略判断（>= 9.8% 视为涨停，ST/北交所等规则差异这里不细分）。
    """
    if "is_limit_up_today" in row.index and pd.notna(row.get("is_limit_up_today")):
        try:
            return int(row.get("is_limit_up_today"))
        except Exception:
            pass

    pct = row.get("pct_chg", None)
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return None
    try:
        return 1 if float(pct) >= 9.8 else 0
    except Exception:
        return None


def update_realized_for_target_date(
    snapshot_today: pd.DataFrame,
    today_trade_date: str,
    cfg: LiveTrackingConfig = LiveTrackingConfig(),
    verbose: bool = True,
) -> int:
    """
    用“今天收盘快照”（today_trade_date）回填历史预测中 target_date==today_trade_date 的行。
    返回：成功回填的行数
    """
    df = _read_tracking_df(cfg)
    if df.empty:
        return 0

    snap = snapshot_today.copy()
    if "code" not in snap.columns:
        return 0

    snap["code"] = snap["code"].apply(_normalize_code)
    snap = snap.set_index("code", drop=False)

    mask = (df["target_date"].astype(str) == str(today_trade_date))
    # 只更新尚未回填的行（close_real 为空）
    if "close_real" in df.columns:
        mask = mask & (df["close_real"].isna() | (df["close_real"].astype(str) == "") )

    to_update_idx = df.index[mask].tolist()
    if not to_update_idx:
        if verbose:
            print(f"[LiveTracking] 无需回填：target_date={today_trade_date}")
        return 0

    updated = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in to_update_idx:
        code = _normalize_code(df.at[i, "code"])
        if not code or code not in snap.index:
            continue

        r = snap.loc[code]
        close_real = r.get("close", None)
        pct_real = r.get("pct_chg", None)
        is_lu = _compute_is_limit_up_from_snapshot(r)

        df.at[i, "close_real"] = close_real
        df.at[i, "pct_chg_real"] = pct_real
        df.at[i, "is_limit_up_real"] = is_lu
        df.at[i, "hit"] = (1 if is_lu == 1 else 0) if is_lu is not None else None
        df.at[i, "updated_at"] = now
        updated += 1

    if updated > 0:
        _write_tracking_df(cfg, df)
        if verbose:
            print(f"[LiveTracking] 回填完成：target_date={today_trade_date} updated={updated}")

    return updated


def append_predictions(
    preds_topk: pd.DataFrame,
    predict_date: str,
    target_date: str,
    cfg: LiveTrackingConfig = LiveTrackingConfig(),
    verbose: bool = True,
) -> int:
    """
    把当天预测的 TopK 追加到 Excel。
    去重策略：同一 predict_date + target_date + code 不重复追加。
    """
    df = _read_tracking_df(cfg)

    p = preds_topk.copy()
    if "code" not in p.columns:
        raise ValueError("preds_topk 必须包含 code 列")

    p["code"] = p["code"].apply(_normalize_code)

    # 兼容你现在输出列名（尽量不强依赖）
    def pick(col: str) -> Optional[str]:
        return col if col in p.columns else None

    col_name = pick("name")
    col_rank = pick("rank")
    col_rank_score = pick("rank_score")
    col_ml = pick("ml_prob")
    col_cal = pick("prob_calibrated")
    col_star = pick("star") or pick("推荐星级")
    col_close = pick("close")
    col_pct = pick("pct_chg")

    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in p.iterrows():
        code = _normalize_code(row.get("code"))
        if not code:
            continue

        rows.append({
            "predict_date": str(predict_date),
            "target_date": str(target_date),
            "rank": int(row[col_rank]) if col_rank and pd.notna(row.get(col_rank)) else None,
            "code": code,
            "name": row.get(col_name) if col_name else None,
            "rank_score": float(row[col_rank_score]) if col_rank_score and pd.notna(row.get(col_rank_score)) else None,
            "ml_prob": float(row[col_ml]) if col_ml and pd.notna(row.get(col_ml)) else None,
            "prob_calibrated": float(row[col_cal]) if col_cal and pd.notna(row.get(col_cal)) else None,
            "star": row.get(col_star) if col_star else None,
            "close_pred": float(row[col_close]) if col_close and pd.notna(row.get(col_close)) else None,
            "pct_chg_pred": float(row[col_pct]) if col_pct and pd.notna(row.get(col_pct)) else None,
            "close_real": None,
            "pct_chg_real": None,
            "is_limit_up_real": None,
            "hit": None,
            "updated_at": now,
        })

    if not rows:
        return 0

    add_df = pd.DataFrame(rows)

    # 去重：predict_date + target_date + code
    key_cols = ["predict_date", "target_date", "code"]
    if all(c in df.columns for c in key_cols):
        before = len(df)
        df = pd.concat([df, add_df], ignore_index=True)
        df = df.drop_duplicates(subset=key_cols, keep="first")
        added = len(df) - before
    else:
        df = pd.concat([df, add_df], ignore_index=True)
        added = len(add_df)

    _write_tracking_df(cfg, df)

    if verbose:
        print(f"[LiveTracking] 追加预测：predict_date={predict_date} target_date={target_date} added={added}")

    return int(added)