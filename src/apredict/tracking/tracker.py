import os
import pandas as pd
from datetime import datetime


def append_tracking(pred: pd.DataFrame, tracking_path: str):
    """
    将当天预测追加写入 tracking.csv
    - 同一天同 code 去重（保留最后一次运行）
    - 预留标签 reached_limit_up（后续你手动/脚本补）
    """
    os.makedirs(os.path.dirname(tracking_path), exist_ok=True)

    df = pred.copy()
    df["run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "reached_limit_up" not in df.columns:
        df["reached_limit_up"] = pd.NA

    if os.path.exists(tracking_path):
        old = pd.read_csv(tracking_path, encoding="utf-8-sig")
        out = pd.concat([old, df], ignore_index=True)
        if "trade_date" in out.columns and "code" in out.columns:
            out = out.drop_duplicates(subset=["trade_date", "code"], keep="last")
    else:
        out = df

    out.to_csv(tracking_path, index=False, encoding="utf-8-sig")