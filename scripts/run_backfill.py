import os
import subprocess
import argparse
import pandas as pd

PROJECT_ROOT = r"D:\PycharmProjects\A股涨停预测"

CACHE_PARQUET = os.path.join(PROJECT_ROOT, "data", "cache_history", "market_daily.parquet")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RUN_DAILY = os.path.join(PROJECT_ROOT, "scripts", "run_daily.py")

# raw daily 只需要这些基础列（你 cache 里就有）
RAW_COLS = [
    "trade_date", "ts_code", "code", "name",
    "open", "close", "high", "low", "vol", "amount", "pct_chg", "turnover_rate"
]

def ensure_exists(path: str):
    os.makedirs(path, exist_ok=True)

def load_trade_calendar_from_parquet(parquet_path: str) -> list[str]:
    """
    只读取 trade_date 列，获取交易日历（避免读全表）。
    """
    df = pd.read_parquet(parquet_path, columns=["trade_date"])
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    return dates

def read_snapshot_for_date(parquet_path: str, trade_date: str) -> pd.DataFrame:
    """
    读取某一天的全市场快照（优先使用 pyarrow.dataset 过滤读取，速度快很多）。
    如果没有安装 pyarrow，则回退到 pandas 全表读取再过滤（会慢）。
    """
    # 优先：pyarrow.dataset 过滤读取（秒级）
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        table = dataset.to_table(
            filter=(ds.field("trade_date") == trade_date),
            columns=RAW_COLS
        )
        snap = table.to_pandas()
    except Exception as e:
        # 回退：pandas 读列后过滤（会慢）
        df = pd.read_parquet(parquet_path, columns=RAW_COLS)
        snap = df[df["trade_date"].astype(str) == trade_date].copy()

    # 保险：去掉缺失
    snap = snap.dropna(subset=["ts_code", "close"])
    return snap

def write_raw_snapshot(snap: pd.DataFrame, trade_date: str) -> str:
    """
    写入 data/raw/daily_YYYY-MM-DD.csv，供 run_daily.py 读取
    """
    ensure_exists(RAW_DIR)
    out_path = os.path.join(
        RAW_DIR,
        f"daily_{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}.csv"
    )
    snap.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

def run_prediction(daily_path: str, topk: int = 5):
    """
    调用你现有 scripts/run_daily.py，不改原系统结构
    """
    print(f"Running prediction for daily: {daily_path}")

    import sys
    cmd = [
        sys.executable,                 # ✅ 确保使用当前 venv 的 python
        RUN_DAILY,
        "--daily-path", daily_path,     # ✅ run_daily.py 要求的参数
        "--topk", str(topk)
    ]

    # ✅ 给子进程注入 src 路径，确保 import apredict 成功
    env = os.environ.copy()
    src_path = os.path.join(PROJECT_ROOT, "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)

def already_done(trade_date: str) -> bool:
    """
    判断是否已经生成过 predictions.csv
    """
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed", trade_date)
    pred_path = os.path.join(processed_dir, "predictions.csv")
    return os.path.exists(pred_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=250, help="回放最近N个交易日（默认250）")
    ap.add_argument("--start", type=str, default=None, help="起始日期 YYYYMMDD（可选）")
    ap.add_argument("--end", type=str, default=None, help="结束日期 YYYYMMDD（可选，包含）")
    ap.add_argument("--topk", type=int, default=5, help="输出TopK（默认5）")
    args = ap.parse_args()

    if not os.path.exists(CACHE_PARQUET):
        raise FileNotFoundError(f"Cache not found: {CACHE_PARQUET}. Run build-cache first.")

    dates = load_trade_calendar_from_parquet(CACHE_PARQUET)

    # 选择日期范围
    if args.start or args.end:
        start = args.start or dates[0]
        end = args.end or dates[-1]
        dates = [d for d in dates if start <= d <= end]
    else:
        dates = dates[-args.last:]

    if not dates:
        print("[WARN] No dates selected. Check --start/--end.")
        return

    print(f"[INFO] backfill dates: {len(dates)} from {dates[0]} to {dates[-1]}")

    failed = []
    for trade_date in dates:
        if already_done(trade_date):
            print(f"Skip {trade_date}, already exists")
            continue

        try:
            snap = read_snapshot_for_date(CACHE_PARQUET, trade_date)
            if snap.empty:
                print(f"[WARN] snapshot empty for {trade_date}, skip")
                continue

            daily_path = write_raw_snapshot(snap, trade_date)
            run_prediction(daily_path, topk=args.topk)

        except Exception as e:
            print(f"[FAIL] {trade_date}: {e}")
            failed.append({"trade_date": trade_date, "error": str(e)})
            continue

    if failed:
        fail_path = os.path.join(PROJECT_ROOT, "data", "processed", "_backfill_failed.json")
        ensure_exists(os.path.dirname(fail_path))
        pd.DataFrame(failed).to_json(fail_path, force_ascii=False, orient="records", indent=2)
        print(f"[WARN] some dates failed. saved: {fail_path}")

    print("Backfill complete.")

if __name__ == "__main__":
    main()