import os
import glob
import argparse
import pandas as pd

ROOT = r"D:\PycharmProjects\A股涨停预测"
META_PATH = os.path.join(ROOT, "data", "meta", "stock_basic.csv")
HISTORY_DIR = os.path.join(ROOT, "data", "history")
CACHE_DIR = os.path.join(ROOT, "data", "cache_history")
CACHE_PARQUET = os.path.join(CACHE_DIR, "market_daily.parquet")

def parse_date_to_yyyymmdd(x) -> str:
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y%m%d")

def load_stock_basic(meta_path: str) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    df.columns = [c.strip() for c in df.columns]
    # 兼容你的中文表头
    rename = {"TS代码": "ts_code", "股票代码": "code", "股票名称": "name"}
    df = df.rename(columns=rename)
    need = ["ts_code", "code", "name"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"stock_basic.csv missing columns: {miss}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    return df[["code", "ts_code", "name"]]

def load_one_history_csv(path: str, code_to_ts: dict, code_to_name: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # 兼容中文列
    rename = {
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "vol",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename)

    need = ["date", "code", "open", "close", "high", "low", "vol", "amount", "pct_chg", "turnover_rate"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{os.path.basename(path)} missing columns: {miss}")

    df["trade_date"] = df["date"].apply(parse_date_to_yyyymmdd)
    df = df[df["trade_date"].notna()].copy()

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["ts_code"] = df["code"].map(code_to_ts)
    df["name"] = df["code"].map(code_to_name)

    df = df[df["ts_code"].notna()].copy()

    out = df[[
        "trade_date", "ts_code", "code", "name",
        "open", "close", "high", "low",
        "vol", "amount", "pct_chg", "turnover_rate"
    ]].copy()

    # 类型收紧（节省空间）
    for c in ["open","close","high","low","amount","pct_chg","turnover_rate"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["vol"] = pd.to_numeric(out["vol"], errors="coerce")

    out = out.dropna(subset=["close"])
    return out

def cmd_build_cache(limit_files=None):
    os.makedirs(CACHE_DIR, exist_ok=True)

    basic = load_stock_basic(META_PATH)
    code_to_ts = dict(zip(basic["code"], basic["ts_code"]))
    code_to_name = dict(zip(basic["code"], basic["name"]))

    files = glob.glob(os.path.join(HISTORY_DIR, "*_daily.csv"))
    files = sorted(files)
    if limit_files:
        files = files[:limit_files]

    chunks = []
    total = len(files)
    for i, p in enumerate(files, 1):
        try:
            one = load_one_history_csv(p, code_to_ts, code_to_name)
            chunks.append(one)
        except Exception as e:
            print(f"[WARN] skip {os.path.basename(p)}: {e}")

        if i % 200 == 0:
            print(f"[INFO] loaded {i}/{total} files...")

    if not chunks:
        raise RuntimeError("No history data loaded; check HISTORY_DIR and file format.")

    market = pd.concat(chunks, ignore_index=True)

    # 排序 + 去重：同一股票同一日保留最后一条
    market = market.sort_values(["trade_date", "ts_code"])
    market = market.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    # 保存 Parquet
    market.to_parquet(CACHE_PARQUET, index=False)
    print(f"[OK] cache written: {CACHE_PARQUET}")
    print(f"[OK] rows: {len(market):,}, dates: {market['trade_date'].nunique():,}, symbols: {market['ts_code'].nunique():,}")

def cmd_dump_snapshot(trade_date: str, out_csv: str):
    if not os.path.exists(CACHE_PARQUET):
        raise FileNotFoundError(f"Cache not found: {CACHE_PARQUET}. Run build-cache first.")
    market = pd.read_parquet(CACHE_PARQUET)
    snap = market[market["trade_date"] == trade_date].copy()
    if snap.empty:
        raise ValueError(f"No data for trade_date={trade_date} in cache.")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    snap.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] snapshot: {out_csv} rows={len(snap)}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_cache = sub.add_parser("build-cache")
    ap_cache.add_argument("--limit_files", type=int, default=None, help="debug: limit number of stock files")

    ap_snap = sub.add_parser("dump-snapshot")
    ap_snap.add_argument("--trade_date", required=True, help="YYYYMMDD")
    ap_snap.add_argument("--out", required=True, help="output csv path")

    args = ap.parse_args()

    if args.cmd == "build-cache":
        cmd_build_cache(args.limit_files)
    elif args.cmd == "dump-snapshot":
        cmd_dump_snapshot(args.trade_date, args.out)

if __name__ == "__main__":
    main()