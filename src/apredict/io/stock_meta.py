import pandas as pd


def load_stock_meta(path: str) -> pd.DataFrame:
    """
    读取股票基础信息表（支持中文列名），返回标准列：
    - ts_code: 000001.SZ
    - code:    000001
    - name:    平安银行
    - market, industry...（若存在会保留）
    """
    df = pd.read_csv(path, dtype=str)
    df = df.copy()

    # 兼容中文列名
    rename_map = {
        "TS代码": "ts_code",
        "股票代码": "code",
        "股票名称": "name",
        "所属行业": "industry",
        "市场类型": "market",
        "交易所代码": "exchange",
        "地域": "area",
        "上市日期": "list_date",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # 清洗
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.strip()
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    # 去重：按 code 保留第一条
    if "code" in df.columns:
        df = df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")

    return df


def attach_stock_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    给 snapshot/universe/features/predictions 补齐 name/ts_code 等字段
    """
    if df is None or df.empty:
        return df

    if meta is None or meta.empty:
        return df

    if "code" not in df.columns:
        return df

    cols = [c for c in ["code", "ts_code", "name", "industry", "market", "exchange", "area", "list_date"] if c in meta.columns]
    meta2 = meta[cols].copy()

    out = df.merge(meta2, on="code", how="left")
    return out


def is_st_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.upper()
    return ("ST" in n)  # 包括 *ST 和 ST