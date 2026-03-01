import pandas as pd

def phase_a_limitup_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    专业涨停候选池筛选（可按你市场慢慢调参）
    需要列（尽量具备）：
    amount, atr_pct, pos_20, vol_ratio_10, dist_to_hh20, limit_ups_5, is_limit_up_yday
    """
    x = df.copy()

    # 基础：成交额过滤（避免僵尸/超大盘）
    if "amount" in x.columns:
        x = x[(x["amount"] >= 8e7) & (x["amount"] <= 8e10)]

    # 波动护栏
    if "atr_pct" in x.columns:
        x = x[x["atr_pct"] <= 0.10]

    # 放量迹象
    if "vol_ratio_10" in x.columns:
        x = x[x["vol_ratio_10"] >= 1.2]

    # 避免极高位（低位启动更容易出首板）
    if "pos_20" in x.columns:
        x = x[x["pos_20"] <= 0.85]

    # 你如果做“首板”，建议：近5日无涨停 + 昨日非涨停
    if "limit_ups_5" in x.columns:
        x = x[x["limit_ups_5"] == 0]
    if "is_limit_up_yday" in x.columns:
        x = x[x["is_limit_up_yday"] == 0]

    return x