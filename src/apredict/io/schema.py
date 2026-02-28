# 中文字段映射到内部字段名

CN_TO_EN = {
    "日期": "trade_date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "换手率": "turnover_rate",
}

REQUIRED_SNAPSHOT_COLS = [
    "trade_date",
    "code",
    "open",
    "close",
    "high",
    "low",
    "vol",
    "amount",
]