\# 🚀 A股涨停概率预测系统 (A-Share Limit-Up Prediction System)



> 基于机器学习 + 多因子模型的量化选股系统  

> 目标：预测下一交易日最有可能涨停的股票（Top-K）



---



\# 📊 系统概述



本系统是一个完整的量化研究与预测框架，结合：



\- 多因子量化模型

\- LightGBM机器学习

\- 历史数据回测

\- 自动预测流水线



用于预测：



> 下一交易日最有可能涨停的股票候选池



适用于：



\- 量化基金

\- 私募策略研究

\- AI选股系统

\- 高频Alpha研究



---



\# 🧠 核心架构



系统采用模块化设计：





Data Layer

↓

Factor Engine

↓

Candidate Filter (Phase A)

↓

ML Model Ranking (Phase B)

↓

Prediction Output





---



\# 📁 项目结构





newpredict/

│

├─ src/apredict/

│ ├─ io/ # 数据加载模块

│ ├─ ml/ # ML模型训练和推理

│ │ ├─ train.py

│ │ ├─ predict.py

│ │ ├─ dataset.py

│ │ └─ model.py

│

├─ scripts/

│ ├─ train\_lightgbm.py

│ ├─ run\_backfill.py

│ ├─ eval\_topk\_strict.py

│

├─ tools/

│ └─ backfill\_from\_history.py

│

├─ output/

│ └─ predictions.csv

│

├─ requirements.txt

├─ pyproject.toml

└─ README.md





---



\# 🔬 核心预测流程



完整预测流水线：





加载市场数据



计算技术因子



Phase A: 候选池筛选



Phase B: ML模型预测概率



排序Top-K股票



输出预测结果





---



\# 📈 使用的核心因子



系统计算以下关键因子：



\### 动量因子

\- 5日涨幅

\- 10日涨幅

\- 20日涨幅



\### 波动因子

\- ATR波动率

\- 波动压缩



\### 资金因子

\- 成交额变化率

\- 放量倍数



\### 突破因子

\- 是否突破20日新高

\- 距离历史高点距离



\### 行为因子

\- 收盘强度

\- 上影线比例



---



\# 🤖 机器学习模型



使用：





LightGBM Gradient Boosting





优势：



\- 高性能

\- 高精度

\- 快速训练

\- 支持非线性特征



预测目标：





P(涨停 | 当前特征)





---



\# ⚙️ 安装





git clone https://github.com/samavay-001/newpredict.git



cd newpredict



pip install -r requirements.txt





---



\# ▶️ 使用方法



\## 训练模型





python scripts/train\_lightgbm.py





---



\## 运行预测





python scripts/run\_backfill.py --last 10 --topk 5





输出：





output/predictions.csv





---



\## 回测





python scripts/eval\_topk\_strict.py





---



\# 📊 输出示例





股票代码 股票名称 预测概率 排名

300750 宁德时代 0.18 1

002466 天齐锂业 0.16 2





---



\# 🎯 系统优势



✔ 多因子量化模型

✔ ML概率预测

✔ 自动回测系统

✔ 模块化架构

✔ 高扩展性



---



\# 📉 应用场景



适用于：



\- 日内交易策略

\- 涨停板策略

\- Alpha挖掘

\- 量化研究



---



\# 🔮 未来规划



计划增加：



\- 实时预测

\- 自动训练

\- Web Dashboard

\- 自动交易接口



---



\# ⚠️ 风险提示



本系统仅用于研究用途，不构成投资建议。



金融市场存在风险。



---



\# 👨‍💻 作者



samavay-001



---



\# ⭐ 如果本项目对你有帮助，请Star支持

