# 台股強勢分析 MVP V3

## 本次修正
- 移除 yfinance
- 改用 FinMind `TaiwanStockPrice`
- 直接輸入台股代號，例如 2330
- 保留 MA5 / MA10 / MA20 / MA60
- 保留 KD / MACD / RSI
- 保留成交量與 100 分技術評分
- 增加 FinMind API 錯誤診斷

## 資料來源
FinMind REST API：
`https://api.finmindtrade.com/api/v4/data`

資料集：
`TaiwanStockPrice`

## 啟動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意
本版本用於測試與學習。
正式商用前，請確認 FinMind 與相關市場資料之授權及服務條款。
