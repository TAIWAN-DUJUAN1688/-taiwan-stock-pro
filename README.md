# 台股強勢分析 MVP

## 功能
- 輸入台股代號（例如 2330）
- K線
- MA5 / MA10 / MA20 / MA60
- KD
- MACD
- RSI
- 成交量
- 100 分技術評分

## 啟動方式
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意
此版本使用 yfinance 作為測試資料來源。
若要做正式商用版，建議改接具授權的台股行情資料來源。
