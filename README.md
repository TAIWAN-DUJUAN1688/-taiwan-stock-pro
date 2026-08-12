# 台股強勢分析 MVP V4

## 新增功能
- 今日強勢股 Top20
- 先以最近交易日「成交金額高」的股票建立高流動性候選池
- 候選池可選 30 / 40 / 50 / 60 檔
- 對候選股抓取歷史日線，計算：
  - MA5 / MA10 / MA20 / MA60
  - KD
  - MACD
  - RSI
  - 成交量 / 5日均量
  - 20日突破
  - 100分技術評分
- Top20 依評分、漲跌幅、量比排序
- 保留 V3 個股分析頁

## 重要說明
V4 MVP 為避免 FinMind API 用量過高，Top20 並不是逐一掃描所有台股，
而是先從最近交易日的高流動性股票建立候選池，再做技術排名。

FinMind 官方文件顯示 API 有 request 次數限制；
後續若要做真正「全市場掃描」，建議加入 FinMind token、
每日資料快取與資料庫，避免每次開網頁都重新呼叫大量 API。

## 啟動
```bash
pip install -r requirements.txt
streamlit run app.py
```
