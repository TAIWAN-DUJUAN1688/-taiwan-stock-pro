# 台股強勢分析 MVP V2

本次修正：
- 自動嘗試 `.TW` 與 `.TWO`
- `Ticker.history()` 主抓，`yf.download()` 備援
- 增加 timeout、repair、錯誤診斷
- 預設資料期間改為 2 年
- 增加重新下載最新資料
