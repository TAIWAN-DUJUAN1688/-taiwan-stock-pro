# 台股法人籌碼雷達 V5

## 新增
- 三大法人：外資、投信、自營商
- 法人合計與連買天數
- 融資增減
- 融券增減
- 技術面 100 分
- 籌碼面 40 分
- 綜合分：技術面 60% + 籌碼面 40%
- 綜合選股 Top20
- 個股籌碼分析

## FinMind 資料集
- TaiwanStockPrice
- TaiwanStockInstitutionalInvestorsBuySellWide
- TaiwanStockMarginPurchaseShortSale
- TaiwanStockInfo

## API Token
側邊欄可選填 FinMind Token。
程式使用 `Authorization: Bearer <token>` Header，不會把 Token 寫入程式碼。

## API 使用量
候選池預設 15 檔，每檔約 3 個 API request。
如未使用 Token，請避免短時間反覆大量掃描。
