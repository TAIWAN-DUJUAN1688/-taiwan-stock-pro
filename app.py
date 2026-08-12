
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股獲利策略雷達 V12.1", layout="wide")

st.title("📈 台股獲利策略雷達 V12.1")
st.caption("▲ 買進訊號｜▼ 賣出訊號｜日線 / 週線切換｜停損價｜法人＋技術面｜FinMind")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# ==================== API ====================
def api_get(params, token="", timeout=30):
    headers = {}
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind API 錯誤：{payload.get('status')} / {payload.get('msg')}")
    return payload

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info(token=""):
    payload = api_get({"dataset": "TaiwanStockInfo"}, token)
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df
    df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_price_history(symbol, days=220, token=""):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    payload = api_get({
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }, token)

    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
        "Trading_money": "TradingMoney",
    })

    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume", "TradingMoney"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
          .sort_values("Date")
          .drop_duplicates("Date", keep="last")
          .set_index("Date")
    )
    df["Volume"] = df["Volume"].fillna(0)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_institutional(symbol, days=30, token=""):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    payload = api_get({
        "dataset": "TaiwanStockInstitutionalInvestorsBuySellWide",
        "data_id": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }, token)

    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    needed = [
        "Foreign_Investor_buy","Foreign_Investor_sell",
        "Investment_Trust_buy","Investment_Trust_sell",
        "Dealer_buy","Dealer_sell",
        "Dealer_self_buy","Dealer_self_sell",
        "Dealer_Hedging_buy","Dealer_Hedging_sell",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ForeignNet"] = df["Foreign_Investor_buy"] - df["Foreign_Investor_sell"]
    df["TrustNet"] = df["Investment_Trust_buy"] - df["Investment_Trust_sell"]
    df["DealerNet"] = (
        (df["Dealer_buy"] - df["Dealer_sell"]) +
        (df["Dealer_self_buy"] - df["Dealer_self_sell"]) +
        (df["Dealer_Hedging_buy"] - df["Dealer_Hedging_sell"])
    )
    df["TotalInstNet"] = df["ForeignNet"] + df["TrustNet"] + df["DealerNet"]

    return df.sort_values("date")

@st.cache_data(ttl=1800, show_spinner=False)
def get_margin(symbol, days=30, token=""):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    payload = api_get({
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }, token)

    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [
        "MarginPurchaseTodayBalance","MarginPurchaseYesterdayBalance",
        "ShortSaleTodayBalance","ShortSaleYesterdayBalance",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date")

# ==================== 指標 ====================
def calc_indicators(df):
    d = df.copy()

    for n in [5, 10, 20, 60]:
        d[f"MA{n}"] = d["Close"].rolling(n).mean()

    low9 = d["Low"].rolling(9).min()
    high9 = d["High"].rolling(9).max()
    denom = (high9 - low9).replace(0, np.nan)
    rsv = (d["Close"] - low9) / denom * 100
    d["K"] = rsv.ewm(com=2, adjust=False).mean()
    d["D"] = d["K"].ewm(com=2, adjust=False).mean()

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()

    # CCI20：與參考畫面一致，加入每根K線分析明細
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    d["CCI20"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    d["VOL_MA5"] = d["Volume"].rolling(5).mean()
    d["HIGH20_PREV"] = d["High"].shift(1).rolling(20).max()
    d["LOW20_PREV"] = d["Low"].shift(1).rolling(20).min()
    d["HIGH60_PREV"] = d["High"].shift(1).rolling(60).max()
    d["LOW60_PREV"] = d["Low"].shift(1).rolling(60).min()
    return d

def consecutive_positive(series):
    n = 0
    for v in reversed(series.tolist()):
        if pd.notna(v) and v > 0:
            n += 1
        else:
            break
    return n

def technical_score(d):
    if len(d) < 61:
        return 0, []

    x = d.iloc[-1]
    p = d.iloc[-2]
    score = 0
    reasons = []

    if x["Close"] > x["MA20"]:
        score += 10; reasons.append("站上MA20")
    if x["MA20"] > p["MA20"]:
        score += 10; reasons.append("MA20向上")
    if x["Close"] > x["MA60"]:
        score += 10; reasons.append("站上MA60")
    if x["MA5"] > x["MA10"]:
        score += 10; reasons.append("MA5>MA10")
    if x["MA10"] > x["MA20"]:
        score += 10; reasons.append("MA10>MA20")

    if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0:
        if x["Volume"] > x["VOL_MA5"]:
            score += 10; reasons.append("量>5日均量")
        if x["Volume"] > x["VOL_MA5"] * 1.5:
            score += 10; reasons.append("量增1.5倍")

    if x["K"] > x["D"]:
        score += 7; reasons.append("KD偏多")
    if x["MACD"] > x["Signal"]:
        score += 7; reasons.append("MACD偏多")
    if x["RSI"] > 50:
        score += 6; reasons.append("RSI>50")
    if pd.notna(x["HIGH20_PREV"]) and x["Close"] > x["HIGH20_PREV"]:
        score += 10; reasons.append("突破20日高")

    return min(score, 100), reasons

def chip_summary(inst, margin):
    r = {
        "外資": 0.0, "投信": 0.0, "自營": 0.0, "法人合計": 0.0,
        "法人連買": 0, "融資增減": 0.0, "融券增減": 0.0,
        "籌碼分": 0, "訊號": []
    }

    if not inst.empty:
        x = inst.iloc[-1]
        r["外資"] = float(x["ForeignNet"]) / 1000
        r["投信"] = float(x["TrustNet"]) / 1000
        r["自營"] = float(x["DealerNet"]) / 1000
        r["法人合計"] = float(x["TotalInstNet"]) / 1000
        r["法人連買"] = consecutive_positive(inst["TotalInstNet"])

        if x["ForeignNet"] > 0:
            r["籌碼分"] += 10; r["訊號"].append("外資買超")
        if x["TrustNet"] > 0:
            r["籌碼分"] += 10; r["訊號"].append("投信買超")
        if x["DealerNet"] > 0:
            r["籌碼分"] += 5; r["訊號"].append("自營商買超")
        if r["法人連買"] >= 3:
            r["籌碼分"] += 5; r["訊號"].append("法人連買≥3日")

    if not margin.empty:
        x = margin.iloc[-1]
        if pd.notna(x.get("MarginPurchaseTodayBalance")) and pd.notna(x.get("MarginPurchaseYesterdayBalance")):
            r["融資增減"] = float(x["MarginPurchaseTodayBalance"] - x["MarginPurchaseYesterdayBalance"])
            if r["融資增減"] < 0:
                r["籌碼分"] += 5; r["訊號"].append("融資減少")

        if pd.notna(x.get("ShortSaleTodayBalance")) and pd.notna(x.get("ShortSaleYesterdayBalance")):
            r["融券增減"] = float(x["ShortSaleTodayBalance"] - x["ShortSaleYesterdayBalance"])
            if r["融券增減"] > 0:
                r["籌碼分"] += 5; r["訊號"].append("融券增加")

    r["籌碼分"] = min(r["籌碼分"], 40)
    return r

# ==================== 實戰參考 ====================
def price_levels(d):
    x = d.iloc[-1]
    close = float(x["Close"])
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) else close * 0.02

    support_candidates = [
        x.get("MA5"), x.get("MA10"), x.get("MA20"),
        x.get("LOW20_PREV"), x.get("LOW60_PREV")
    ]
    support_candidates = [
        float(v) for v in support_candidates
        if pd.notna(v) and float(v) <= close
    ]
    support = max(support_candidates) if support_candidates else close - atr

    resistance_candidates = [x.get("HIGH20_PREV"), x.get("HIGH60_PREV")]
    resistance_candidates = [
        float(v) for v in resistance_candidates
        if pd.notna(v) and float(v) > close
    ]
    resistance = min(resistance_candidates) if resistance_candidates else close + 2 * atr

    entry_low = support
    entry_high = min(close, support + 0.5 * atr)
    stop = max(0.01, support - 1.2 * atr)

    return {
        "支撐": round(support, 2),
        "壓力": round(resistance, 2),
        "觀察區低": round(entry_low, 2),
        "觀察區高": round(entry_high, 2),
        "風險線": round(stop, 2),
        "ATR": round(atr, 2),
    }

def lock_score(chip, d):
    """
    法人鎖碼代理分：不是實際持股集中度。
    僅用法人連買、法人淨買超、融資變化與技術結構做代理評估。
    """
    s = 0
    x = d.iloc[-1]

    if chip["法人合計"] > 0: s += 25
    if chip["法人連買"] >= 3: s += 25
    if chip["投信"] > 0: s += 15
    if chip["外資"] > 0: s += 10
    if chip["融資增減"] < 0: s += 10
    if x["Close"] > x["MA20"]: s += 10
    if x["Volume"] > x["VOL_MA5"]: s += 5

    return min(s, 100)

def overall_score(t_score, chip_score, lock):
    # 技術 50%、籌碼 35%、籌碼強度 15%
    return round(t_score * 0.50 + chip_score * 0.875 + lock * 0.15, 1)

def verdict(score):
    if score >= 82: return "🔥 優先觀察"
    if score >= 70: return "🟢 偏多"
    if score >= 58: return "🟡 等待"
    if score >= 45: return "⚠️ 偏弱"
    return "🔴 避開"

WATCHLIST = [
    "2330","2317","2454","2382","3231","2308","2303","2881","2882","2886",
    "2891","2884","2885","2880","2883","2892","2887","1301","1303","2002",
    "1216","2207","2412","3711","2379","3034","6669","3008","2327","2357",
    "2345","2360","2356","2376","2383","2395","2408","2449","2603","2609"
]


# ==================== V7 決策引擎 ====================
def trade_plan(d, levels, total_score, chip):
    """產生條件式交易計畫；不是保證買賣價。"""
    x = d.iloc[-1]
    close = float(x["Close"])
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) and x["ATR14"] > 0 else close * 0.02
    support = float(levels["支撐"])
    resistance1 = float(levels["壓力"])

    # 拉回區縮窄：靠近短中期支撐，避免 V6 區間過寬
    pull_low = max(support, close - 0.65 * atr)
    pull_high = min(close, support + 0.30 * atr)
    if pull_high < pull_low:
        pull_high = pull_low

    # 突破條件：突破近期壓力且量能至少達 5 日均量 1.2 倍
    breakout = max(close, resistance1)
    resistance2 = max(resistance1 + atr, close + 1.5 * atr)

    # 停損價：跌破支撐約 0.8 ATR
    invalid = max(0.01, support - 0.8 * atr)

    # 以拉回區中值作為風報計算基準
    ref_entry = (pull_low + pull_high) / 2
    risk = max(ref_entry - invalid, 0.01)
    reward = max(resistance1 - ref_entry, 0)
    rr = reward / risk if risk > 0 else 0

    if total_score >= 82 and chip["法人合計"] > 0 and chip["法人連買"] >= 3:
        action = "🔥 優先觀察"
    elif total_score >= 70 and chip["法人合計"] > 0:
        action = "🟢 偏多等待買點"
    elif total_score >= 58:
        action = "🟡 等待確認"
    else:
        action = "🔴 暫不追價"

    return {
        "決策": action,
        "拉回觀察區": f"{pull_low:.2f}~{pull_high:.2f}",
        "突破參考": round(breakout, 2),
        "第一壓力": round(resistance1, 2),
        "第二壓力": round(resistance2, 2),
        "停損價": round(invalid, 2),
        "": round(rr, 2),
        "突破條件": f"收盤/盤中有效突破 {breakout:.2f} 且量能≥5日均量1.2倍",
    }


# ==================== V8 買賣訊號引擎 ====================
def build_trade_signals(d):
    """
    只使用當日及以前資料產生歷史訊號，避免偷看未來資料。
    訊號以「收盤確認」為原則；若使用當日收盤資料，最早於下一交易日執行。
    """
    z = d.copy()
    z["BuySignal"] = False
    z["SellSignal"] = False
    z["BuyReason"] = ""
    z["SellReason"] = ""

    # 交叉與趨勢條件
    ma_cross_up = (z["MA5"] > z["MA20"]) & (z["MA5"].shift(1) <= z["MA20"].shift(1))
    ma_cross_dn = (z["MA5"] < z["MA20"]) & (z["MA5"].shift(1) >= z["MA20"].shift(1))
    macd_up = (z["MACD"] > z["Signal"]) & (z["MACD"].shift(1) <= z["Signal"].shift(1))
    macd_dn = (z["MACD"] < z["Signal"]) & (z["MACD"].shift(1) >= z["Signal"].shift(1))
    kd_up = (z["K"] > z["D"]) & (z["K"].shift(1) <= z["D"].shift(1))
    kd_dn = (z["K"] < z["D"]) & (z["K"].shift(1) >= z["D"].shift(1))
    breakout = (z["Close"] > z["HIGH20_PREV"]) & (z["Volume"] >= z["VOL_MA5"] * 1.2)
    trend_ok = (z["Close"] > z["MA20"]) & (z["MA20"] >= z["MA20"].shift(1))
    weak_break = (z["Close"] < z["MA20"]) & (z["Close"].shift(1) >= z["MA20"].shift(1))
    rsi_ok = z["RSI"].between(50, 75)
    rsi_hot = z["RSI"] >= 78

    # 買進分數：至少 3 個條件，且趨勢需站上 MA20。
    buy_points = (
        ma_cross_up.astype(int) * 2 +
        macd_up.astype(int) * 2 +
        kd_up.astype(int) +
        breakout.astype(int) * 2 +
        rsi_ok.astype(int) +
        (z["Volume"] > z["VOL_MA5"]).astype(int)
    )
    raw_buy = trend_ok & (buy_points >= 3)

    # 賣出：趨勢破壞、MACD/均線死叉，或過熱後轉弱。
    sell_points = (
        ma_cross_dn.astype(int) * 2 +
        macd_dn.astype(int) * 2 +
        kd_dn.astype(int) +
        weak_break.astype(int) * 2 +
        rsi_hot.astype(int)
    )
    raw_sell = (sell_points >= 3) | (weak_break & macd_dn)

    # 去除連續重複訊號：只有從 False -> True 時畫一次。
    z["BuySignal"] = raw_buy & ~raw_buy.shift(1, fill_value=False)
    z["SellSignal"] = raw_sell & ~raw_sell.shift(1, fill_value=False)

    for i in range(1, len(z)):
        if z["BuySignal"].iat[i]:
            rs = []
            if bool(ma_cross_up.iat[i]): rs.append("MA5上穿MA20")
            if bool(macd_up.iat[i]): rs.append("MACD黃金交叉")
            if bool(kd_up.iat[i]): rs.append("KD黃金交叉")
            if bool(breakout.iat[i]): rs.append("20日突破＋量增")
            if bool(rsi_ok.iat[i]): rs.append("RSI多方")
            z.iat[i, z.columns.get_loc("BuyReason")] = "、".join(rs[:4])
        if z["SellSignal"].iat[i]:
            rs = []
            if bool(ma_cross_dn.iat[i]): rs.append("MA5跌破MA20")
            if bool(macd_dn.iat[i]): rs.append("MACD死叉")
            if bool(kd_dn.iat[i]): rs.append("KD死叉")
            if bool(weak_break.iat[i]): rs.append("跌破MA20")
            if bool(rsi_hot.iat[i]): rs.append("RSI過熱")
            z.iat[i, z.columns.get_loc("SellReason")] = "、".join(rs[:4])
    return z

def current_signal_card(d, plan):
    x = d.iloc[-1]
    p = d.iloc[-2]
    close = float(x["Close"])
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) and x["ATR14"] > 0 else close * 0.02

    breakout = float(plan["突破參考"])
    stop = float(plan["停損價"])

    # 目標價以突破價/現價為基準，避免第一壓力與突破價完全重複。
    entry_ref = max(close, breakout)
    target1 = max(entry_ref + atr, float(plan["第一壓力"]))
    target2 = max(entry_ref + 2 * atr, float(plan["第二壓力"]))

    buy_now = (
        close > x["MA20"] and x["MA5"] > x["MA20"] and
        x["MACD"] > x["Signal"] and x["RSI"] >= 50 and x["RSI"] < 75
    )
    sell_now = (
        close < x["MA20"] or
        (x["MACD"] < x["Signal"] and x["MA5"] < x["MA20"]) or
        close <= stop
    )

    if sell_now:
        state = "🔴 賣出／退出訊號"
        instruction = f"收盤跌破關鍵趨勢或停損價 {stop:.2f}，優先控管風險。"
    elif buy_now:
        state = "🟢 偏多，等待買點"
        instruction = (
            f"方案A：拉回 {plan['拉回觀察區']} 止穩；"
            f"方案B：突破 {breakout:.2f} 且量能≥5日均量1.2倍。"
        )
    else:
        state = "🟡 觀望"
        instruction = "條件尚未完整，不追價；等待拉回止穩或突破放量確認。"

    risk = max(entry_ref - stop, 0.01)
    rr1 = max(target1 - entry_ref, 0) / risk
    rr2 = max(target2 - entry_ref, 0) / risk

    return {
        "目前訊號": state,
        "操作條件": instruction,
        "參考進場": round(entry_ref, 2),
        "停損價": round(stop, 2),
        "目標": round(target1, 2),
        "目標": round(target2, 2),
        "1": round(rr1, 2),
        "2": round(rr2, 2),
    }


# ==================== V9 買賣決策引擎 ====================
def action_decision(d, signal_card, chip):
    """
    將 V8 訊號轉成更直覺的四種目前動作：
    買進 / 等待 / 減碼 / 賣出。
    """
    x = d.iloc[-1]
    p = d.iloc[-2]

    close = float(x["Close"])
    ma20 = float(x["MA20"]) if pd.notna(x["MA20"]) else close
    rsi = float(x["RSI"]) if pd.notna(x["RSI"]) else 50
    vol_ma5 = float(x["VOL_MA5"]) if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0 else float(x["Volume"])
    vol_ratio = float(x["Volume"]) / vol_ma5 if vol_ma5 > 0 else 1.0

    entry = float(signal_card["參考進場"])
    stop = float(signal_card["停損價"])
    target1 = float(signal_card["目標"])
    target2 = float(signal_card["目標"])

    near_entry = abs(close - entry) / max(entry, 0.01) <= 0.015
    breakout_ok = close >= entry and vol_ratio >= 1.2
    trend_ok = close > ma20 and x["MA5"] > x["MA20"] and x["MACD"] > x["Signal"]
    chip_ok = chip["法人合計"] > 0 or chip["法人連買"] >= 3

    # 優先順序：賣出 > 減碼 > 買進 > 等待
    if close <= stop or (close < ma20 and x["MACD"] < x["Signal"]):
        action = "🔴 賣出"
        reason = "跌破停損價或中期趨勢轉弱，優先控制風險。"
    elif close >= target1 or rsi >= 78:
        action = "🟠 減碼"
        reason = "接近/突破目標或動能過熱，可分批調節。"
    elif trend_ok and chip_ok and (near_entry or breakout_ok):
        action = "🟢 買進"
        reason = "趨勢、籌碼與價量條件同時轉強，進入可執行區。"
    else:
        action = "🟡 等待"
        reason = "條件偏多但尚未到理想買點，不追價，等待拉回或突破確認。"

    distance_pct = (entry / close - 1) * 100 if close > 0 else 0

    return {
        "目前動作": action,
        "動作原因": reason,
        "距離買進價%": round(distance_pct, 2),
        "量比": round(vol_ratio, 2),
        "參考買進": round(entry, 2),
        "停損價": round(stop, 2),
        "目標": round(target1, 2),
        "目標": round(target2, 2),
        "1": signal_card["1"],
        "2": signal_card["2"],
    }



# ==================== V10 日線 / 週線與交替買賣訊號 ====================
def to_weekly(df):
    """將日線轉成週線（週五收盤週期）。"""
    if df.empty:
        return df
    w = pd.DataFrame({
        "Open": df["Open"].resample("W-FRI").first(),
        "High": df["High"].resample("W-FRI").max(),
        "Low": df["Low"].resample("W-FRI").min(),
        "Close": df["Close"].resample("W-FRI").last(),
        "Volume": df["Volume"].resample("W-FRI").sum(),
    }).dropna(subset=["Open", "High", "Low", "Close"])
    return w

def build_alternating_signals(d):
    """
    交替式訊號：
    空手時只允許出現 ▲ 買進；
    持有後只允許出現 ▼ 賣出；
    避免同一段行情重複出現多個買點或賣點。
    """
    z = d.copy()

    ma_up = (z["MA5"] > z["MA20"]) & (z["MA5"].shift(1) <= z["MA20"].shift(1))
    ma_dn = (z["MA5"] < z["MA20"]) & (z["MA5"].shift(1) >= z["MA20"].shift(1))
    macd_bull = z["MACD"] > z["Signal"]
    macd_bear = z["MACD"] < z["Signal"]
    kd_bull = z["K"] > z["D"]
    kd_bear = z["K"] < z["D"]
    trend_bull = (z["Close"] > z["MA20"]) & (z["MA20"] >= z["MA20"].shift(1))
    break20 = (z["Close"] > z["HIGH20_PREV"]) & (z["Volume"] >= z["VOL_MA5"] * 1.15)
    trend_break = z["Close"] < z["MA20"]

    z["TradeSignal"] = ""
    z["TradeReason"] = ""
    z["TradePrice"] = np.nan
    z["TradeStop"] = np.nan

    in_position = False
    entry_stop = np.nan

    for i in range(1, len(z)):
        row = z.iloc[i]

        # 買進：MA5 上穿 MA20 為主，MACD / KD / 突破至少再有一項確認。
        confirmations = int(bool(macd_bull.iloc[i])) + int(bool(kd_bull.iloc[i])) + int(bool(break20.iloc[i]))
        buy_ok = bool(ma_up.iloc[i] and trend_bull.iloc[i] and confirmations >= 1)

        # 另外允許「突破放量」型買點，但必須整體趨勢多頭。
        breakout_buy = bool(break20.iloc[i] and trend_bull.iloc[i] and macd_bull.iloc[i])

        # 賣出：MA5 跌破 MA20，或跌破 MA20 且 MACD 轉空。
        sell_ok = bool(ma_dn.iloc[i] or (trend_break.iloc[i] and macd_bear.iloc[i]))

        if not in_position and (buy_ok or breakout_buy):
            reasons = []
            if ma_up.iloc[i]: reasons.append("MA5上穿MA20")
            if break20.iloc[i]: reasons.append("突破20期高點＋量能")
            if macd_bull.iloc[i]: reasons.append("MACD偏多")
            if kd_bull.iloc[i]: reasons.append("KD偏多")

            atr = row["ATR14"] if pd.notna(row["ATR14"]) and row["ATR14"] > 0 else row["Close"] * 0.02
            base_support = row["MA20"] if pd.notna(row["MA20"]) else row["Close"]
            entry_stop = max(0.01, float(base_support) - 0.8 * float(atr))

            z.iat[i, z.columns.get_loc("TradeSignal")] = "BUY"
            z.iat[i, z.columns.get_loc("TradeReason")] = "、".join(reasons[:4])
            z.iat[i, z.columns.get_loc("TradePrice")] = float(row["Close"])
            z.iat[i, z.columns.get_loc("TradeStop")] = entry_stop
            in_position = True

        elif in_position:
            stop_hit = pd.notna(entry_stop) and float(row["Close"]) <= float(entry_stop)

            if sell_ok or stop_hit:
                reasons = []
                if stop_hit: reasons.append("跌破停損價")
                if ma_dn.iloc[i]: reasons.append("MA5跌破MA20")
                if trend_break.iloc[i]: reasons.append("跌破MA20")
                if macd_bear.iloc[i]: reasons.append("MACD轉空")
                if kd_bear.iloc[i]: reasons.append("KD轉弱")

                z.iat[i, z.columns.get_loc("TradeSignal")] = "SELL"
                z.iat[i, z.columns.get_loc("TradeReason")] = "、".join(reasons[:4])
                z.iat[i, z.columns.get_loc("TradePrice")] = float(row["Close"])
                z.iat[i, z.columns.get_loc("TradeStop")] = entry_stop
                in_position = False
                entry_stop = np.nan

    # 最新交易狀態
    signals = z[z["TradeSignal"] != ""]
    if signals.empty:
        latest_state = "🟡 尚無完整買賣訊號"
        current_stop = np.nan
    else:
        last_signal = signals.iloc[-1]
        if last_signal["TradeSignal"] == "BUY":
            latest_state = "🟢 持有 / 等待賣出訊號"
            current_stop = float(last_signal["TradeStop"]) if pd.notna(last_signal["TradeStop"]) else np.nan
        else:
            latest_state = "⚪ 空手 / 等待買進訊號"
            current_stop = np.nan

    return z, latest_state, current_stop



# ==================== V11 穩健買賣策略 ====================
def weekly_trend_filter(daily_df):
    """週線只負責定方向，不直接當日線進場點。"""
    w = to_weekly(daily_df)
    if len(w) < 25:
        return "UNKNOWN", w
    w = calc_indicators(w)
    x = w.iloc[-1]
    bullish = (
        pd.notna(x["MA10"]) and pd.notna(x["MA20"]) and
        x["Close"] > x["MA20"] and
        x["MA10"] > x["MA20"] and
        x["MACD"] >= x["Signal"]
    )
    bearish = (
        pd.notna(x["MA10"]) and pd.notna(x["MA20"]) and
        x["Close"] < x["MA20"] and
        x["MA10"] < x["MA20"] and
        x["MACD"] < x["Signal"]
    )
    if bullish:
        return "BULL", w
    if bearish:
        return "BEAR", w
    return "NEUTRAL", w

def robust_trade_signals(daily_df, inst=None, margin=None):
    """
    V11：
    1. 週線定方向
    2. 日線找進場
    3. MA10 做短線確認
    4. 法人籌碼過濾
    5. 盤整區禁止交易
    6. ATR 停損
    訊號以收盤確認，下一交易日開盤才視為可執行。
    """
    d = calc_indicators(daily_df.copy())
    weekly_state, weekly = weekly_trend_filter(daily_df)

    # 法人過濾：資料不足時不硬判負面
    chip = chip_summary(inst if inst is not None else pd.DataFrame(),
                        margin if margin is not None else pd.DataFrame())
    chip_ok = True
    if inst is not None and not inst.empty:
        recent = inst.tail(3)["TotalInstNet"].sum()
        chip_ok = recent >= 0 or chip["法人連買"] >= 2

    # 盤整判定：MA10/MA20 太接近 + ATR 太小，避免來回洗
    d["ATRpct"] = d["ATR14"] / d["Close"].replace(0, np.nan)
    d["MAgap"] = (d["MA10"] - d["MA20"]).abs() / d["Close"].replace(0, np.nan)
    d["Sideways"] = (d["MAgap"] < 0.008) & (d["ATRpct"] < 0.025)

    d["V11Signal"] = ""
    d["V11Reason"] = ""
    d["V11Stop"] = np.nan
    d["V11ExecPrice"] = np.nan

    in_position = False
    stop = np.nan

    for i in range(61, len(d) - 1):
        x = d.iloc[i]
        prev = d.iloc[i-1]
        next_open = float(d.iloc[i+1]["Open"])

        # 週線歷史方向：只使用截至當日可取得的週資料，避免偷看未來
        hist = daily_df.loc[:d.index[i]]
        hist_week_state, _ = weekly_trend_filter(hist)

        trend_ok = (
            hist_week_state == "BULL" and
            x["Close"] > x["MA20"] and
            x["MA10"] > x["MA20"] and
            x["MA20"] >= prev["MA20"]
        )
        momentum_ok = (
            x["MACD"] >= x["Signal"] and
            48 <= x["RSI"] <= 72
        )
        volume_ok = (
            pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0 and
            x["Volume"] >= x["VOL_MA5"] * 0.9
        )

        # 兩種買法：拉回 MA10 止穩 / 突破 20 日高
        pullback = (
            x["Low"] <= x["MA10"] * 1.01 and
            x["Close"] >= x["MA10"] and
            x["Close"] > prev["Close"]
        )
        breakout = (
            pd.notna(x["HIGH20_PREV"]) and
            x["Close"] > x["HIGH20_PREV"] and
            x["Volume"] >= x["VOL_MA5"] * 1.2
        )

        buy_ok = (
            (not in_position) and
            trend_ok and momentum_ok and volume_ok and chip_ok and
            (not bool(x["Sideways"])) and
            (pullback or breakout)
        )

        if buy_ok:
            atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) and x["ATR14"] > 0 else float(x["Close"]) * 0.02
            structural = min(float(x["MA20"]), float(x["Low"]))
            stop = max(0.01, structural - 1.0 * atr)

            reasons = ["週線多頭", "MA10>MA20", "MACD偏多", "法人未明顯轉空"]
            if pullback: reasons.append("拉回MA10止穩")
            if breakout: reasons.append("突破20日高＋放量")

            d.iat[i, d.columns.get_loc("V11Signal")] = "BUY"
            d.iat[i, d.columns.get_loc("V11Reason")] = "、".join(reasons)
            d.iat[i, d.columns.get_loc("V11Stop")] = stop
            d.iat[i, d.columns.get_loc("V11ExecPrice")] = next_open
            in_position = True
            continue

        if in_position:
            stop_hit = float(x["Close"]) <= float(stop)
            trend_exit = (
                x["Close"] < x["MA20"] and
                x["MA10"] < x["MA20"] and
                x["MACD"] < x["Signal"]
            )
            weekly_exit = hist_week_state == "BEAR"

            if stop_hit or trend_exit or weekly_exit:
                reasons = []
                if stop_hit: reasons.append("跌破停損價")
                if trend_exit: reasons.append("日線趨勢轉弱")
                if weekly_exit: reasons.append("週線翻空")

                d.iat[i, d.columns.get_loc("V11Signal")] = "SELL"
                d.iat[i, d.columns.get_loc("V11Reason")] = "、".join(reasons)
                d.iat[i, d.columns.get_loc("V11Stop")] = stop
                d.iat[i, d.columns.get_loc("V11ExecPrice")] = next_open
                in_position = False
                stop = np.nan

    return d, weekly_state, chip

def backtest_v11(d, fee_rate=0.001425, tax_rate=0.003):
    """
    簡易 long-only 回測：
    訊號日收盤確認，下一交易日開盤成交。
    買進扣手續費，賣出扣手續費與證交稅。
    """
    trades = []
    entry = None

    sig = d[d["V11Signal"] != ""]
    for idx, row in sig.iterrows():
        if row["V11Signal"] == "BUY" and entry is None:
            entry = {
                "signal_date": idx,
                "price": float(row["V11ExecPrice"]),
                "stop": float(row["V11Stop"]) if pd.notna(row["V11Stop"]) else np.nan,
            }
        elif row["V11Signal"] == "SELL" and entry is not None:
            exit_price = float(row["V11ExecPrice"])
            gross = exit_price / entry["price"] - 1
            net = (exit_price * (1-fee_rate-tax_rate)) / (entry["price"] * (1+fee_rate)) - 1
            trades.append({
                "買進訊號日": entry["signal_date"],
                "賣出訊號日": idx,
                "買進執行價": entry["price"],
                "賣出執行價": exit_price,
                "淨報酬%": net * 100,
            })
            entry = None

    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"交易次數":0, "勝率":0, "總淨報酬":0, "ProfitFactor":0, "最大單筆虧損":0}

    wins = t[t["淨報酬%"] > 0]["淨報酬%"]
    losses = t[t["淨報酬%"] < 0]["淨報酬%"]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and abs(losses.sum()) > 0 else (999 if len(wins) else 0)

    # 複利總報酬
    total = ((1 + t["淨報酬%"]/100).prod() - 1) * 100
    stats = {
        "交易次數": len(t),
        "勝率": round((t["淨報酬%"] > 0).mean()*100, 1),
        "總淨報酬": round(total, 1),
        "ProfitFactor": round(float(pf), 2),
        "最大單筆虧損": round(float(t["淨報酬%"].min()), 1),
    }
    return t, stats



# ==================== V12 策略驗證層 ====================
def buy_hold_return(df, fee_rate=0.001425, tax_rate=0.003):
    if len(df) < 2:
        return 0.0
    buy = float(df.iloc[0]["Open"]) * (1 + fee_rate)
    sell = float(df.iloc[-1]["Close"]) * (1 - fee_rate - tax_rate)
    return (sell / buy - 1) * 100

def validation_grade(stats, benchmark):
    if stats["交易次數"] < 4:
        return False, "交易樣本不足"
    if stats["總淨報酬"] <= 0:
        return False, "策略歷史淨報酬未轉正"
    if stats["ProfitFactor"] < 1.15:
        return False, "Profit Factor 不足"
    if stats["勝率"] < 40:
        return False, "歷史勝率偏低"
    if stats["總淨報酬"] < benchmark:
        return False, "歷史績效未勝過買進持有"
    return True, "通過歷史策略驗證"

def latest_trade_action(d, passed):
    sig = d[d["V11Signal"] != ""]
    if not passed:
        return "⚪ 不交易", "歷史策略驗證未通過，不發新的買進訊號。"
    if sig.empty:
        return "🟡 等待", "尚未形成完整買進條件。"
    last = sig.iloc[-1]
    if last["V11Signal"] == "BUY":
        return "🟢 持有／可觀察", "最近有效訊號為買進；持有期間依停損與趨勢退出。"
    return "🟡 等待下一次買點", "最近有效訊號為賣出，目前不追價。"

# ==================== UI ====================
with st.sidebar:
    st.header("設定")
    token = st.text_input(
        "FinMind Token（選填）",
        type="password",
        help="不填也可使用；填入後通常可提高 API 使用上限。"
    )
    page = st.radio(
        "功能",
        ["🚦 明日決策 Top5", "🔥 法人＋技術雙強 Top5", "📊 綜合排行 Top20", "🔍 個股實戰分析"],
        index=0
    )

def scan_market(pool_size, token):
    info = get_stock_info(token)
    info_map = dict(zip(info["stock_id"], info["stock_name"])) if not info.empty else {}

    rows = []
    progress = st.progress(0)
    status = st.empty()

    for idx, symbol in enumerate(WATCHLIST[:pool_size], start=1):
        status.write(f"分析中：{symbol}（{idx}/{pool_size}）")

        try:
            price = get_price_history(symbol, 220, token)
            inst = get_institutional(symbol, 30, token)
            margin = get_margin(symbol, 30, token)

            if len(price) < 61:
                progress.progress(idx / pool_size)
                continue

            d = calc_indicators(price)
            t_score, t_reasons = technical_score(d)
            chip = chip_summary(inst, margin)
            lock = lock_score(chip, d)
            total = overall_score(t_score, chip["籌碼分"], lock)
            levels = price_levels(d)
            plan = trade_plan(d, levels, total, chip)

            x = d.iloc[-1]
            p = d.iloc[-2]
            pct = (x["Close"] / p["Close"] - 1) * 100
            vol_ratio = (
                x["Volume"] / x["VOL_MA5"]
                if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0
                else np.nan
            )

            rows.append({
                "股票代號": symbol,
                "名稱": info_map.get(symbol, ""),
                "收盤價": round(float(x["Close"]), 2),
                "漲跌幅%": round(float(pct), 2),
                "實戰分": total,
                "判定": verdict(total),
                "技術分": t_score,
                "籌碼分": chip["籌碼分"],
                "籌碼強度": lock,
                "外資(張)": round(chip["外資"], 0),
                "投信(張)": round(chip["投信"], 0),
                "法人連買": chip["法人連買"],
                "融資增減(張)": round(chip["融資增減"], 0),
                "量比": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
                "支撐": levels["支撐"],
                "壓力": levels["壓力"],
                "觀察區": f"{levels['觀察區低']}~{levels['觀察區高']}",
                "風險線": levels["風險線"],
                "決策": plan["決策"],
                "拉回觀察區": plan["拉回觀察區"],
                "突破參考": plan["突破參考"],
                "第一壓力": plan["第一壓力"],
                "第二壓力": plan["第二壓力"],
                "停損價": plan["停損價"],
                "": plan[""],
                "突破條件": plan["突破條件"],
                "訊號": "、".join((chip["訊號"] + t_reasons)[:6]),
            })

        except Exception:
            pass

        progress.progress(idx / pool_size)

    progress.empty()
    status.empty()
    return pd.DataFrame(rows)

if page in ["🚦 明日決策 Top5", "🔥 法人＋技術雙強 Top5", "📊 綜合排行 Top20"]:
    pool_size = st.selectbox("候選池大小", [10, 15, 20], index=1)
    run = st.button("開始掃描", type="primary")

    if run:
        with st.spinner("正在整合法人、融資、量價與技術面..."):
            out = scan_market(pool_size, token)

        if out.empty:
            st.error("本次沒有成功完成分析，可能遇到 API 使用次數限制或資料暫未更新。")
            st.stop()

        out = out.sort_values(
            ["實戰分", "籌碼強度", "技術分"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        if page == "🚦 明日決策 Top5":
            top5 = out.head(5).copy()
            top5.index = top5.index + 1

            st.subheader("🚦 明日決策 Top5")
            st.warning("此為盤後條件排序，不代表隔日必漲；請搭配開盤價、量能與大盤環境再次確認。")
            st.dataframe(
                top5[[
                    "股票代號","名稱","收盤價","實戰分","決策",
                    "外資(張)","投信(張)","法人連買","籌碼強度","量比",
                    "拉回觀察區","突破參考","停損價",""
                ]],
                use_container_width=True,
                height=330
            )

            best = top5.iloc[0]
            st.success(
                f"目前第 1 名：{best['股票代號']} {best['名稱']}｜"
                f"實戰分 {best['實戰分']}｜{best['決策']}｜"
                f"拉回區 {best['拉回觀察區']}｜突破參考 {best['突破參考']}｜停損價 {best['停損價']}"
            )

        elif page == "🔥 法人＋技術雙強 Top5":
            strong = out[
                (out["外資(張)"] + out["投信(張)"] > 0) &
                (out["技術分"] >= 60) &
                (out["籌碼強度"] >= 60)
            ].copy()
            strong = strong.sort_values(
                ["籌碼強度", "技術分", "實戰分"],
                ascending=[False, False, False]
            ).head(5)
            strong.index = range(1, len(strong) + 1)
            st.subheader("🔥 法人＋技術雙強 Top5")
            st.caption("優先篩選法人偏多、技術結構偏強且籌碼強度較高的標的。")
            if strong.empty:
                st.info("本次候選池沒有同時符合法人＋技術雙強條件的股票。")
            else:
                st.dataframe(
                    strong[[
                        "股票代號","名稱","實戰分","技術分","籌碼強度",
                        "外資(張)","投信(張)","法人連買","量比",
                        "拉回觀察區","突破參考","停損價",""
                    ]],
                    use_container_width=True,
                    height=330
                )
        else:
            top20 = out.head(20).copy()
            top20.index = top20.index + 1
            st.subheader("📊 綜合排行 Top20")
            st.dataframe(top20, use_container_width=True, height=760)

else:
    st.subheader("🔍 V11 穩健買賣分析")
    symbol = st.text_input("股票代號", value="2330", max_chars=8)
    run = st.button("開始穩健分析", type="primary")

    if run:
        try:
            daily = get_price_history(symbol, 1900, token)
            inst = get_institutional(symbol, 30, token)
            margin = get_margin(symbol, 30, token)

            if len(daily) < 120:
                st.error("歷史資料不足，暫時無法做穩健策略分析。")
                st.stop()

            d, weekly_state, chip = robust_trade_signals(daily, inst, margin)
            trades, stats = backtest_v11(d)
            benchmark = round(buy_hold_return(d), 1)
            passed, validation_reason = validation_grade(stats, benchmark)
            current_action, action_reason = latest_trade_action(d, passed)

            state_text = {
                "BULL":"🟢 週線多頭：只找日線買點",
                "NEUTRAL":"🟡 週線盤整：降低交易頻率",
                "BEAR":"🔴 週線空頭：暫停新增多單",
                "UNKNOWN":"⚪ 週線資料不足"
            }.get(weekly_state, "⚪ 未知")

            st.markdown(f"## {state_text}")

            st.subheader("💰 策略驗證結果")
            if passed:
                st.success(f"✅ 通過｜{validation_reason}")
            else:
                st.error(f"❌ 不通過｜{validation_reason}")
            st.markdown(f"## 目前動作：{current_action}")
            st.write(action_reason)
            st.caption("未通過歷史驗證時，系統不顯示新的買進訊號；任何回測都不保證未來獲利。")

            signals = d[d["V11Signal"] != ""]
            latest = signals.iloc[-1] if not signals.empty else None

            if latest is None:
                st.info("目前沒有符合 V12 完整條件的買賣訊號。")
            elif latest["V11Signal"] == "BUY":
                st.success(
                    f"最近訊號：▲ 買進｜訊號日 {latest.name.strftime('%Y-%m-%d')}｜"
                    f"下一交易日參考執行價 {latest['V11ExecPrice']:.2f}｜"
                    f"🛑 停損價 {latest['V11Stop']:.2f}"
                )
            else:
                st.warning(
                    f"最近訊號：▼ 賣出｜訊號日 {latest.name.strftime('%Y-%m-%d')}｜"
                    f"下一交易日參考執行價 {latest['V11ExecPrice']:.2f}"
                )

            st.subheader("📈 日線買賣圖")
            st.caption("週線只定方向；▲/▼ 由日線執行。紅K＝上漲、綠K＝下跌。")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                name="K線",
                increasing_line_color="#FF0000",
                increasing_fillcolor="#FF0000",
                decreasing_line_color="#00A000",
                decreasing_fillcolor="#00A000"
            ))
            fig.add_trace(go.Scatter(x=d.index, y=d["MA5"], mode="lines", name="MA5",
                                     line=dict(color="#FFD400", width=1.6)))
            fig.add_trace(go.Scatter(x=d.index, y=d["MA10"], mode="lines", name="MA10",
                                     line=dict(color="#38BDF8", width=1.8)))
            fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], mode="lines", name="MA20",
                                     line=dict(color="#A855F7", width=2.0)))

            buys = d[d["V11Signal"] == "BUY"] if passed else d.iloc[0:0]
            sells = d[d["V11Signal"] == "SELL"]

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys["Low"] - buys["ATR14"].fillna(0)*0.35,
                    mode="markers", name="▲ 買進",
                    marker=dict(symbol="triangle-up", size=15, color="#FF0000",
                                line=dict(color="white", width=1)),
                    text=buys["V11Reason"],
                    hovertemplate="%{x}<br>▲ 買進訊號<br>%{text}<extra></extra>"
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells["High"] + sells["ATR14"].fillna(0)*0.35,
                    mode="markers", name="▼ 賣出",
                    marker=dict(symbol="triangle-down", size=15, color="#00A000",
                                line=dict(color="white", width=1)),
                    text=sells["V11Reason"],
                    hovertemplate="%{x}<br>▼ 賣出訊號<br>%{text}<extra></extra>"
                ))

            fig.update_layout(
                height=720, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.08, x=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧪 歷史回測")
            st.caption("回測採訊號日收盤確認、下一交易日開盤成交；已計入預設手續費與賣出證交稅。")
            a,b,c,dcol,e = st.columns(5)
            a.metric("交易次數", stats["交易次數"])
            b.metric("勝率", f'{stats["勝率"]}%')
            c.metric("策略淨報酬", f'{stats["總淨報酬"]}%')
            dcol.metric("Profit Factor", stats["ProfitFactor"])
            e.metric("最大單筆虧損", f'{stats["最大單筆虧損"]}%')

            x1,x2 = st.columns(2)
            x1.metric("買進持有基準", f"{benchmark}%")
            alpha = round(stats["總淨報酬"] - benchmark, 1)
            x2.metric("策略超額報酬", f"{alpha:+.1f}%")

            if not trades.empty:
                show = trades.tail(20).copy()
                show["買進訊號日"] = pd.to_datetime(show["買進訊號日"]).dt.strftime("%Y-%m-%d")
                show["賣出訊號日"] = pd.to_datetime(show["賣出訊號日"]).dt.strftime("%Y-%m-%d")
                st.dataframe(show, use_container_width=True, hide_index=True)

            # ------------------------------------------------------------
            # V12.1：依參考畫面新增「每根K線策略分析明細」
            # ------------------------------------------------------------
            st.subheader("📋 每根 K 線策略分析")
            st.caption(
                "逐日顯示收盤價、成交量、MA5、MA20、CCI20、策略訊號、"
                "下一根K線模擬成交日/價、回測動作與目前部位。"
            )

            detail = d.copy()
            detail["策略訊號"] = "無新訊號"
            detail.loc[detail["V11Signal"] == "BUY", "策略訊號"] = "模擬轉多訊號"
            detail.loc[detail["V11Signal"] == "SELL", "策略訊號"] = "模擬轉空訊號"

            # 訊號在當日收盤確認，下一根 K 線開盤模擬成交
            detail["模擬成交日"] = ""
            detail["模擬成交價"] = np.nan
            detail["策略回測動作"] = "無回測動作"
            detail["策略回測部位"] = "模擬空手"

            position_state = "模擬空手"

            for i in range(len(detail)):
                sig = detail["V11Signal"].iloc[i]

                if sig == "BUY":
                    detail.iloc[i, detail.columns.get_loc("策略回測動作")] = "模擬轉多"
                elif sig == "SELL":
                    detail.iloc[i, detail.columns.get_loc("策略回測動作")] = "模擬轉空"

                # 訊號下一根 K 線才正式改變部位
                if i > 0:
                    prev_sig = detail["V11Signal"].iloc[i-1]
                    if prev_sig == "BUY":
                        position_state = "模擬多單"
                    elif prev_sig == "SELL":
                        position_state = "模擬空手"

                detail.iloc[i, detail.columns.get_loc("策略回測部位")] = position_state

                # 將當日訊號對應到下一根K線的實際模擬成交
                if sig in ("BUY", "SELL") and i + 1 < len(detail):
                    next_date = detail.index[i+1]
                    detail.iloc[i, detail.columns.get_loc("模擬成交日")] = next_date.strftime("%Y/%m/%d")
                    detail.iloc[i, detail.columns.get_loc("模擬成交價")] = float(detail["Open"].iloc[i+1])

            analysis_table = pd.DataFrame({
                "日期": detail.index.strftime("%Y/%m/%d"),
                "收盤價": detail["Close"].round(2),
                "成交量": detail["Volume"].fillna(0).astype("int64"),
                "MA5": detail["MA5"].round(2),
                "MA20": detail["MA20"].round(2),
                "CCI20": detail["CCI20"].round(2),
                "策略訊號": detail["策略訊號"],
                "模擬成交日": detail["模擬成交日"],
                "模擬成交價": detail["模擬成交價"].round(2),
                "策略回測動作": detail["策略回測動作"],
                "策略回測部位": detail["策略回測部位"],
            })

            # 預設顯示最近 80 根 K 線，表格內可捲動
            st.dataframe(
                analysis_table.tail(80),
                use_container_width=True,
                hide_index=True,
                height=610,
                column_config={
                    "日期": st.column_config.TextColumn("日期", width="small"),
                    "收盤價": st.column_config.NumberColumn("收盤價", format="%.2f"),
                    "成交量": st.column_config.NumberColumn("成交量", format="%d"),
                    "MA5": st.column_config.NumberColumn("MA5", format="%.2f"),
                    "MA20": st.column_config.NumberColumn("MA20", format="%.2f"),
                    "CCI20": st.column_config.NumberColumn("CCI20", format="%.2f"),
                    "策略訊號": st.column_config.TextColumn("策略訊號", width="medium"),
                    "模擬成交日": st.column_config.TextColumn("模擬成交日", width="small"),
                    "模擬成交價": st.column_config.NumberColumn("模擬成交價", format="%.2f"),
                    "策略回測動作": st.column_config.TextColumn("策略回測動作", width="medium"),
                    "策略回測部位": st.column_config.TextColumn("策略回測部位", width="medium"),
                }
            )

            with st.expander("欄位說明"):
                st.write("**策略訊號**：V12 原本策略當日收盤確認後產生的買進／賣出訊號。")
                st.write("**模擬成交日／價**：訊號確認後，下一根 K 線開盤價作為回測成交基準。")
                st.write("**策略回測動作**：當日是否出現模擬轉多／轉空動作。")
                st.write("**策略回測部位**：模擬成交後目前處於多單或空手狀態。")
                st.write("**CCI20**：20期商品通道指標，作為額外分析資訊；目前不改變 V12 原策略買賣規則。")

            st.subheader("V12 規則")
            st.write("① **週線定方向**：週線多頭才允許新增多單。")
            st.write("② **日線找買點**：拉回 MA10 止穩或突破 20 日高點放量。")
            st.write("③ **籌碼過濾**：法人近期不能明顯轉空。")
            st.write("④ **盤整禁做**：MA10/MA20 過度糾結、波動太低時不進場。")
            st.write("⑤ **賣出**：跌破停損價，或日線/週線趨勢明確轉弱。")
            st.warning("回測結果不代表未來績效。策略仍可能虧損；正式使用前應以不同市場階段與更多股票做樣本外測試。")

        except Exception as e:
            st.error("V12.1 分析發生錯誤。")
            st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption(
    "本系統僅供策略研究與回測，不構成投資建議。任何策略均可能虧損。拉回觀察區、突破參考、壓力、停損價與均為模型條件值，不是保證成交或獲利價位。"
)
