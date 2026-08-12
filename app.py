
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股日週線買賣雷達 V10.2", layout="wide")

st.title("📈 台股日週線買賣雷達 V10.2")
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
    st.subheader("🔍 個股日線 / 週線買賣訊號")

    c1, c2 = st.columns([1, 1])
    with c1:
        symbol = st.text_input("股票代號", value="2330", max_chars=8)
    with c2:
        timeframe = st.selectbox("買賣週期", ["日線", "週線"], index=0)

    run = st.button("開始分析買賣訊號", type="primary")

    if run:
        try:
            # 週線 MA60 需要較長歷史，因此週線固定抓約 5 年。
            fetch_days = 800 if timeframe == "日線" else 1900

            daily = get_price_history(symbol, fetch_days, token)
            inst = get_institutional(symbol, 30, token)
            margin = get_margin(symbol, 30, token)

            if daily.empty:
                st.error("目前沒有取得這檔股票資料。")
                st.stop()

            base = daily if timeframe == "日線" else to_weekly(daily)

            if len(base) < 70:
                st.error(f"{timeframe}資料不足，目前只有 {len(base)} 根 K 棒。")
                st.stop()

            d = calc_indicators(base)
            d, latest_state, current_stop = build_alternating_signals(d)

            chip = chip_summary(inst, margin)
            t_score, t_reasons = technical_score(d)

            st.success(f"{symbol}｜{timeframe}｜共 {len(d)} 根 K 棒")
            st.markdown(f"## 目前狀態：{latest_state}")

            signals = d[d["TradeSignal"] != ""].copy()
            buys = signals[signals["TradeSignal"] == "BUY"]
            sells = signals[signals["TradeSignal"] == "SELL"]

            last_buy = buys.iloc[-1] if not buys.empty else None
            last_sell = sells.iloc[-1] if not sells.empty else None

            a,b,c,dcol = st.columns(4)
            a.metric(
                "最近買進訊號",
                f"{last_buy.name.strftime('%Y-%m-%d')}" if last_buy is not None else "尚無"
            )
            b.metric(
                "最近買進價",
                f"{last_buy['TradePrice']:.2f}" if last_buy is not None else "-"
            )
            c.metric(
                "最近賣出訊號",
                f"{last_sell.name.strftime('%Y-%m-%d')}" if last_sell is not None else "尚無"
            )
            dcol.metric(
                "最近賣出價",
                f"{last_sell['TradePrice']:.2f}" if last_sell is not None else "-"
            )

            if pd.notna(current_stop):
                st.metric("🛑 目前停損價", f"{current_stop:.2f}")

            st.subheader(f"📈 {timeframe} K線＋MA5／MA10／MA20＋▲買進 / ▼賣出＋🔄多空反轉")
            st.caption("▲ 買進後進入持有狀態；持有期間不重複買。▼ 賣出後回到空手，等待下一次 ▲。")

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=d.index,
                open=d["Open"],
                high=d["High"],
                low=d["Low"],
                close=d["Close"],
                name="K線",
                increasing_line_color="#00c176",
                decreasing_line_color="#ff4d4f"
            ))

            fig.add_trace(go.Scatter(
                x=d.index, y=d["MA5"],
                mode="lines", name="MA5",
                line=dict(color="#ffd400", width=2)
            ))

            fig.add_trace(go.Scatter(
                x=d.index, y=d["MA10"],
                mode="lines", name="MA10",
                line=dict(color="#38bdf8", width=2)
            ))

            fig.add_trace(go.Scatter(
                x=d.index, y=d["MA20"],
                mode="lines", name="MA20",
                line=dict(color="#9b4dff", width=2)
            ))

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index,
                    y=buys["Low"] - buys["ATR14"].fillna(0) * 0.35,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#ff4d4f",
                        line=dict(color="white", width=1.5)
                    ),
                    name="買進訊號 ▲",
                    text=buys["TradeReason"],
                    customdata=buys["TradePrice"],
                    hovertemplate="%{x}<br>▲ 買進<br>價格：%{customdata:.2f}<br>%{text}<extra></extra>"
                ))

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index,
                    y=sells["High"] + sells["ATR14"].fillna(0) * 0.35,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=14,
                        color="#00c176",
                        line=dict(color="white", width=1.5)
                    ),
                    name="賣出訊號 ▼",
                    text=sells["TradeReason"],
                    customdata=sells["TradePrice"],
                    hovertemplate="%{x}<br>▼ 賣出<br>價格：%{customdata:.2f}<br>%{text}<extra></extra>"
                ))

            if pd.notna(current_stop):
                fig.add_hline(
                    y=current_stop,
                    line_dash="dot",
                    annotation_text="停損價"
                )


            # 多空雙向反轉提示
            bull_rev = d[d.get("BullReversal", False) == True] if "BullReversal" in d.columns else d.iloc[0:0]
            bear_rev = d[d.get("BearReversal", False) == True] if "BearReversal" in d.columns else d.iloc[0:0]

            if not bull_rev.empty:
                fig.add_trace(go.Scatter(
                    x=bull_rev.index,
                    y=bull_rev["Low"] * 0.985,
                    mode="markers",
                    name="🔄 空翻多",
                    marker=dict(symbol="diamond", size=11, color="#2563eb",
                                line=dict(color="white", width=1))
                ))

            if not bear_rev.empty:
                fig.add_trace(go.Scatter(
                    x=bear_rev.index,
                    y=bear_rev["High"] * 1.015,
                    mode="markers",
                    name="🔄 多翻空",
                    marker=dict(symbol="diamond", size=11, color="#f97316",
                                line=dict(color="white", width=1))
                ))

            fig.update_layout(
                height=720,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("最近買賣紀錄")
            if signals.empty:
                st.info("目前歷史資料尚未出現符合完整規則的買賣訊號。")
            else:
                recent = signals.tail(16)
                table = pd.DataFrame({
                    "日期": recent.index.strftime("%Y-%m-%d"),
                    "週期": timeframe,
                    "訊號": recent["TradeSignal"].map({"BUY":"▲ 買進", "SELL":"▼ 賣出"}),
                    "價格": recent["TradePrice"].round(2),
                    "原因": recent["TradeReason"],
                    "停損價": recent["TradeStop"].round(2),
                })
                st.dataframe(table, use_container_width=True, hide_index=True)

            st.subheader("訊號規則")
            st.write("▲ **買進**：MA5 上穿 MA20 為主，並由 MACD、KD 或突破放量至少一項確認。")
            st.write("▼ **賣出**：MA5 跌破 MA20，或跌破 MA20 且 MACD 轉空；持有期間若跌破停損價也會出場。")
            st.info("日線與週線是兩套獨立訊號。週線速度較慢、訊號較少；日線較靈敏、訊號較多。所有訊號均以該根 K 棒收盤資料確認。")

        except Exception as e:
            st.error("買賣訊號分析發生錯誤。")
            st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption(
    "本系統僅供技術與籌碼研究，不構成投資建議。拉回觀察區、突破參考、壓力、停損價與均為模型條件值，不是保證成交或獲利價位。"
)
