
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股盤前盤後決策雷達 V7", layout="wide")

st.title("🚦 台股盤前盤後決策雷達 V7")
st.caption("明日決策 Top5｜法人＋融資＋量價＋技術面｜買點條件＋失效價＋風險報酬｜FinMind")

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

    # 失效價：跌破支撐約 0.8 ATR
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
        "失效價": round(invalid, 2),
        "風報比": round(rr, 2),
        "突破條件": f"收盤/盤中有效突破 {breakout:.2f} 且量能≥5日均量1.2倍",
    }

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
                "失效價": plan["失效價"],
                "風報比": plan["風報比"],
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
                    "拉回觀察區","突破參考","第一壓力","第二壓力","失效價","風報比"
                ]],
                use_container_width=True,
                height=330
            )

            best = top5.iloc[0]
            st.success(
                f"目前第 1 名：{best['股票代號']} {best['名稱']}｜"
                f"實戰分 {best['實戰分']}｜{best['決策']}｜"
                f"拉回區 {best['拉回觀察區']}｜突破參考 {best['突破參考']}｜失效價 {best['失效價']}"
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
                        "拉回觀察區","突破參考","失效價","風報比"
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
    st.subheader("🔍 個股實戰分析")
    symbol = st.text_input("股票代號", value="2330", max_chars=8)
    run = st.button("開始分析", type="primary")

    if run:
        try:
            price = get_price_history(symbol, 800, token)
            inst = get_institutional(symbol, 30, token)
            margin = get_margin(symbol, 30, token)

            if len(price) < 61:
                st.error("股價資料不足。")
                st.stop()

            d = calc_indicators(price)
            t_score, t_reasons = technical_score(d)
            chip = chip_summary(inst, margin)
            lock = lock_score(chip, d)
            total = overall_score(t_score, chip["籌碼分"], lock)
            levels = price_levels(d)
            plan = trade_plan(d, levels, total, chip)
            x = d.iloc[-1]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{x['Close']:.2f}")
            c2.metric("實戰分", f"{total}/100")
            c3.metric("籌碼強度", f"{lock}/100")
            c4.metric("判定", verdict(total))

            st.subheader("🚦 明日決策卡")
            st.success(f"{plan['決策']}｜{plan['突破條件']}")
            a,b,c,dcol = st.columns(4)
            a.metric("拉回觀察區", plan["拉回觀察區"])
            b.metric("突破參考", f"{plan['突破參考']}")
            c.metric("第一 / 第二壓力", f"{plan['第一壓力']} / {plan['第二壓力']}")
            dcol.metric("失效價", f"{plan['失效價']}")
            st.metric("風險報酬比", f"{plan['風報比']}:1")

            st.subheader("法人籌碼")
            a,b,c,dcol = st.columns(4)
            a.metric("外資", f"{chip['外資']:,.0f} 張")
            b.metric("投信", f"{chip['投信']:,.0f} 張")
            c.metric("自營商", f"{chip['自營']:,.0f} 張")
            dcol.metric("法人連買", f"{chip['法人連買']} 天")

            st.subheader("融資融券")
            a,b = st.columns(2)
            a.metric("融資增減", f"{chip['融資增減']:,.0f} 張")
            b.metric("融券增減", f"{chip['融券增減']:,.0f} 張")

            st.subheader("K線＋均線")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=d.index, open=d["Open"], high=d["High"],
                low=d["Low"], close=d["Close"], name="K線"
            ))
            for n in [5,10,20,60]:
                fig.add_trace(go.Scatter(
                    x=d.index, y=d[f"MA{n}"], mode="lines", name=f"MA{n}"
                ))
            fig.add_hline(y=levels["支撐"], line_dash="dash", annotation_text="支撐")
            fig.add_hline(y=plan["第一壓力"], line_dash="dash", annotation_text="第一壓力")
            fig.add_hline(y=plan["第二壓力"], line_dash="dash", annotation_text="第二壓力")
            fig.add_hline(y=plan["失效價"], line_dash="dot", annotation_text="失效價")
            fig.update_layout(height=650, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("主要訊號")
            for s in chip["訊號"] + t_reasons:
                st.write(f"✅ {s}")

            st.info(
                "「籌碼強度」不是實際法人持股集中度，而是依法人淨買超、連買、融資變化與技術結構建立的代理分數。"
            )

        except Exception as e:
            st.error("分析發生錯誤。")
            st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption(
    "本系統僅供技術與籌碼研究，不構成投資建議。拉回觀察區、突破參考、壓力、失效價與風險報酬比均為模型條件值，不是保證成交或獲利價位。"
)
