
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股法人籌碼雷達 V5", layout="wide")

st.title("📊 台股法人籌碼雷達 V5")
st.caption("法人＋融資＋技術面綜合選股｜FinMind 台股資料｜研究測試用途")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# ---------------- API ----------------
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
def get_price_history(symbol, days=180, token=""):
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
def get_institutional(symbol, days=20, token=""):
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
    numeric_cols = [c for c in df.columns if c.endswith("_buy") or c.endswith("_sell")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in [
        "Foreign_Investor_buy","Foreign_Investor_sell",
        "Investment_Trust_buy","Investment_Trust_sell",
        "Dealer_buy","Dealer_sell",
        "Dealer_self_buy","Dealer_self_sell",
        "Dealer_Hedging_buy","Dealer_Hedging_sell",
    ]:
        if c not in df.columns:
            df[c] = 0

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
def get_margin(symbol, days=20, token=""):
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
    for c in [
        "MarginPurchaseTodayBalance","MarginPurchaseYesterdayBalance",
        "ShortSaleTodayBalance","ShortSaleYesterdayBalance",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date")

# ---------------- 技術面 ----------------
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

    d["VOL_MA5"] = d["Volume"].rolling(5).mean()
    d["HIGH20_PREV"] = d["High"].shift(1).rolling(20).max()
    return d

def technical_score(d):
    if len(d) < 61:
        return 0, []

    x = d.iloc[-1]
    prev = d.iloc[-2]
    score = 0
    reasons = []

    if x["Close"] > x["MA20"]:
        score += 10; reasons.append("站上MA20")
    if x["MA20"] > prev["MA20"]:
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

# ---------------- 籌碼面 ----------------
def consecutive_positive(series):
    n = 0
    for v in reversed(series.tolist()):
        if pd.notna(v) and v > 0:
            n += 1
        else:
            break
    return n

def chip_summary(inst, margin):
    result = {
        "外資買賣超": 0,
        "投信買賣超": 0,
        "自營商買賣超": 0,
        "法人合計": 0,
        "法人連買天數": 0,
        "融資增減": 0,
        "融券增減": 0,
        "籌碼分": 0,
        "籌碼訊號": [],
    }

    if not inst.empty:
        x = inst.iloc[-1]
        result["外資買賣超"] = float(x["ForeignNet"]) / 1000
        result["投信買賣超"] = float(x["TrustNet"]) / 1000
        result["自營商買賣超"] = float(x["DealerNet"]) / 1000
        result["法人合計"] = float(x["TotalInstNet"]) / 1000
        result["法人連買天數"] = consecutive_positive(inst["TotalInstNet"])

        if x["ForeignNet"] > 0:
            result["籌碼分"] += 10
            result["籌碼訊號"].append("外資買超")
        if x["TrustNet"] > 0:
            result["籌碼分"] += 10
            result["籌碼訊號"].append("投信買超")
        if x["DealerNet"] > 0:
            result["籌碼分"] += 5
            result["籌碼訊號"].append("自營商買超")
        if result["法人連買天數"] >= 3:
            result["籌碼分"] += 5
            result["籌碼訊號"].append("法人連買≥3日")

    if not margin.empty:
        x = margin.iloc[-1]
        if pd.notna(x.get("MarginPurchaseTodayBalance")) and pd.notna(x.get("MarginPurchaseYesterdayBalance")):
            result["融資增減"] = float(x["MarginPurchaseTodayBalance"] - x["MarginPurchaseYesterdayBalance"])
            if result["融資增減"] < 0:
                result["籌碼分"] += 5
                result["籌碼訊號"].append("融資減少")

        if pd.notna(x.get("ShortSaleTodayBalance")) and pd.notna(x.get("ShortSaleYesterdayBalance")):
            result["融券增減"] = float(x["ShortSaleTodayBalance"] - x["ShortSaleYesterdayBalance"])
            if result["融券增減"] > 0:
                result["籌碼分"] += 5
                result["籌碼訊號"].append("融券增加")

    result["籌碼分"] = min(result["籌碼分"], 40)
    return result

def overall_grade(score):
    if score >= 85: return "🔥 重點關注"
    if score >= 70: return "🟢 偏多"
    if score >= 55: return "🟡 觀察"
    if score >= 40: return "⚠️ 偏弱"
    return "🔴 避開"

# ---------------- 候選池 ----------------
WATCHLIST = [
    "2330","2317","2454","2382","3231","2308","2303","2881","2882","2886",
    "2891","2884","2885","2880","2883","2892","2887","1301","1303","2002",
    "1216","2207","2412","3711","2379","3034","6669","3008","2327","2357",
    "2345","2360","2356","2376","2383","2395","2408","2449","2603","2609"
]

# ---------------- 介面 ----------------
with st.sidebar:
    st.header("設定")
    token = st.text_input(
        "FinMind Token（選填）",
        type="password",
        help="不填也可使用；填入後可提高 API 使用上限。Token 不會寫入程式檔。"
    )
    page = st.radio("功能", ["🔥 綜合選股 Top20", "🔍 個股籌碼分析"], index=0)

if page == "🔥 綜合選股 Top20":
    st.subheader("🔥 法人＋籌碼＋技術面 Top20")
    st.caption("綜合分＝技術面 60%＋籌碼面 40%。候選池採大型／高流動性股票，以控制 API 使用量。")

    pool_size = st.selectbox("候選池大小", [10, 15, 20], index=1)
    run_scan = st.button("開始綜合掃描", type="primary")

    if run_scan:
        info = get_stock_info(token)
        info_map = dict(zip(info["stock_id"], info["stock_name"])) if not info.empty else {}

        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, symbol in enumerate(WATCHLIST[:pool_size], start=1):
            status.write(f"分析中：{symbol}（{idx}/{pool_size}）")
            try:
                price = get_price_history(symbol, 180, token)
                inst = get_institutional(symbol, 20, token)
                margin = get_margin(symbol, 20, token)

                if len(price) < 61:
                    progress.progress(idx / pool_size)
                    continue

                d = calc_indicators(price)
                t_score, t_reasons = technical_score(d)
                chip = chip_summary(inst, margin)

                tech_60 = round(t_score * 0.60, 1)
                chip_40 = chip["籌碼分"]
                overall = min(round(tech_60 + chip_40, 1), 100)

                latest = d.iloc[-1]
                prev = d.iloc[-2]
                pct = (latest["Close"] / prev["Close"] - 1) * 100
                vol_ratio = (
                    latest["Volume"] / latest["VOL_MA5"]
                    if pd.notna(latest["VOL_MA5"]) and latest["VOL_MA5"] > 0 else np.nan
                )

                results.append({
                    "股票代號": symbol,
                    "名稱": info_map.get(symbol, ""),
                    "收盤價": round(float(latest["Close"]), 2),
                    "漲跌幅%": round(float(pct), 2),
                    "綜合分": overall,
                    "判定": overall_grade(overall),
                    "技術分": t_score,
                    "籌碼分": chip["籌碼分"],
                    "外資(張)": round(chip["外資買賣超"], 0),
                    "投信(張)": round(chip["投信買賣超"], 0),
                    "自營(張)": round(chip["自營商買賣超"], 0),
                    "法人連買": chip["法人連買天數"],
                    "融資增減(張)": round(chip["融資增減"], 0),
                    "RSI": round(float(latest["RSI"]), 1),
                    "量比": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
                    "訊號": "、".join((chip["籌碼訊號"] + t_reasons)[:6]),
                })
            except Exception:
                pass

            progress.progress(idx / pool_size)

        progress.empty()
        status.empty()

        if not results:
            st.error("本次沒有成功完成分析。可能是 API 使用次數已達上限，或部分資料尚未更新。")
        else:
            out = pd.DataFrame(results).sort_values(
                ["綜合分", "籌碼分", "技術分"],
                ascending=[False, False, False]
            ).head(20).reset_index(drop=True)
            out.index = out.index + 1
            st.success(f"完成｜候選池 {pool_size} 檔")
            st.dataframe(out, use_container_width=True, height=760)

else:
    st.subheader("🔍 個股籌碼分析")
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
            overall = min(round(t_score * 0.60 + chip["籌碼分"], 1), 100)

            latest = d.iloc[-1]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{latest['Close']:.2f}")
            c2.metric("技術分", f"{t_score}/100")
            c3.metric("籌碼分", f"{chip['籌碼分']}/40")
            c4.metric("綜合判定", f"{overall_grade(overall)}｜{overall}")

            st.subheader("三大法人")
            a,b,c,dcol = st.columns(4)
            a.metric("外資買賣超", f"{chip['外資買賣超']:,.0f} 張")
            b.metric("投信買賣超", f"{chip['投信買賣超']:,.0f} 張")
            c.metric("自營商買賣超", f"{chip['自營商買賣超']:,.0f} 張")
            dcol.metric("法人連買", f"{chip['法人連買天數']} 天")

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
                fig.add_trace(go.Scatter(x=d.index, y=d[f"MA{n}"], mode="lines", name=f"MA{n}"))
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("主要訊號")
            for x in chip["籌碼訊號"] + t_reasons:
                st.write(f"✅ {x}")

        except Exception as e:
            st.error("分析發生錯誤。")
            st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption("本系統僅供技術與籌碼研究，不構成投資建議。法人資料與融資融券資料有各自更新時間，盤後資料請以 API 當下回傳為準。")
