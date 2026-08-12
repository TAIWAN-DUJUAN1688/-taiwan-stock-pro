
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股強勢分析 MVP V3", layout="wide")

st.title("📈 台股強勢分析 MVP V3")
st.caption("第三版：改用 FinMind 台股資料｜技術分析＋100分評分｜測試用途")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

def period_to_days(period):
    return {
        "6mo": 220,
        "1y": 430,
        "2y": 800,
        "5y": 1900,
    }[period]

@st.cache_data(ttl=1800, show_spinner=False)
def load_price_finmind(symbol: str, period: str):
    symbol = symbol.strip()
    end_date = date.today()
    start_date = end_date - timedelta(days=period_to_days(period))

    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    status = payload.get("status")
    msg = payload.get("msg")
    rows = payload.get("data", [])

    if status not in (200, None):
        raise RuntimeError(f"FinMind API 錯誤：{status} / {msg}")

    if not rows:
        return pd.DataFrame(), msg

    df = pd.DataFrame(rows)

    rename_map = {
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"FinMind 回傳欄位缺少：{missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
          .sort_values("Date")
          .drop_duplicates("Date", keep="last")
          .set_index("Date")
    )

    df["Volume"] = df["Volume"].fillna(0)
    return df, msg

def calc_indicators(df: pd.DataFrame):
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
    d["Hist"] = d["MACD"] - d["Signal"]

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

def score_stock(d: pd.DataFrame):
    x = d.iloc[-1]
    prev = d.iloc[-2]
    score = 0
    reasons = []

    if pd.notna(x["MA20"]) and x["Close"] > x["MA20"]:
        score += 10
        reasons.append("股價站上 MA20")
    if pd.notna(x["MA20"]) and pd.notna(prev["MA20"]) and x["MA20"] > prev["MA20"]:
        score += 10
        reasons.append("MA20 向上")
    if pd.notna(x["MA60"]) and x["Close"] > x["MA60"]:
        score += 10
        reasons.append("股價站上 MA60")

    if pd.notna(x["MA5"]) and pd.notna(x["MA10"]) and x["MA5"] > x["MA10"]:
        score += 10
        reasons.append("MA5 > MA10")
    if pd.notna(x["MA10"]) and pd.notna(x["MA20"]) and x["MA10"] > x["MA20"]:
        score += 10
        reasons.append("MA10 > MA20")

    if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0:
        if x["Volume"] > x["VOL_MA5"]:
            score += 10
            reasons.append("成交量高於 5 日均量")
        if x["Volume"] > x["VOL_MA5"] * 1.5:
            score += 10
            reasons.append("成交量放大 1.5 倍")

    if pd.notna(x["K"]) and pd.notna(x["D"]) and x["K"] > x["D"]:
        score += 7
        reasons.append("KD 偏多")
    if pd.notna(x["MACD"]) and pd.notna(x["Signal"]) and x["MACD"] > x["Signal"]:
        score += 7
        reasons.append("MACD 偏多")
    if pd.notna(x["RSI"]) and x["RSI"] > 50:
        score += 6
        reasons.append("RSI > 50")

    if pd.notna(x["HIGH20_PREV"]) and x["Close"] > x["HIGH20_PREV"]:
        score += 10
        reasons.append("突破前 20 日高點")

    return min(int(score), 100), reasons

def grade(score):
    if score >= 85:
        return "🔥 A級強勢"
    if score >= 70:
        return "🟢 B級觀察"
    if score >= 55:
        return "🟡 中性"
    if score >= 40:
        return "⚠️ 偏弱"
    return "🔴 弱勢"

with st.sidebar:
    st.header("設定")
    symbol = st.text_input("股票代號", value="2330", max_chars=8)
    period = st.selectbox("資料期間", ["6mo", "1y", "2y", "5y"], index=2)
    refresh = st.checkbox("重新下載最新資料")
    run = st.button("開始分析", type="primary", use_container_width=True)

if refresh:
    st.cache_data.clear()

if not run:
    st.info("左側輸入股票代號，例如 2330、2317、2454，然後按『開始分析』。")
    st.stop()

try:
    with st.spinner("正在讀取台股資料並計算技術指標..."):
        df, api_msg = load_price_finmind(symbol, period)

    if df.empty:
        st.error("FinMind 目前沒有回傳這檔股票的日線資料。")
        st.warning("請先確認股票代號是否正確；若正確，稍後再試。")
        with st.expander("查看技術診斷"):
            st.write("股票代號：", symbol)
            st.write("API 訊息：", api_msg)
        st.stop()

    if len(df) < 70:
        st.error(f"目前只有 {len(df)} 筆日線資料，至少需要約 70 筆才能完整計算 MA60 與評分。")
        st.info("請把資料期間改成 1y、2y 或 5y。")
        st.stop()

    d = calc_indicators(df)
    score, reasons = score_stock(d)

    latest = d.iloc[-1]
    prev = d.iloc[-2]
    pct = (latest["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else 0

    st.success(f"資料讀取成功｜{symbol}｜共 {len(df)} 筆日線")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("股票代號", symbol)
    c2.metric("收盤價", f"{latest['Close']:.2f}", f"{pct:+.2f}%")
    c3.metric("綜合評分", f"{score}/100")
    c4.metric("系統判定", grade(score))

    st.subheader("K線＋均線")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="K線"
    ))
    for n in [5, 10, 20, 60]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d[f"MA{n}"],
            mode="lines", name=f"MA{n}"
        ))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("KD")
        kd = go.Figure()
        kd.add_trace(go.Scatter(x=d.index, y=d["K"], mode="lines", name="K"))
        kd.add_trace(go.Scatter(x=d.index, y=d["D"], mode="lines", name="D"))
        kd.update_layout(height=300)
        st.plotly_chart(kd, use_container_width=True)

        st.subheader("RSI")
        rsi = go.Figure()
        rsi.add_trace(go.Scatter(x=d.index, y=d["RSI"], mode="lines", name="RSI14"))
        rsi.update_layout(height=300)
        st.plotly_chart(rsi, use_container_width=True)

    with right:
        st.subheader("MACD")
        macd = go.Figure()
        macd.add_trace(go.Scatter(x=d.index, y=d["MACD"], mode="lines", name="MACD"))
        macd.add_trace(go.Scatter(x=d.index, y=d["Signal"], mode="lines", name="Signal"))
        macd.add_trace(go.Bar(x=d.index, y=d["Hist"], name="Hist"))
        macd.update_layout(height=300)
        st.plotly_chart(macd, use_container_width=True)

        st.subheader("成交量")
        vol = go.Figure()
        vol.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume"))
        vol.add_trace(go.Scatter(x=d.index, y=d["VOL_MA5"], mode="lines", name="5日均量"))
        vol.update_layout(height=300)
        st.plotly_chart(vol, use_container_width=True)

    st.subheader("評分依據")
    if reasons:
        for item in reasons:
            st.write(f"✅ {item}")
    else:
        st.write("目前沒有符合主要偏多條件。")

    with st.expander("資料資訊"):
        st.write("資料來源：FinMind TaiwanStockPrice")
        st.write("資料筆數：", len(df))
        st.write("資料起日：", df.index.min())
        st.write("資料迄日：", df.index.max())
        st.write("API 訊息：", api_msg)

    st.info("此版本僅做技術分析與程式測試，不構成投資建議。若未來商用，仍應確認資料授權與服務條款。")

except requests.HTTPError as e:
    st.error("FinMind 連線失敗。")
    st.code(str(e))
except requests.RequestException as e:
    st.error("目前無法連線到 FinMind。")
    st.code(str(e))
except Exception as e:
    st.error("程式執行發生錯誤。")
    st.code(f"{type(e).__name__}: {e}")
