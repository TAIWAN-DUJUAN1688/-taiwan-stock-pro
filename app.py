
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="台股強勢分析 MVP", layout="wide")

st.title("📈 台股強勢分析 MVP")
st.caption("第一版：個股技術分析＋100分評分｜資料來源：Yahoo Finance（測試用途）")

@st.cache_data(ttl=900)
def load_price(symbol: str, period="1y"):
    ticker = f"{symbol}.TW"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        ticker = f"{symbol}.TWO"
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def calc_indicators(df: pd.DataFrame):
    d = df.copy()
    for n in [5, 10, 20, 60]:
        d[f"MA{n}"] = d["Close"].rolling(n).mean()

    low9 = d["Low"].rolling(9).min()
    high9 = d["High"].rolling(9).max()
    rsv = (d["Close"] - low9) / (high9 - low9) * 100
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

    # 趨勢 30
    if x["Close"] > x["MA20"]:
        score += 10; reasons.append("股價站上MA20")
    if x["MA20"] > prev["MA20"]:
        score += 10; reasons.append("MA20向上")
    if x["Close"] > x["MA60"]:
        score += 10; reasons.append("股價站上MA60")

    # 均線 20
    if x["MA5"] > x["MA10"]:
        score += 10; reasons.append("MA5 > MA10")
    if x["MA10"] > x["MA20"]:
        score += 10; reasons.append("MA10 > MA20")

    # 成交量 20
    if pd.notna(x["VOL_MA5"]) and x["Volume"] > x["VOL_MA5"]:
        score += 10; reasons.append("成交量高於5日均量")
    if pd.notna(x["VOL_MA5"]) and x["Volume"] > x["VOL_MA5"] * 1.5:
        score += 10; reasons.append("成交量放大1.5倍")

    # 動能 20
    if x["K"] > x["D"]:
        score += 7; reasons.append("KD偏多")
    if x["MACD"] > x["Signal"]:
        score += 7; reasons.append("MACD偏多")
    if x["RSI"] > 50:
        score += 6; reasons.append("RSI > 50")

    # 突破 10
    if pd.notna(x["HIGH20_PREV"]) and x["Close"] > x["HIGH20_PREV"]:
        score += 10; reasons.append("突破前20日高點")

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
    symbol = st.text_input("股票代號", value="2330", max_chars=6)
    period = st.selectbox("資料期間", ["6mo", "1y", "2y", "5y"], index=1)
    run = st.button("開始分析", type="primary", use_container_width=True)

if run or symbol:
    try:
        df = load_price(symbol.strip(), period)
        if len(df) < 70:
            st.error("資料不足，請確認股票代號或改用較長資料期間。")
            st.stop()

        d = calc_indicators(df)
        score, reasons = score_stock(d)
        latest = d.iloc[-1]
        prev = d.iloc[-2]
        pct = (latest["Close"] / prev["Close"] - 1) * 100

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
            fig.add_trace(go.Scatter(x=d.index, y=d[f"MA{n}"], mode="lines", name=f"MA{n}"))
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
        st.write("、".join(reasons) if reasons else "目前沒有符合主要偏多條件。")

        st.info("此版本僅做技術分析與程式測試，不構成投資建議。Yahoo Finance / yfinance 資料不應直接視為正式商用行情來源。")

    except Exception as e:
        st.exception(e)
