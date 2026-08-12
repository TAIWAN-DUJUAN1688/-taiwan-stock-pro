
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股強勢分析 MVP V4.1", layout="wide")

st.title("📈 台股強勢分析 MVP V4.1")
st.caption("修正版：強勢股 Top20＋個股技術分析｜FinMind 台股資料｜測試用途")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

def api_get(params, timeout=30):
    r = requests.get(FINMIND_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind API 錯誤：{payload.get('status')} / {payload.get('msg')}")
    return payload

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

def score_stock(d):
    if len(d) < 61:
        return None, []

    x = d.iloc[-1]
    prev = d.iloc[-2]
    score = 0
    reasons = []

    if pd.notna(x["MA20"]) and x["Close"] > x["MA20"]:
        score += 10; reasons.append("站上MA20")
    if pd.notna(x["MA20"]) and pd.notna(prev["MA20"]) and x["MA20"] > prev["MA20"]:
        score += 10; reasons.append("MA20向上")
    if pd.notna(x["MA60"]) and x["Close"] > x["MA60"]:
        score += 10; reasons.append("站上MA60")

    if pd.notna(x["MA5"]) and pd.notna(x["MA10"]) and x["MA5"] > x["MA10"]:
        score += 10; reasons.append("MA5>MA10")
    if pd.notna(x["MA10"]) and pd.notna(x["MA20"]) and x["MA10"] > x["MA20"]:
        score += 10; reasons.append("MA10>MA20")

    if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0:
        if x["Volume"] > x["VOL_MA5"]:
            score += 10; reasons.append("量>5日均量")
        if x["Volume"] > x["VOL_MA5"] * 1.5:
            score += 10; reasons.append("量增1.5倍")

    if pd.notna(x["K"]) and pd.notna(x["D"]) and x["K"] > x["D"]:
        score += 7; reasons.append("KD偏多")
    if pd.notna(x["MACD"]) and pd.notna(x["Signal"]) and x["MACD"] > x["Signal"]:
        score += 7; reasons.append("MACD偏多")
    if pd.notna(x["RSI"]) and x["RSI"] > 50:
        score += 6; reasons.append("RSI>50")

    if pd.notna(x["HIGH20_PREV"]) and x["Close"] > x["HIGH20_PREV"]:
        score += 10; reasons.append("突破20日高")

    return min(int(score), 100), reasons

def grade(score):
    if score >= 85: return "🔥 A級"
    if score >= 70: return "🟢 B級"
    if score >= 55: return "🟡 中性"
    if score >= 40: return "⚠️ 偏弱"
    return "🔴 弱勢"

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info():
    payload = api_get({"dataset": "TaiwanStockInfo"})
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df
    df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_price_history(symbol: str, days=180):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    payload = api_get({
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    })
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

@st.cache_data(ttl=3600, show_spinner=False)
def build_candidate_pool(limit=40):
    """
    FinMind 免費層無法直接用 TaiwanStockPrice 不帶 data_id 取得全市場日資料。
    因此改用 TaiwanStockInfo 建立候選清單，再逐檔抓歷史資料。
    為避免 API 過量，只先取常見大型/高流動性股票清單。
    """
    watch = [
        "2330","2317","2454","2382","3231","2308","2303","2881","2882","2886",
        "2891","2884","2885","2880","2883","2892","2887","1301","1303","2002",
        "1216","2207","2412","3711","2379","3034","6669","3008","2327","2357",
        "2345","2360","2356","2376","2383","2395","2408","2449","2603","2609",
        "2615","2618","2610","2617","2634","5871","5880","2888","2801","2823",
        "2912","9910","6505","2105","1101","1102","1519","1504","1605","1476",
        "1590","2059","2404","4938","5269","5483","8046","8299","5347","5274",
    ]
    info = get_stock_info()
    info_map = {}
    if not info.empty:
        info_map = dict(zip(info["stock_id"], info["stock_name"]))

    rows = []
    for s in watch[:max(limit, 20)]:
        rows.append({"股票代號": s, "名稱": info_map.get(s, "")})
    return pd.DataFrame(rows)

page = st.sidebar.radio("功能", ["🔥 今日強勢股 Top20", "🔍 個股分析"], index=0)

if page == "🔥 今日強勢股 Top20":
    st.subheader("🔥 今日強勢股 Top20")
    st.caption(
        "V4.1 已修正 FinMind 400 錯誤。免費 API 無法直接抓全市場單日 TaiwanStockPrice，"
        "所以改成從大型/高流動性候選清單逐檔計算技術分數。"
    )

    pool_size = st.selectbox("候選池大小", [20, 30, 40], index=1)
    run_scan = st.button("開始掃描強勢股", type="primary")

    if run_scan:
        pool = build_candidate_pool(pool_size)
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, row in pool.head(pool_size).iterrows():
            symbol = row["股票代號"]
            status.write(f"分析中：{symbol}（{i+1}/{pool_size}）")
            try:
                hist = get_price_history(symbol, days=180)
                if len(hist) < 61:
                    progress.progress((i+1)/pool_size)
                    continue
                d = calc_indicators(hist)
                score, reasons = score_stock(d)
                if score is None:
                    progress.progress((i+1)/pool_size)
                    continue

                latest = d.iloc[-1]
                prev = d.iloc[-2]
                pct = (latest["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else np.nan
                vol_ratio = (
                    float(latest["Volume"] / latest["VOL_MA5"])
                    if pd.notna(latest["VOL_MA5"]) and latest["VOL_MA5"] > 0 else np.nan
                )

                results.append({
                    "股票代號": symbol,
                    "名稱": row["名稱"],
                    "收盤價": round(float(latest["Close"]), 2),
                    "漲跌幅%": round(float(pct), 2),
                    "評分": score,
                    "等級": grade(score),
                    "RSI": round(float(latest["RSI"]), 1) if pd.notna(latest["RSI"]) else None,
                    "量比": round(vol_ratio, 2) if pd.notna(vol_ratio) else None,
                    "主要訊號": "、".join(reasons[:4]),
                })
            except Exception:
                pass

            progress.progress((i+1)/pool_size)

        progress.empty()
        status.empty()

        if not results:
            st.error("本次沒有成功完成分析，可能遇到 API 次數限制或暫時連線異常。")
        else:
            out = pd.DataFrame(results).sort_values(
                ["評分", "漲跌幅%", "量比"],
                ascending=[False, False, False]
            ).head(20).reset_index(drop=True)
            out.index = out.index + 1
            st.success(f"完成｜候選池 {pool_size} 檔｜顯示前 20 名")
            st.dataframe(out, use_container_width=True, height=760)

else:
    st.subheader("🔍 個股分析")
    symbol = st.sidebar.text_input("股票代號", value="2330", max_chars=8)
    period = st.sidebar.selectbox("資料期間", ["6mo", "1y", "2y", "5y"], index=2)
    refresh = st.sidebar.checkbox("重新下載最新資料")
    run = st.sidebar.button("開始分析", type="primary", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    days_map = {"6mo":220, "1y":430, "2y":800, "5y":1900}

    if not run:
        st.info("左側輸入股票代號，例如 2330、2317、2454，然後按『開始分析』。")
        st.stop()

    try:
        df = get_price_history(symbol, days=days_map[period])
        if df.empty or len(df) < 70:
            st.error("資料不足或目前無法取得這檔股票資料。")
            st.stop()

        d = calc_indicators(df)
        score, reasons = score_stock(d)
        latest = d.iloc[-1]
        prev = d.iloc[-2]
        pct = (latest["Close"] / prev["Close"] - 1) * 100

        st.success(f"資料讀取成功｜{symbol}｜共 {len(df)} 筆日線")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("股票代號", symbol)
        c2.metric("收盤價", f"{latest['Close']:.2f}", f"{pct:+.2f}%")
        c3.metric("綜合評分", f"{score}/100")
        c4.metric("系統判定", grade(score))

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"],
            low=d["Low"], close=d["Close"], name="K線"
        ))
        for n in [5, 10, 20, 60]:
            fig.add_trace(go.Scatter(x=d.index, y=d[f"MA{n}"], mode="lines", name=f"MA{n}"))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("個股分析發生錯誤。")
        st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption("本系統僅供技術分析與研究測試，不構成投資建議。正式商用前應確認資料授權與 FinMind 服務條款。")
