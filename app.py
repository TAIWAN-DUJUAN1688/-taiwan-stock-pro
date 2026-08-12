
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="台股強勢分析 MVP V4", layout="wide")

st.title("📈 台股強勢分析 MVP V4")
st.caption("第四版：強勢股 Top20＋個股技術分析｜FinMind 台股資料｜測試用途")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# ---------- 共用 ----------
def api_get(params, timeout=30):
    r = requests.get(FINMIND_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind API 錯誤：{payload.get('status')} / {payload.get('msg')}")
    return payload

def period_to_days(period):
    return {"6mo": 220, "1y": 430, "2y": 800, "5y": 1900}[period]

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info():
    payload = api_get({"dataset": "TaiwanStockInfo"})
    rows = payload.get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # FinMind 欄位可能包含 stock_id / stock_name / type / industry_category
    keep = [c for c in ["stock_id", "stock_name", "type", "industry_category"] if c in df.columns]
    df = df[keep].copy()
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_price_history(symbol: str, days=180):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    payload = api_get({
        "dataset": "TaiwanStockPrice",
        "data_id": symbol.strip(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    })
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
        "Trading_money": "TradingMoney",
    })

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
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

@st.cache_data(ttl=900, show_spinner=False)
def get_latest_market_day(max_lookback=10):
    # FinMind 可依特定日期一次取得全市場股價。
    today = date.today()
    for i in range(max_lookback):
        d = today - timedelta(days=i)
        payload = api_get({
            "dataset": "TaiwanStockPrice",
            "start_date": d.isoformat(),
            "end_date": d.isoformat(),
        })
        rows = payload.get("data", [])
        if rows:
            df = pd.DataFrame(rows)
            if "date" in df.columns:
                return d.isoformat(), df
    return None, pd.DataFrame()

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

    # 趨勢 30
    if pd.notna(x["MA20"]) and x["Close"] > x["MA20"]:
        score += 10; reasons.append("站上MA20")
    if pd.notna(x["MA20"]) and pd.notna(prev["MA20"]) and x["MA20"] > prev["MA20"]:
        score += 10; reasons.append("MA20向上")
    if pd.notna(x["MA60"]) and x["Close"] > x["MA60"]:
        score += 10; reasons.append("站上MA60")

    # 均線 20
    if pd.notna(x["MA5"]) and pd.notna(x["MA10"]) and x["MA5"] > x["MA10"]:
        score += 10; reasons.append("MA5>MA10")
    if pd.notna(x["MA10"]) and pd.notna(x["MA20"]) and x["MA10"] > x["MA20"]:
        score += 10; reasons.append("MA10>MA20")

    # 成交量 20
    if pd.notna(x["VOL_MA5"]) and x["VOL_MA5"] > 0:
        if x["Volume"] > x["VOL_MA5"]:
            score += 10; reasons.append("量>5日均量")
        if x["Volume"] > x["VOL_MA5"] * 1.5:
            score += 10; reasons.append("量增1.5倍")

    # 動能 20
    if pd.notna(x["K"]) and pd.notna(x["D"]) and x["K"] > x["D"]:
        score += 7; reasons.append("KD偏多")
    if pd.notna(x["MACD"]) and pd.notna(x["Signal"]) and x["MACD"] > x["Signal"]:
        score += 7; reasons.append("MACD偏多")
    if pd.notna(x["RSI"]) and x["RSI"] > 50:
        score += 6; reasons.append("RSI>50")

    # 突破 10
    if pd.notna(x["HIGH20_PREV"]) and x["Close"] > x["HIGH20_PREV"]:
        score += 10; reasons.append("突破20日高")

    return min(int(score), 100), reasons

def grade(score):
    if score >= 85: return "🔥 A級"
    if score >= 70: return "🟢 B級"
    if score >= 55: return "🟡 中性"
    if score >= 40: return "⚠️ 偏弱"
    return "🔴 弱勢"

def pct_change(d):
    if len(d) < 2:
        return np.nan
    return (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100

# ---------- 頁面 ----------
page = st.sidebar.radio(
    "功能",
    ["🔥 今日強勢股 Top20", "🔍 個股分析"],
    index=0
)

# ---------- 強勢股 Top20 ----------
if page == "🔥 今日強勢股 Top20":
    st.subheader("🔥 今日強勢股 Top20")

    st.caption(
        "V4 MVP 先從『當日成交金額較高的高流動性股票』建立候選池，再計算技術評分。"
        "這樣可大幅降低 API 次數與等待時間；不是宣稱掃描所有股票後的絕對排名。"
    )

    pool_size = st.selectbox(
        "候選池大小",
        [30, 40, 50, 60],
        index=1,
        help="越大越接近全市場，但需要更多 API 請求與等待時間。"
    )

    run_scan = st.button("開始掃描強勢股", type="primary")

    if run_scan:
        try:
            with st.spinner("正在取得最新市場資料..."):
                market_date, market = get_latest_market_day()

            if market.empty:
                st.error("目前無法取得最近交易日的全市場股價。")
                st.stop()

            market = market.rename(columns={
                "stock_id": "股票代號",
                "close": "收盤價",
                "Trading_Volume": "成交量",
                "Trading_money": "成交金額",
            })

            for c in ["收盤價", "成交量", "成交金額"]:
                if c in market.columns:
                    market[c] = pd.to_numeric(market[c], errors="coerce")

            # 僅保留一般 4 碼股票代號，先排除 ETF、權證等大多數非普通股
            market["股票代號"] = market["股票代號"].astype(str)
            market = market[market["股票代號"].str.fullmatch(r"\d{4}", na=False)]

            # 取得股票名稱
            info = get_stock_info()
            if not info.empty and "stock_id" in info.columns:
                info_map = dict(zip(info["stock_id"], info.get("stock_name", info["stock_id"])))
            else:
                info_map = {}

            if "成交金額" in market.columns:
                candidates = market.sort_values("成交金額", ascending=False).head(pool_size)
            else:
                candidates = market.sort_values("成交量", ascending=False).head(pool_size)

            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, row in enumerate(candidates.itertuples(index=False), start=1):
                symbol = str(getattr(row, "股票代號"))
                status.write(f"分析中：{symbol}（{i}/{len(candidates)}）")

                try:
                    hist = get_price_history(symbol, days=180)
                    if len(hist) < 61:
                        continue
                    d = calc_indicators(hist)
                    score, reasons = score_stock(d)
                    if score is None:
                        continue

                    latest = d.iloc[-1]
                    results.append({
                        "股票代號": symbol,
                        "名稱": info_map.get(symbol, ""),
                        "收盤價": round(float(latest["Close"]), 2),
                        "漲跌幅%": round(float(pct_change(d)), 2),
                        "評分": score,
                        "等級": grade(score),
                        "RSI": round(float(latest["RSI"]), 1) if pd.notna(latest["RSI"]) else None,
                        "量比": round(float(latest["Volume"] / latest["VOL_MA5"]), 2)
                                if pd.notna(latest["VOL_MA5"]) and latest["VOL_MA5"] > 0 else None,
                        "主要訊號": "、".join(reasons[:4]),
                    })
                except Exception:
                    pass

                progress.progress(i / len(candidates))

            progress.empty()
            status.empty()

            if not results:
                st.error("本次沒有成功完成候選股分析，可能遇到 API 限制或暫時連線異常。")
                st.stop()

            out = pd.DataFrame(results)
            out = out.sort_values(
                ["評分", "漲跌幅%", "量比"],
                ascending=[False, False, False]
            ).head(20).reset_index(drop=True)
            out.index = out.index + 1

            st.success(f"完成｜資料基準日：{market_date}｜候選池：{pool_size} 檔")
            st.dataframe(out, use_container_width=True, height=760)

            top = out.iloc[0]
            st.info(
                f"目前候選池第 1 名：{top['股票代號']} {top['名稱']}｜"
                f"{top['評分']} 分｜{top['等級']}。"
            )

        except requests.HTTPError as e:
            st.error("FinMind API 連線失敗或使用次數受限。")
            st.code(str(e))
        except Exception as e:
            st.error("掃描發生錯誤。")
            st.code(f"{type(e).__name__}: {e}")

# ---------- 個股分析 ----------
else:
    st.subheader("🔍 個股分析")

    symbol = st.sidebar.text_input("股票代號", value="2330", max_chars=8)
    period = st.sidebar.selectbox("資料期間", ["6mo", "1y", "2y", "5y"], index=2)
    refresh = st.sidebar.checkbox("重新下載最新資料")
    run = st.sidebar.button("開始分析", type="primary", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    if not run:
        st.info("左側輸入股票代號，例如 2330、2317、2454，然後按『開始分析』。")
        st.stop()

    try:
        with st.spinner("正在讀取台股資料並計算技術指標..."):
            df = get_price_history(symbol, days=period_to_days(period))

        if df.empty:
            st.error("目前沒有取得這檔股票的日線資料。")
            st.stop()

        if len(df) < 70:
            st.error(f"目前只有 {len(df)} 筆日線資料，至少需要約 70 筆。")
            st.stop()

        d = calc_indicators(df)
        score, reasons = score_stock(d)
        latest = d.iloc[-1]
        pct = pct_change(d)

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
        for item in reasons:
            st.write(f"✅ {item}")

    except Exception as e:
        st.error("個股分析發生錯誤。")
        st.code(f"{type(e).__name__}: {e}")

st.divider()
st.caption("本系統僅供技術分析與研究測試，不構成投資建議。正式商用前應確認資料授權與 FinMind 服務條款。")
