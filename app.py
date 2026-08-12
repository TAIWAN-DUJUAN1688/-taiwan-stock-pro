
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime

st.set_page_config(
    page_title="EG Trader Pro Style V13",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# -------------------- 深色介面 --------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f14; --panel:#121820; --panel2:#171e27; --line:#29323d;
  --text:#f5f7fa; --muted:#aeb7c2; --red:#ff4d4f; --green:#00c176;
  --yellow:#ffd400; --purple:#9b4dff; --blue:#21b8f2; --gold:#d7a928;
}
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stSidebar"] { background:#10161d; border-right:1px solid #27303a; }
[data-testid="stSidebar"] * { color:#f1f4f7; }
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1600px; }
h1,h2,h3 { color:#f7f7f8 !important; letter-spacing:.01em; }
div[data-testid="stMetric"] {
  background:var(--panel2); border:1px solid #303946; border-radius:14px;
  padding:14px 16px; min-height:116px;
}
div[data-testid="stMetricLabel"] { color:#b8c0ca; }
div[data-testid="stMetricValue"] { color:#f6f7f9; }
div[data-testid="stDataFrame"] { border:1px solid #28313b; border-radius:12px; overflow:hidden; }
.notice {
  border:1px solid #98751b; border-radius:13px; padding:18px 20px;
  background:#11161d; color:#d9dde3; margin:12px 0 26px 0;
}
.stock-card {
  background:#151b22; border:1px solid #2d3742; border-radius:16px;
  padding:18px 22px; margin-bottom:18px;
}
.badge {
  display:inline-block; padding:5px 11px; margin-right:8px; border-radius:16px;
  background:#202833; border:1px solid #34404e; font-size:.92rem;
}
.status-bull {
  display:inline-block; padding:11px 35px; border-radius:24px; min-width:260px;
  text-align:center; color:#ff8a8a; background:#3a2023; border:1px solid #623237;
  font-weight:700;
}
.status-bear {
  display:inline-block; padding:11px 35px; border-radius:24px; min-width:260px;
  text-align:center; color:#7ee2ae; background:#173024; border:1px solid #27503b;
  font-weight:700;
}
.small-muted { color:#aab2bd; font-size:.92rem; }
hr { border-color:#27313c; }
</style>
""", unsafe_allow_html=True)

# -------------------- API --------------------
def api_get(params, timeout=30):
    r = requests.get(FINMIND_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind API 錯誤：{payload.get('status')} / {payload.get('msg')}")
    return payload

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info():
    p = api_get({"dataset":"TaiwanStockInfo"})
    df = pd.DataFrame(p.get("data", []))
    if not df.empty:
        df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_prices(symbol, start_date, end_date):
    p = api_get({
        "dataset":"TaiwanStockPrice",
        "data_id":str(symbol),
        "start_date":str(start_date),
        "end_date":str(end_date),
    })
    df = pd.DataFrame(p.get("data", []))
    if df.empty:
        return df
    df = df.rename(columns={
        "date":"Date","open":"Open","max":"High","min":"Low","close":"Close",
        "Trading_Volume":"Volume","Trading_money":"TradingMoney"
    })
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["Date","Open","High","Low","Close"])
              .sort_values("Date").drop_duplicates("Date").set_index("Date"))

def to_weekly(df):
    return pd.DataFrame({
        "Open":df["Open"].resample("W-FRI").first(),
        "High":df["High"].resample("W-FRI").max(),
        "Low":df["Low"].resample("W-FRI").min(),
        "Close":df["Close"].resample("W-FRI").last(),
        "Volume":df["Volume"].resample("W-FRI").sum(),
    }).dropna()

def indicators(df):
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()

    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["CCI20"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))

    prev = d["Close"].shift(1)
    tr = pd.concat([
        d["High"]-d["Low"], (d["High"]-prev).abs(), (d["Low"]-prev).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()

    d["BullCross"] = (d["MA5"] > d["MA20"]) & (d["MA5"].shift(1) <= d["MA20"].shift(1))
    d["BearCross"] = (d["MA5"] < d["MA20"]) & (d["MA5"].shift(1) >= d["MA20"].shift(1))
    return d

def strategy_signals(d):
    z = d.copy()
    z["Signal"] = ""
    z.loc[z["BullCross"], "Signal"] = "模擬轉多訊號"
    z.loc[z["BearCross"], "Signal"] = "模擬轉空訊號"
    return z

def backtest_bidirectional(d, initial_capital):
    """
    雙向理論回測：
    訊號在收盤確認，下一根K線開盤執行。
    多翻空或空翻多時，先平倉再反向建立新部位。
    """
    rows = []
    equity = float(initial_capital)
    position = 0  # 1 long, -1 short
    entry_price = None
    entry_signal_date = None
    entry_exec_date = None
    entry_equity = None

    signal_rows = d[d["Signal"] != ""]
    for sig_date, row in signal_rows.iterrows():
        loc = d.index.get_loc(sig_date)
        if isinstance(loc, slice) or loc >= len(d)-1:
            continue
        exec_date = d.index[loc+1]
        exec_price = float(d.iloc[loc+1]["Open"])
        direction = 1 if row["Signal"] == "模擬轉多訊號" else -1

        if position == direction:
            continue

        # 先平原部位
        if position != 0:
            if position == 1:
                ret = exec_price / entry_price - 1
                side = "多單"
            else:
                ret = entry_price / exec_price - 1
                side = "空單"

            pnl = entry_equity * ret
            equity = entry_equity + pnl
            rows.append({
                "方向":side,
                "進場訊號日期":entry_signal_date,
                "進場日期":entry_exec_date,
                "進場K線日期":entry_exec_date,
                "進場價格":entry_price,
                "進場價格基準":"下一根K線開盤價",
                "出場訊號日期":sig_date,
                "出場日期":exec_date,
                "出場K線日期":exec_date,
                "出場價格":exec_price,
                "出場價格基準":"下一根K線開盤價",
                "報酬金額":pnl,
                "報酬率%":ret*100,
            })

        position = direction
        entry_price = exec_price
        entry_signal_date = sig_date
        entry_exec_date = exec_date
        entry_equity = equity

    # 未平倉：只列浮動損益，不算完成交易
    unrealized = 0.0
    if position != 0 and entry_price is not None:
        last = float(d["Close"].iloc[-1])
        if position == 1:
            unrealized = entry_equity * (last / entry_price - 1)
        else:
            unrealized = entry_equity * (entry_price / last - 1)

    trades = pd.DataFrame(rows)
    return trades, equity, unrealized, position

def metrics_from_trades(trades, initial_capital, realized_equity, unrealized, d):
    if trades.empty:
        completed = 0; winrate = 0; realized = 0; maxdd = 0
        strategy_ret = (realized_equity + unrealized) / initial_capital - 1
    else:
        completed = len(trades)
        winrate = (trades["報酬金額"] > 0).mean()
        realized = trades["報酬金額"].sum()
        curve = initial_capital + trades["報酬金額"].cumsum()
        peak = curve.cummax()
        dd = curve / peak - 1
        maxdd = abs(dd.min()) if len(dd) else 0
        strategy_ret = (realized_equity + unrealized) / initial_capital - 1

    buy_hold = d["Close"].iloc[-1] / d["Close"].iloc[0] - 1
    return {
        "期末資產": realized_equity + unrealized,
        "策略總報酬率": strategy_ret,
        "買進持有總報酬率": buy_hold,
        "最大回撤": maxdd,
        "已實現損益": realized,
        "未實現損益": unrealized,
        "勝率": winrate,
        "完成交易": completed,
    }

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## ▾ 標的與回測設定")
    st.caption("設定標的、週期、日期與回測資金。")
    symbol = st.text_input("股票代號", value="2317")
    timeframe = st.selectbox("K線週期", ["日線","週線"], index=0)

    today = date.today()
    start_default = today - timedelta(days=220)
    start = st.date_input("開始日期", value=start_default)
    end = st.date_input("結束日期", value=today - timedelta(days=1))

    st.divider()
    capital = st.number_input("初始資金", min_value=100000, max_value=100000000,
                              value=1000000, step=100000)
    refresh = st.checkbox("重新下載最新資料")
    run = st.button("開始分析與回測", type="primary", use_container_width=True)

if refresh:
    st.cache_data.clear()

# -------------------- Header --------------------
st.markdown("## 📈 **EG Trader Pro**")
st.caption("V0.8.8 Family Web Beta V0.2 （R62）｜技術分析與策略回測")
st.markdown("""
<div class="notice">
<b>🔒 親友封閉測試版・雙層存取保護</b><br><br>
僅供受邀親友測試使用，請勿轉傳、複製、重製或作商業用途。本系統僅供技術分析與歷史回測測試，不構成任何投資建議。<br>
網頁測試版的自訂策略與股票清單僅保留於本次瀏覽工作階段；重要策略請自行備份。
</div>
""", unsafe_allow_html=True)

if not run:
    st.info("左側設定股票、K線週期、日期與資金後，按「開始分析與回測」。")
    st.stop()

try:
    with st.spinner("正在讀取資料並計算策略..."):
        info = get_stock_info()
        name_map = dict(zip(info["stock_id"], info["stock_name"])) if not info.empty else {}
        name = name_map.get(str(symbol), "")
        raw = get_prices(symbol, start, end)

    if raw.empty:
        st.error("目前沒有取得這檔股票的價格資料。")
        st.stop()

    base = raw if timeframe == "日線" else to_weekly(raw)
    if len(base) < 25:
        st.error("資料筆數不足，請擴大日期區間。")
        st.stop()

    d = strategy_signals(indicators(base))
    trades, realized_equity, unrealized, position = backtest_bidirectional(d, capital)
    m = metrics_from_trades(trades, capital, realized_equity, unrealized, d)

    latest = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else latest
    chg = float(latest["Close"] - prev["Close"])
    pct = chg / float(prev["Close"]) * 100 if prev["Close"] else 0

    # -------------------- 股票卡 --------------------
    bull = latest["MA5"] > latest["MA20"]
    status_html = f'<span class="{"status-bull" if bull else "status-bear"}">{"MA5 > MA20" if bull else "MA5 < MA20"}</span>'
    st.markdown(f"""
    <div class="stock-card">
      <div style="display:flex;justify-content:space-between;gap:20px;align-items:center;">
        <div>
          <div style="font-size:1.55rem;font-weight:800;">{symbol}.TW ｜ {name}</div>
          <div style="margin-top:12px;">
            <span class="badge">{timeframe}</span>
            <span class="badge">多空雙向</span>
            <span class="badge">雙均線交叉</span>
            <span class="small-muted">策略 MA5 / MA20 交叉　實際回測 {len(d)} 根 K 線</span>
          </div>
        </div>
        <div style="text-align:center;">
          <div class="small-muted" style="margin-bottom:8px;">{timeframe}｜均線狀態</div>
          {status_html}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("最新收盤價", f"{latest['Close']:.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    c2.metric("短期均線 MA5", f"{latest['MA5']:.2f}")
    c3.metric("長期均線 MA20", f"{latest['MA20']:.2f}")
    c4.metric("K線筆數", len(d))

    # -------------------- 主圖 --------------------
    st.markdown(f"### {timeframe}K線｜多空雙向｜雙均線交叉（MA5／MA20）")
    st.caption("圖表內容：K線、策略均線、成交量、CCI。訊號規則：MA5 上穿 MA20 為多方訊號；下穿為空方訊號。訊號於該根 K 線收盤後確認，下一根 K 線開盤價進行模擬成交。")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.74,0.26])

    # 台股紅漲綠跌
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="K線",
        increasing_line_color="#ff3b30", increasing_fillcolor="#ff3b30",
        decreasing_line_color="#00b76a", decreasing_fillcolor="#00b76a"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index,y=d["MA5"],mode="lines",name="MA5",
                             line=dict(color="#ffd400",width=2.2)), row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d["MA20"],mode="lines",name="MA20",
                             line=dict(color="#9b4dff",width=2.2)), row=1,col=1)

    bulls = d[d["Signal"]=="模擬轉多訊號"]
    bears = d[d["Signal"]=="模擬轉空訊號"]
    if not bulls.empty:
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls["Low"]-bulls["ATR14"].fillna(0)*.25,
            mode="markers", name="模擬轉多訊號",
            marker=dict(symbol="triangle-up",size=12,color="#ff4d4f",
                        line=dict(color="white",width=1.4))
        ), row=1,col=1)
    if not bears.empty:
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears["High"]+bears["ATR14"].fillna(0)*.25,
            mode="markers", name="模擬轉空訊號",
            marker=dict(symbol="triangle-down",size=12,color="#00c176",
                        line=dict(color="white",width=1.4))
        ), row=1,col=1)

    vol_colors = np.where(d["Close"]>=d["Open"], "#ff3b30", "#00b76a")
    fig.add_trace(go.Bar(x=d.index,y=d["Volume"],name="成交量",
                         marker_color=vol_colors, opacity=.85), row=2,col=1)

    fig.update_layout(
        height=720, template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=50,b=10),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="#11161d", bordercolor="#44505c", borderwidth=1)
    )
    fig.update_yaxes(title_text="價格", side="right", row=1,col=1)
    fig.update_yaxes(title_text="成交量", side="right", row=2,col=1)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- 回測績效 --------------------
    st.markdown("## 回測績效")
    a,b,c,dcol = st.columns(4)
    a.metric("期末資產", f"{m['期末資產']:,.0f}", "含已實現與未實現損益")
    b.metric("策略總報酬率（含未實現）", f"{m['策略總報酬率']*100:.2f}%",
             f"相較買進持有 {(m['策略總報酬率']-m['買進持有總報酬率'])*100:+.2f} 個百分點")
    c.metric("買進持有總報酬率", f"{m['買進持有總報酬率']*100:.2f}%", "同期買進持有至期末")
    dcol.metric("最大回撤", f"{m['最大回撤']*100:.2f}%", "資產高點至其後低點最大跌幅")

    e,f,g,h = st.columns(4)
    e.metric("已實現損益", f"{m['已實現損益']:+,.0f}", "僅計已完成交易")
    f.metric("未實現損益", f"{m['未實現損益']:+,.0f}", "期末仍持有部位的浮動損益")
    g.metric("勝率（已完成）", f"{m['勝率']*100:.2f}%", f"共 {m['完成交易']} 筆完成交易")
    state = "持有多單" if position==1 else ("持有空單" if position==-1 else "空手")
    h.metric("目前狀態", state, "多空雙向")

    long_entries = int((trades["方向"]=="多單").sum()) if not trades.empty else 0
    short_entries = int((trades["方向"]=="空單").sum()) if not trades.empty else 0
    st.markdown(f"""
    <div class="stock-card">
      <b>交易統計</b><br><br>
      <span class="badge">完成交易 {m['完成交易']}</span>
      <span class="badge">做多進場 {long_entries}</span>
      <span class="badge">放空進場 {short_entries}</span>
      <span class="badge">未平倉 {1 if position else 0}</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("計算基準：策略交易績效與買進持有總報酬率皆使用 FinMind 還原後的日線資料。雙向回測為理論模型，未納入券源、融券、借券限制、借券費、追繳、強制回補、手續費、證交稅與滑價。")

    # -------------------- Equity curve --------------------
    if not trades.empty:
        eq = pd.DataFrame({
            "日期": trades["出場日期"],
            "策略資產": capital + trades["報酬金額"].cumsum()
        }).set_index("日期")
        peak = eq["策略資產"].cummax()
        eq["回撤(%)"] = (eq["策略資產"]/peak - 1)*100
        eqfig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.7,.3],vertical_spacing=.04)
        eqfig.add_trace(go.Scatter(x=eq.index,y=eq["策略資產"],name="策略資產",
                                   line=dict(color="#19c7f3",width=2.3)),row=1,col=1)
        eqfig.add_trace(go.Scatter(x=eq.index,y=eq["回撤(%)"],name="回撤(%)",
                                   line=dict(color="#00b76a",width=2)),row=2,col=1)
        eqfig.update_layout(height=430,template="plotly_dark",paper_bgcolor="#0b0f14",plot_bgcolor="#0b0f14")
        st.plotly_chart(eqfig,use_container_width=True)

    # -------------------- 智慧選股掃描 --------------------
    with st.expander("智慧選股掃描"):
        st.write("此區保留為候選股票掃描入口；目前版本先聚焦單一標的雙向回測與結果檢視。")

    # -------------------- 基準資訊 --------------------
    with st.expander("回測資料與計算基準", expanded=True):
        l,r = st.columns(2)
        with l:
            st.write(f"實際代號：{symbol}.TW")
            st.write(f"實際期間：{d.index.min().date()}～{d.index.max().date()}")
            st.write(f"週期生成：{'直接使用日線資料' if timeframe=='日線' else '由日線重採樣為週線'}")
            st.write("回測資料版本：本次最新下載並儲存資料版本")
            st.write("前次資料版本：尚無可比較紀錄")
            st.write("成交假設：策略訊號於該根 K 線收盤後確認，於下一根 K 線開盤價模擬成交。")
        with r:
            st.write(f"中文名稱：{name}")
            st.write(f"資料筆數：{len(d)}")
            st.write("價格模式：FinMind 台股日線資料")
            st.write("交易方向：多空雙向")
            st.write("策略類型：雙均線交叉")
            st.write("策略條件：MA5 上穿 MA20 為多方訊號；下穿為空方訊號")
            st.write(f"資料下載時間：{datetime.now().isoformat(timespec='seconds')}")

    # -------------------- 完成交易 --------------------
    st.markdown("### 已完成交易紀錄")
    if trades.empty:
        st.info("目前沒有完整完成交易。")
    else:
        t = trades.copy()
        for c in ["進場訊號日期","進場日期","進場K線日期","出場訊號日期","出場日期","出場K線日期"]:
            t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
        t["進場價格"] = t["進場價格"].round(2)
        t["出場價格"] = t["出場價格"].round(2)
        t["報酬金額"] = t["報酬金額"].round(0).astype(int)
        t["報酬率%"] = t["報酬率%"].round(2)
        st.dataframe(t, use_container_width=True, hide_index=True, height=330)

    # -------------------- 每根K線明細 --------------------
    st.markdown("### 每根 K 線策略明細")
    detail = d.reset_index().copy()
    detail["日期"] = detail["Date"].dt.strftime("%Y/%m/%d")
    detail["收盤價"] = detail["Close"].round(2)
    detail["成交量"] = detail["Volume"].fillna(0).astype(int)
    detail["MA5"] = detail["MA5"].round(2)
    detail["MA20"] = detail["MA20"].round(2)
    detail["CCI20"] = detail["CCI20"].round(2)
    detail["策略訊號"] = detail["Signal"].replace("", "無新訊號")

    exec_map = {}
    action_map = {}
    pos = "模擬空手"
    for _, row in detail.iterrows():
        sig = row["策略訊號"]
        if sig == "模擬轉多訊號":
            pos = "模擬多單"
            action_map[row["日期"]] = "模擬轉多"
        elif sig == "模擬轉空訊號":
            pos = "模擬空單"
            action_map[row["日期"]] = "模擬轉空"
        else:
            action_map[row["日期"]] = "無回測動作"
        exec_map[row["日期"]] = pos

    detail["模擬成交日"] = ""
    detail["模擬成交價"] = np.nan
    # 訊號下一根 K 線成交
    for i in range(len(detail)-1):
        if detail.loc[i,"策略訊號"] != "無新訊號":
            detail.loc[i+1,"模擬成交日"] = detail.loc[i+1,"日期"]
            detail.loc[i+1,"模擬成交價"] = detail.loc[i+1,"Open"]
    detail["策略回測動作"] = detail["日期"].map(action_map)
    detail["策略回測部位"] = detail["日期"].map(exec_map)

    show_cols = ["日期","收盤價","成交量","MA5","MA20","CCI20","策略訊號",
                 "模擬成交日","模擬成交價","策略回測動作","策略回測部位"]
    st.dataframe(detail[show_cols].tail(80), use_container_width=True, hide_index=True, height=520)

except Exception as e:
    st.error("分析或回測發生錯誤。")
    st.code(f"{type(e).__name__}: {e}")
