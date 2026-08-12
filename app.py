import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title='台股強勢分析 MVP', layout='wide')
st.title('📈 台股強勢分析 MVP')
st.caption('第二版：加強台股資料抓取、上市櫃自動判斷、錯誤診斷｜Yahoo Finance 測試用途')

def clean_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    need = ['Open','High','Low','Close','Volume']
    if any(c not in d.columns for c in need):
        return pd.DataFrame()
    d = d[need].apply(pd.to_numeric, errors='coerce')
    d = d.dropna(subset=['Open','High','Low','Close'])
    d['Volume'] = d['Volume'].fillna(0)
    return d[~d.index.duplicated(keep='last')].sort_index()

def fetch_one(ticker, period):
    errors = []
    try:
        d = yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=False,
                                      actions=False, repair=True, timeout=20)
        d = clean_df(d)
        if not d.empty:
            return d, 'Ticker.history', errors
    except Exception as e:
        errors.append(f'{ticker} history: {type(e).__name__}: {e}')
    try:
        d = yf.download(ticker, period=period, interval='1d', auto_adjust=False,
                        actions=False, repair=True, progress=False, threads=False,
                        timeout=20)
        d = clean_df(d)
        if not d.empty:
            return d, 'yf.download', errors
    except Exception as e:
        errors.append(f'{ticker} download: {type(e).__name__}: {e}')
    return pd.DataFrame(), None, errors

@st.cache_data(ttl=900, show_spinner=False)
def load_price(symbol, period='2y'):
    s = symbol.strip().upper()
    candidates = [s] if '.' in s else [f'{s}.TW', f'{s}.TWO']
    all_errors = []
    best = (pd.DataFrame(), None, None)
    for ticker in candidates:
        d, method, errors = fetch_one(ticker, period)
        all_errors.extend(errors)
        if len(d) >= 70:
            return d, ticker, method, all_errors
        if len(d) > len(best[0]):
            best = (d, ticker, method)
    return best[0], best[1], best[2], all_errors

def indicators(d):
    d = d.copy()
    for n in [5,10,20,60]:
        d[f'MA{n}'] = d['Close'].rolling(n).mean()
    low9, high9 = d['Low'].rolling(9).min(), d['High'].rolling(9).max()
    rsv = (d['Close']-low9)/(high9-low9).replace(0,np.nan)*100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    e12, e26 = d['Close'].ewm(span=12,adjust=False).mean(), d['Close'].ewm(span=26,adjust=False).mean()
    d['MACD'] = e12-e26
    d['Signal'] = d['MACD'].ewm(span=9,adjust=False).mean()
    d['Hist'] = d['MACD']-d['Signal']
    delta = d['Close'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14,adjust=False).mean()/loss.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    d['RSI'] = 100-(100/(1+rs))
    d['VOL_MA5'] = d['Volume'].rolling(5).mean()
    d['HIGH20_PREV'] = d['High'].shift(1).rolling(20).max()
    return d

def score_stock(d):
    x,p = d.iloc[-1], d.iloc[-2]
    score,reasons = 0,[]
    tests = [
        (x['Close']>x['MA20'],10,'股價站上 MA20'),
        (x['MA20']>p['MA20'],10,'MA20 向上'),
        (x['Close']>x['MA60'],10,'股價站上 MA60'),
        (x['MA5']>x['MA10'],10,'MA5 > MA10'),
        (x['MA10']>x['MA20'],10,'MA10 > MA20'),
        (x['Volume']>x['VOL_MA5'],10,'成交量高於 5 日均量'),
        (x['Volume']>x['VOL_MA5']*1.5,10,'成交量放大 1.5 倍'),
        (x['K']>x['D'],7,'KD 偏多'),
        (x['MACD']>x['Signal'],7,'MACD 偏多'),
        (x['RSI']>50,6,'RSI > 50'),
        (x['Close']>x['HIGH20_PREV'],10,'突破前 20 日高點')]
    for ok,pts,text in tests:
        try:
            if pd.notna(ok) and bool(ok):
                score += pts; reasons.append(text)
        except Exception:
            pass
    return min(int(score),100), reasons

def grade(s):
    return '🔥 A級強勢' if s>=85 else '🟢 B級觀察' if s>=70 else '🟡 中性' if s>=55 else '⚠️ 偏弱' if s>=40 else '🔴 弱勢'

with st.sidebar:
    st.header('設定')
    symbol = st.text_input('股票代號', value='2330', max_chars=12)
    period = st.selectbox('資料期間', ['6mo','1y','2y','5y'], index=2)
    refresh = st.checkbox('重新下載最新資料')
    run = st.button('開始分析', type='primary', use_container_width=True)

if refresh:
    st.cache_data.clear()

if not run:
    st.info('輸入股票代號後按「開始分析」。建議先測 2330、2317、2454。')
    st.stop()

with st.spinner('正在抓取台股資料...'):
    df, ticker_used, method_used, errors = load_price(symbol, period)

if df.empty:
    st.error('Yahoo Finance 目前沒有成功回傳這檔股票的日線資料。')
    st.warning('若股票代號正確，可能是 Yahoo Finance 暫時限制或連線異常，請稍後再試。')
    with st.expander('查看技術診斷'):
        st.write('嘗試代號：', symbol if '.' in symbol else f'{symbol}.TW / {symbol}.TWO')
        for e in errors: st.code(e)
    st.stop()

if len(df) < 70:
    st.error(f'目前只取得 {len(df)} 筆日線資料，至少需要約 70 筆才能完整計算 MA60。')
    st.info('請把「資料期間」改成 1y、2y 或 5y。')
    st.stop()

d = indicators(df)
score,reasons = score_stock(d)
x,p = d.iloc[-1], d.iloc[-2]
pct = (x['Close']/p['Close']-1)*100 if p['Close'] else 0
st.success(f'資料抓取成功：{ticker_used}｜{method_used}｜共 {len(df)} 筆日線')

c1,c2,c3,c4 = st.columns(4)
c1.metric('股票代號', ticker_used)
c2.metric('收盤價', f"{x['Close']:.2f}", f'{pct:+.2f}%')
c3.metric('綜合評分', f'{score}/100')
c4.metric('系統判定', grade(score))

st.subheader('K線＋均線')
fig = go.Figure()
fig.add_trace(go.Candlestick(x=d.index,open=d['Open'],high=d['High'],low=d['Low'],close=d['Close'],name='K線'))
for n in [5,10,20,60]:
    fig.add_trace(go.Scatter(x=d.index,y=d[f'MA{n}'],mode='lines',name=f'MA{n}'))
fig.update_layout(height=600,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)

left,right = st.columns(2)
with left:
    st.subheader('KD')
    f=go.Figure(); f.add_trace(go.Scatter(x=d.index,y=d['K'],name='K')); f.add_trace(go.Scatter(x=d.index,y=d['D'],name='D')); f.update_layout(height=300); st.plotly_chart(f,use_container_width=True)
    st.subheader('RSI')
    f=go.Figure(); f.add_trace(go.Scatter(x=d.index,y=d['RSI'],name='RSI14')); f.update_layout(height=300); st.plotly_chart(f,use_container_width=True)
with right:
    st.subheader('MACD')
    f=go.Figure(); f.add_trace(go.Scatter(x=d.index,y=d['MACD'],name='MACD')); f.add_trace(go.Scatter(x=d.index,y=d['Signal'],name='Signal')); f.add_trace(go.Bar(x=d.index,y=d['Hist'],name='Hist')); f.update_layout(height=300); st.plotly_chart(f,use_container_width=True)
    st.subheader('成交量')
    f=go.Figure(); f.add_trace(go.Bar(x=d.index,y=d['Volume'],name='Volume')); f.add_trace(go.Scatter(x=d.index,y=d['VOL_MA5'],name='5日均量')); f.update_layout(height=300); st.plotly_chart(f,use_container_width=True)

st.subheader('評分依據')
for r in reasons: st.write(f'✅ {r}')
with st.expander('資料資訊'):
    st.write('實際 Yahoo 代號：', ticker_used)
    st.write('抓取方式：', method_used)
    st.write('資料筆數：', len(df))
    st.write('資料起日：', df.index.min())
    st.write('資料迄日：', df.index.max())
st.info('此版本僅做技術分析與程式測試，不構成投資建議。')
