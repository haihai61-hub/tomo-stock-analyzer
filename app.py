import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import google.generativeai as genai

st.set_page_config(page_title="TOMO流 需給×チャート全自動診断 ＋ AI", page_icon="📈", layout="wide")

components.html(
    """<script>
    const doc = window.parent.document;
    doc.documentElement.lang = 'ja';
    doc.documentElement.setAttribute('translate', 'no');
    doc.documentElement.classList.add('notranslate');
    </script>""",
    height=0, width=0
)

# -------------------------------------------------------------
# 銘柄名 ＆ 信用買残（東証公表値）のスクレイピング
# -------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_info_and_margin(ticker_code: str):
    code = ticker_code.strip().upper()
    url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    company_name = code
    margin_buy, margin_sell, margin_ratio, margin_date = 0, 0, 0.0, ""

    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            title_text = soup.title.string if soup.title else ""
            if "【" in title_text:
                company_name = title_text.split("【")[0].strip()

            text_all = soup.get_text()

            m_buy = re.search(r'信用買残[^\d]*?([0-9,]+)\s*株', text_all)
            if m_buy:
                margin_buy = int(m_buy.group(1).replace(",", ""))

            m_sell = re.search(r'信用売残[^\d]*?([0-9,]+)\s*株', text_all)
            if m_sell:
                margin_sell = int(m_sell.group(1).replace(",", ""))

            m_ratio = re.search(r'信用倍率[^\d]*?([0-9\.]+)\s*倍', text_all)
            if m_ratio:
                margin_ratio = float(m_ratio.group(1))

            m_date = re.search(r'信用買残[^\d]*?[0-9,]+\s*株[^\d]*?([0-9]{2}/[0-9]{2})', text_all)
            if m_date:
                margin_date = m_date.group(1)
    except Exception:
        pass

    return {
        "name": company_name,
        "margin_buy": margin_buy,
        "margin_sell": margin_sell,
        "margin_ratio": margin_ratio,
        "margin_date": margin_date
    }

# -------------------------------------------------------------
# 東証実データの取得
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_real_market_data(ticker_code: str):
    code = ticker_code.strip().upper()
    symbol = f"{code}.T" if not code.endswith(".T") else code

    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period="6mo")
        if df.empty or len(df) < 5:
            return None, f"銘柄「{symbol}」のデータが見つかりませんでした。コードを確認してください。"

        df.index = df.index.tz_localize(None)
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else latest_row
        
        close_price = round(float(latest_row['Close']), 1)
        prev_close = round(float(prev_row['Close']), 1)
        diff_price = round(close_price - prev_close, 1)
        diff_pct = round((diff_price / prev_close) * 100, 2) if prev_close > 0 else 0
        latest_vol = int(latest_row['Volume'])
        latest_date = df.index[-1].strftime('%Y/%m/%d')

        meta = {
            "symbol": symbol,
            "latest_date": latest_date,
            "close": close_price,
            "diff": f"{diff_price:+,.1f} ({diff_pct:+,.2f}%)",
            "volume": latest_vol
        }
        return df, meta
    except Exception as e:
        return None, f"データ取得エラー: {e}"

# -------------------------------------------------------------
# 需給指標の計算
# -------------------------------------------------------------
def calculate_metrics(df: pd.DataFrame, margin_buy: int, current_price: float):
    regular_volume = int(df['Volume'].tail(40).median())
    turnover_days = margin_buy / regular_volume if regular_volume > 0 else 0
    
    bins = 25
    price_min = df['Low'].min()
    price_max = df['High'].max()
    counts, edges = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    
    upper_volume = counts[edges[1:] > current_price].sum()
    total_volume = counts.sum()
    shikori_rate = (upper_volume / total_volume) * 100 if total_volume > 0 else 0
    
    return {
        "regular_volume": regular_volume,
        "turnover_days": round(turnover_days, 2),
        "shikori_rate": round(shikori_rate, 1),
        "price_bins": edges,
        "vol_counts": counts
    }

# -------------------------------------------------------------
# 定型ルール判定（安全な受け皿）
# -------------------------------------------------------------
def run_fallback_diagnostic(company_name: str, metrics: dict, stockscope_inputs: dict):
    shikori = metrics['shikori_rate']
    overhang = metrics['turnover_days']
    seido = stockscope_inputs['seido_ratio']

    if shikori >= 40:
        status = "⚠️ 上値激重・手出し無用（しこり優勢）"
        reason = f"現在値より上に出来高の **{shikori}%** が滞留。上値はすべて戻り待ちの含み損玉（やれやれ売り）の壁になります。"
        action = "出来高が爆発的に急増してこの価格帯を力強く食い破るまで、順張り・買いは見送るのが賢明です。"
    elif shikori <= 15:
        status = "🚀 青天井・全員含み益（上値軽快）"
        reason = f"信用買残はありますが、上値のしこり玉はわずか **{shikori}%**。保有者の大半が含み益のため、売り圧力が極めて限定的です。"
        action = "高値ブレイクや押し目形成に素直についていきやすい理想的な需給環境です。"
    else:
        status = "⚖️ 需給拮抗・もみ合い警戒"
        reason = f"しこり率は **{shikori}%**。現在値と出来高の山が拮抗しており、方向感を欠きやすい水準です。"
        action = "出来高を伴って節目をブレイクするのを確認するまで様子見が無難です。"

    return f"""### 【総合判定】{status}

**1. {company_name} の需給構造**
- 信用買残（{stockscope_inputs['margin_buy']:,} 株）に対し、常時出来高の **{overhang} 倍**。
- {reason}
- 制度信用比率は **{seido}%**（6ヶ月期日の投げ売りリスク）。

**2. トレード結論**
- {action}
"""

# -------------------------------------------------------------
# Gemini AIエンジン（レートリミット保護付き）
# -------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_ai_report(api_key: str, symbol: str, company_name: str, close_price: float, diff: str, volume: int, reg_vol: int, margin_buy: int, turnover_days: float, shikori: float, seido: float, short_trend: str):
    genai.configure(api_key=api_key)
    
    prompt = f"""
あなたは株式需給分析の専門家（トレーダーTOMO流）です。
以下の客観的な東証市場データおよび需給数値をもとに、歯切れの良い実践的なトレード診断レポートを作成してください。

【銘柄情報】
- 銘柄名: {company_name} ({symbol})
- 確定終値: {close_price:,} 円 ({diff})
- 直近取引日出来高: {volume:,} 株
- 平常時の常時出来高（直近中央値）: {reg_vol:,} 株

【需給データ】
- 信用買残: {margin_buy:,} 株
- 買残 / 常時出来高: {turnover_days} 倍（消化にかかる日数感）
- 推定しこり率: {shikori}%（現在値より上で捕まっている含み損玉の割合）
- 制度信用比率: {seido}%（6ヶ月期日リスクの度合い）
- 機関空売り動向: {short_trend}

【診断レポートの構成】
1. 【総合判定】: 「🚀 青天井・全員含み益（上値軽快）」「⚠️ 上値激重・手出し無用（しこり優勢）」「⚖️ 需給拮抗・もみ合い警戒」等の明確な結論。
2. 【需給の急所】: しこり玉の薄さ/重さ、出来高と買残の消化力、期日の投げリスクなどを簡潔に解説。
3. 【トレード戦略】: 節目となるライン（支持帯・抵抗帯）、ブレイク狙い・押し目買い・見送りの具体判断。
※不要な前置きや免責事項は省き、マークダウン形式でシャープに出力してください。
"""

    models_to_try = ["gemini-3.8-flash", "gemini-3.7-flash"]
    last_err_type = "unknown"

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                footer = f"\n\n---\n*🤖 診断エンジン: {model_name}（リアルタイムAI解析）*"
                return response.text + footer, None, None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                return None, "rate_limit", err_str
            last_err_type = err_str
            continue
            
    return None, "other", last_err_type

# -------------------------------------------------------------
# UIレイアウト
# -------------------------------------------------------------
st.title("📊 TOMO流 需給×チャート全自動診断 ＋ AI")
st.caption("東証の完全一致実データ ＋ 価格帯別出来高 ＋ Gemini AI診断")

# サイドバー（フォーム化して無駄なAPI呼び出しを防止）
with st.sidebar:
    st.header("🔍 分析設定")
    with st.form("analysis_form"):
        ticker_input = st.text_input("東証コード（例: 6857, 446A, 285A）", value="6857")
        
        scraped_info = fetch_info_and_margin(ticker_input)
        auto_buy = scraped_info['margin_buy'] if scraped_info['margin_buy'] > 0 else 1000000

        st.markdown("---")
        st.subheader("📋 需給パラメーター")
        margin_buy_input = st.number_input("信用買残（株数）", value=auto_buy, step=10000)
        seido_ratio_input = st.slider("制度信用比率（%）", 0.0, 100.0, 40.0, 5.0)
        short_trend_input = st.selectbox("機関の空売り動向", ["空売り残高なし（該当なし）", "買い戻し（ショートカバー期待）", "横ばい・変化なし", "売り増し傾向（重圧）"])

        # Secretsから取得、なければ空文字
        api_key_env = st.secrets.get("GEMINI_API_KEY", "")
        
        submitted = st.form_submit_button("🔍 この条件で分析を実行", use_container_width=True)

with st.spinner(f"銘柄「{ticker_input}」のデータを取得・解析中..."):
    res = fetch_real_market_data(ticker_input)

if res[0] is None:
    st.error(res[1])
else:
    df, meta_info = res
    company_name = scraped_info['name']
    stockscope_data = {"margin_buy": int(margin_buy_input), "seido_ratio": seido_ratio_input, "short_trend": short_trend_input}
    metrics = calculate_metrics(df, stockscope_data['margin_buy'], meta_info['close'])

    audit_msg = f"【東証公式実データ照合完了】 {company_name} ({meta_info['symbol']}) | 終値: {meta_info['close']:,.1f}円 | 出来高: {meta_info['volume']:,}株 | 最新公表信用買残: {scraped_info['margin_buy']:,}株 ({scraped_info['margin_date']}時点)"
    st.success(audit_msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"東証終値 ({meta_info['latest_date']})", f"{meta_info['close']:,} 円", meta_info['diff'])
    c2.metric("買残 / 常時出来高", f"{metrics['turnover_days']} 倍")
    c3.metric("推定しこり率", f"{metrics['shikori_rate']} %",
              delta="-重体" if metrics['shikori_rate'] >= 40 else ("+軽快" if metrics['shikori_rate'] <= 15 else "均衡"),
              delta_color="inverse")
    c4.metric("信用倍率（参考）", f"{scraped_info['margin_ratio']} 倍" if scraped_info['margin_ratio'] > 0 else "算出不可")

    col_chart, col_ai = st.columns([1.3, 1.0])

    with col_chart:
        st.subheader(f"📈 {company_name} ({meta_info['symbol']}) 日足 & VRVP")
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.75, 0.25], horizontal_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="日足"), row=1, col=1)
        fig.add_hline(y=meta_info['close'], line_dash="dot", line_color="orange", annotation_text="確定終値", row=1, col=1)

        colors = ['rgba(239, 83, 80, 0.7)' if metrics['price_bins'][i] >= meta_info['close'] 
                  else 'rgba(66, 165, 245, 0.7)' for i in range(len(metrics['vol_counts']))]
        bin_centers = (metrics['price_bins'][:-1] + metrics['price_bins'][1:]) / 2
        fig.add_trace(go.Bar(x=metrics['vol_counts'], y=bin_centers, orientation='h', marker_color=colors, name="出来高Profile"), row=1, col=2)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_ai:
        st.subheader(f"🤖 TOMO流 診断レポート [{company_name}]")
        
        if api_key_env:
            with st.spinner("Gemini AIが需給構造を深掘り分析中..."):
                ai_text, err_type, err_raw = get_cached_ai_report(
                    api_key_env,
                    meta_info['symbol'],
                    company_name,
                    meta_info['close'],
                    meta_info['diff'],
                    meta_info['volume'],
                    metrics['regular_volume'],
                    stockscope_data['margin_buy'],
                    metrics['turnover_days'],
                    metrics['shikori_rate'],
                    stockscope_data['seido_ratio'],
                    stockscope_data['short_trend']
                )
                if ai_text:
                    st.markdown(ai_text)
                elif err_type == "rate_limit":
                    st.info("⏳ **無料枠の短時間アクセス制限（1分間5回まで）中です。**\n約30〜40秒待ってから再実行してください。それまでは高速ルール判定を表示します。")
                    st.markdown(run_fallback_diagnostic(company_name, metrics, stockscope_data))
                else:
                    st.warning("⚠️ 通信エラーが発生したため、ルールベース判定を表示します。")
                    with st.expander("🔍 エラー詳細"):
                        st.code(err_raw, language="text")
                    st.markdown(run_fallback_diagnostic(company_name, metrics, stockscope_data))
        else:
            st.markdown(run_fallback_diagnostic(company_name, metrics, stockscope_data))
            st.info("💡 StreamlitのSecretsに `GEMINI_API_KEY` を設定すると、完全自動でAI診断に切り替わります。")

        st.markdown("---")
        st.write(f"- 直近出来高: **{meta_info['volume']:,} 株**")
        st.write(f"- 平常時の常時出来高: **{metrics['regular_volume']:,} 株**")
        st.write(f"- 信用売残（参考）: **{scraped_info['margin_sell']:,} 株**")
        st.write(f"- 機関空売り動向: **{stockscope_data['short_trend']}**")
