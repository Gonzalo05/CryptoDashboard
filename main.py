import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page configigurataion
st.set_page_config(page_title="Crypto Dashboard", page_icon="💰", layout="wide")

# constants 
UA = {"User-Agent": "IntroCS-Student-Streamlit/1.0"}
DEFAULT_COINS = ["bitcoin", "ethereum", "solana"]
VS_CURRENCIES = ["usd", "eur", "gbp", "jpy"]
DEFAULT_DAYS = 30
PAUSE = 0.5  

#  Helpers 
# we use cache throughout differnt functions to avoid calling the function too many times
@st.cache_data(ttl=180)
def markets_snapshot(vs_currency="usd", per_page=200):
    # Returns a DataFrame of top coins by market cap (with 24h % change).
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": min(per_page, 250), 
        "page": 1,
        "price_change_percentage": "24h",
    }
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = ["market_cap_rank","name","symbol","current_price","market_cap","price_change_percentage_24h"]
    df = pd.DataFrame(data)[cols].dropna(subset=["market_cap_rank"]).sort_values("market_cap_rank")
    df["symbol"] = df["symbol"].str.upper()
    return df


# Returns a DataFrame of top coins by market cap (id, symbol, name).
@st.cache_data(ttl=600)
def list_top_coins(vs_currency="usd", per_page=100):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": per_page, "page": 1}
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = [{"id": d["id"], "symbol": d["symbol"].upper(), "name": d["name"]} for d in data]
    return pd.DataFrame(rows)

# Returns a DataFrame of [date, price] for coin_id over the last days.
@st.cache_data(ttl=600)
def fetch_market_chart(coin_id, vs_currency="usd", days=30, interval="daily"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": interval}
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    data = r.json().get("prices", [])
    if not data:
        return pd.DataFrame(columns=["date", coin_id])

    # Convert ms timestamps to date (UTC)
    out = pd.DataFrame(data, columns=["ts_ms", "price"])
    out["date"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True).dt.date
    out = out.groupby("date", as_index=False)["price"].last() 
    out.rename(columns={"price": coin_id}, inplace=True)
    return out[["date", coin_id]]

def pct_change(df):
    return df.pct_change().dropna(how="all")

def summarize_returns(s):
    vol = float(s.std()) if len(s) > 1 else np.nan
    cum = float((1 + s).prod() - 1) if len(s) > 0 else np.nan
    return vol, cum

# SIDEBAR controls
st.sidebar.title("⚙️ Controls")

with st.sidebar.form("controls"):
    vs = st.selectbox("Quote currency", VS_CURRENCIES, index=0)
    days = st.number_input("Lookback (days)", min_value=7, max_value=365, value=DEFAULT_DAYS, step=1)
    use_top = st.checkbox("Pick from Top Market-Cap Coins", value=True)

    selected_labels = []
    coins_text = ""
    if use_top:
        # Cached. One lightweight call, not the heavy per-coin price calls
        try:
            top_df = list_top_coins(vs)
            options = top_df["id"].tolist()
            labels = [f"{row['name']} ({row['symbol']})" for _, row in top_df.iterrows()]
            label_to_id = dict(zip(labels, options))
            default_labels = [lbl for lbl in labels if label_to_id[lbl] in DEFAULT_COINS]
            selected_labels = st.multiselect("Select coins", labels, default=default_labels)
        except Exception as e:
            st.caption("Could not load top coins; fallback to manual IDs.")
            use_top = False 

    if not use_top:
        coins_text = st.text_input("Enter coin IDs (comma-separated)", ",".join(DEFAULT_COINS))

    submitted = st.form_submit_button("Update dashboard ✅")

# Stop until the user clicks the button
if not submitted:
    st.stop()

# Resolve selected coins AFTER submit
if use_top and selected_labels:
    coins = [label_to_id[lbl] for lbl in selected_labels]
else:
    coins = [c.strip().lower() for c in coins_text.split(",") if c.strip()] or DEFAULT_COINS

st.sidebar.markdown("---")
st.sidebar.caption("Data: CoinGecko (no API key). Cached ~10 min.")

# HEADER
st.title("💰 Crypto Dashboard")
st.write(f"Quotes in **{vs.upper()}**, last **{days}** days.")

# MARKET SNAPTSHOT TABLES
try:
    snap = markets_snapshot(vs)
    snap["Coin"] = snap.apply(lambda r: f"{r['name']} ({r['symbol'].upper()})", axis=1)

    # here we define which columns to show
    cols_top = ["Coin", "current_price", "market_cap", "price_change_percentage_24h"]
    cols_gl  = ["Coin", "price_change_percentage_24h", "current_price", "market_cap"]

    top_cap = snap.head(10)[cols_top]
    movers  = snap.dropna(subset=["price_change_percentage_24h"]).copy()
    gainers = movers.sort_values("price_change_percentage_24h", ascending=False).head(10)[cols_gl]
    losers  = movers.sort_values("price_change_percentage_24h", ascending=True ).head(10)[cols_gl]

    # Then rename columns accordingly
    rename_cols = {
        "current_price": "Current Price",
        "market_cap": "Market Cap",
        "price_change_percentage_24h": "24h Change"
    }
    top_cap.rename(columns=rename_cols, inplace=True)
    gainers.rename(columns=rename_cols, inplace=True)
    losers.rename(columns=rename_cols, inplace=True)

    # We apply styling for the in the 24h Change red/green
    def style_tbl(df):
        def pct_color(v):
            import pandas as pd
            if pd.isna(v): return ""
            return "color: #16a34a;" if v >= 0 else "color: #dc2626;"  # green/red
        sty = (df.style
               .format({
                   "Current Price": "{:,.4f}",
                   "Market Cap": "{:,.0f}",
                   "24h Change": "{:+.2f}%"
               })
               .hide(axis="index")
               .applymap(pct_color, subset=["24h Change"]))
        return sty

    st.subheader("📈 Market Snapshot (Top / Gainers / Losers)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("Top by Market Cap")
        st.dataframe(style_tbl(top_cap), use_container_width=True, height=320, hide_index=True)

    with c2:
        st.caption("Biggest Gainers (24h)")
        st.dataframe(style_tbl(gainers), use_container_width=True, height=320, hide_index=True)

    with c3:
        st.caption("Biggest Losers (24h)")
        st.dataframe(style_tbl(losers), use_container_width=True, height=320, hide_index=True)

except requests.HTTPError as e:
    st.warning(f"Market snapshot unavailable: HTTP {e.response.status_code}")
except Exception as e:
    st.warning(f"Market snapshot error: {e}")

# FETCH & ASSEMBLE PRICE TABLE
frames = []
errors = []

for cid in coins:
    try:
        df = fetch_market_chart(cid, vs_currency=vs, days=days, interval="daily")
        frames.append(df)
        time.sleep(PAUSE)
    except requests.HTTPError as e:
        errors.append(f"{cid}: HTTP {e}")
    except Exception as e:
        errors.append(f"{cid}: {e}")

if errors:
    st.warning("Some coins failed:\n- " + "\n- ".join(errors))

if not frames:
    st.stop()

prices = frames[0]
for df in frames[1:]:
    prices = prices.merge(df, on="date", how="outer")

prices = prices.sort_values("date").set_index("date")
st.subheader("Prices (table)")
st.dataframe(prices.style.format("{:,.4f}"))

# METRICS
returns = pct_change(prices)
summary_rows = []
for cid in prices.columns:
    if cid in returns:
        vol, cum = summarize_returns(returns[cid].dropna())
    else:
        vol, cum = np.nan, np.nan
    last = float(prices[cid].dropna().iloc[-1]) if cid in prices else np.nan
    summary_rows.append({"coin": cid, "last_price": last, "cum_return": cum, "volatility": vol})
summary = pd.DataFrame(summary_rows).set_index("coin").sort_values("volatility", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Tracked coins", len(prices.columns))
c2.metric("Days of data", prices.shape[0])
avg_vol = float(summary["volatility"].mean()) if not summary.empty else np.nan
c3.metric("Avg 30d volatility", f"{avg_vol:.4f}" if np.isfinite(avg_vol) else "–")

# CHARTS
st.subheader("📊 Volatility & Correlation Overview")

col1, col2 = st.columns(2)

with col1:
    st.caption("Volatility (std of daily returns)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    summary_sorted = summary.sort_values("volatility", ascending=False)
    ax1.bar(summary_sorted.index, summary_sorted["volatility"], color="#4ade80")
    ax1.set_ylabel("Volatility")
    ax1.set_xlabel("Coin")
    ax1.set_title("Volatility (lookback matches slider)")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig1)

with col2:
    st.caption("Return Correlation (Last 30 Days)")
    if returns.shape[1] >= 2:
        corr = returns.corr()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        im = ax2.imshow(corr, interpolation="nearest", cmap="RdYlGn")
        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_yticks(range(len(corr.index)))
        ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax2.set_yticklabels(corr.index)
        ax2.set_title("Correlation of Daily Returns")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
    else:
        st.info("Need at least two coins to compute a correlation matrix.")


# RECOMMENDATIONS
import numpy as np
import pandas as pd
import streamlit as st

st.subheader("🤖 Recommendations")

# Helper: simple RSI(14)
def rsi(series, period=14):
    s = series.dropna()
    if len(s) < period + 1:
        return pd.Series(index=s.index, dtype="float64")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    return rsi

def sma(series, n):
    return series.rolling(n, min_periods=n).mean()

def edu_signal(price_series):
    #Return (signal_label, score, details_dict) based on simple rules
    s = price_series.dropna()
    if len(s) < 21:  # need at least ~3 weeks of data for these toy rules
        return ("Insufficient", 0, {"Note": "Not enough data"})

    last = s.iloc[-1]
    sma7 = sma(s, 7).iloc[-1]
    sma21 = sma(s, 21).iloc[-1]
    sma50 = sma(s, 50).iloc[-1] if len(s) >= 50 else np.nan
    rsi14 = rsi(s, 14).iloc[-1] if len(s) >= 15 else np.nan
    ret_7d = s.pct_change(7).iloc[-1] if len(s) >= 8 else np.nan

    score = 0
    # Here we establish the rules for the recommendatoins
    # Rule 1: Trend (short > medium > long)
    if not np.isnan(sma50):
        if last > sma50 and sma7 > sma21 > sma50:
            score += 2
        elif last < sma50 and sma7 < sma21:
            score -= 2
    else:
        # fallback if <50 days
        if sma7 > sma21 and last > sma21:
            score += 1
        elif sma7 < sma21 and last < sma21:
            score -= 1

    # Rule 2: Momentum (7-day)
    if not np.isnan(ret_7d):
        if ret_7d > 0:
            score += 1
        elif ret_7d < 0:
            score -= 1

    # Rule 3: RSI extremes
    if not np.isnan(rsi14):
        if rsi14 < 30:
            score += 1   # potential rebound
        elif rsi14 > 70:
            score -= 1   # overbought

    if score >= 2:
        label = "Buy"
    elif score <= -2:
        label = "Sell"
    else:
        label = "Hold"

    details = {
        "Last": last,
        "SMA7": sma7,
        "SMA21": sma21,
        "SMA50": sma50,
        "RSI14": rsi14,
        "7d Return": ret_7d
    }
    return (label, score, details)

rows = []
for cid in prices.columns:
    sig, score, d = edu_signal(prices[cid])
    rows.append({
        "Coin": cid,
        "Signal": sig,
        "Score": score,
        "Last Price": d["Last"],
        "SMA7": d["SMA7"],
        "SMA21": d["SMA21"],
        "SMA50": d["SMA50"],
        "RSI14": d["RSI14"],
        "7d Return": d["7d Return"],
    })

reco_df = pd.DataFrame(rows)

# We apply the formatting used in top gainers table
def color_signal(val):
    if val == "Buy":
        return "color: #16a34a; font-weight: 700;"  # green
    if val == "Sell":
        return "color: #dc2626; font-weight: 700;"  # red
    return "color: #e5e7eb;"

styled = (reco_df
    .rename(columns={
        "Coin": "Coin",
        "Signal": "Recommendation",
        "Score": "Score",
        "Last Price": "Last Price",
        "SMA7": "SMA 7",
        "SMA21": "SMA 21",
        "SMA50": "SMA 50",
        "RSI14": "RSI 14",
        "7d Return": "7d Return",
    })
    .style
    .format({
        "Last Price": "{:,.4f}",
        "SMA 7": "{:,.4f}",
        "SMA 21": "{:,.4f}",
        "SMA 50": "{:,.4f}",
        "RSI 14": "{:.1f}",
        "7d Return": "{:+.2%}",
    })
    .hide(axis="index")
    .applymap(color_signal, subset=["Recommendation"])
)

st.dataframe(styled, use_container_width=True, height=260, hide_index=True)

# Download buttons
st.subheader("Downloads")
col_a, col_b, col_c = st.columns(3)
col_a.download_button("Download prices.csv", prices.to_csv().encode("utf-8"), "prices.csv", "text/csv")
col_b.download_button("Download returns.csv", returns.to_csv().encode("utf-8"), "returns.csv", "text/csv")
col_c.download_button("Download summary.csv", summary.to_csv().encode("utf-8"), "summary.csv", "text/csv")

st.caption("Source: CoinGecko public API")
