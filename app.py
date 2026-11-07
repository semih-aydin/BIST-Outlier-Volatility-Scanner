import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="BIST Anomali ve RVOL Tarayici",
    layout="wide"
)

BIST30_TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS",
    "BRSAN.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS",
    "KONTR.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS", "OYAKC.IS",
    "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS"
]

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df.empty or len(df) <= 20:
            return None
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        print(f"Veri cekme hatasi [{ticker}]: {e}")
        return None

def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy()

    sma_vol = data['Volume'].rolling(window=window).mean()
    data['Volume_SMA20'] = sma_vol
    data['RVOL'] = np.where(sma_vol > 0, data['Volume'] / sma_vol, 1.0)
    data['Volatility'] = (data['High'] - data['Low']) / np.where(data['Close'] > 0, data['Close'], 1.0)
    data['Daily_Return'] = data['Close'].pct_change() * 100

    data.dropna(subset=['Volume_SMA20', 'Daily_Return'], inplace=True)
    return data

def detect_zscore_anomalies(df: pd.DataFrame, z_threshold: float = 2.5, rvol_threshold: float = 2.0) -> pd.DataFrame:
    data = df.copy()

    vol_std = data['RVOL'].std()
    ret_std = data['Daily_Return'].std()

    z_vol = (data['RVOL'] - data['RVOL'].mean()) / vol_std if vol_std > 0 else 0.0
    z_ret = (data['Daily_Return'] - data['Daily_Return'].mean()) / ret_std if ret_std > 0 else 0.0

    data['Z_Volume'] = z_vol
    data['Z_Return'] = z_ret
    data['Anomali_ZScore'] = (
        (np.abs(z_vol) >= z_threshold) |
        (np.abs(z_ret) >= z_threshold) |
        (data['RVOL'] >= rvol_threshold)
    )
    return data

def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    data = df.copy()
    feature_cols = ['RVOL', 'Volatility', 'Daily_Return']
    X = data[feature_cols].fillna(0).values

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    preds = model.fit_predict(X)
    data['Anomali_IsoForest'] = (preds == -1)
    data['Iso_Score'] = model.decision_function(X)
    return data

def build_figure(df: pd.DataFrame, ticker: str, mode: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} Fiyat ve Anomali Noktalari", "Hacim ve 20 Gunluk Ortalama")
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Fiyat",
            increasing_line_color="green",
            decreasing_line_color="red"
        ),
        row=1, col=1
    )

    if mode in ["Z-Score / RVOL", "Karsilastirma"] and 'Anomali_ZScore' in df:
        z_pts = df[df['Anomali_ZScore']]
        if not z_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=z_pts.index,
                    y=z_pts['High'] * 1.012,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=9, color='orange'),
                    name='Z-Score / RVOL',
                    customdata=np.stack((z_pts['RVOL'], z_pts['Daily_Return']), axis=-1),
                    hovertemplate="Tarih: %{x|%Y-%m-%d}<br>Fiyat: %{y:.2f} TL<br>RVOL: %{customdata[0]:.2f}x<br>Getiri: %{customdata[1]:.2f}%<extra></extra>"
                ),
                row=1, col=1
            )

    if mode in ["Isolation Forest", "Karsilastirma"] and 'Anomali_IsoForest' in df:
        iso_pts = df[df['Anomali_IsoForest']]
        if not iso_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=iso_pts.index,
                    y=iso_pts['Low'] * 0.988,
                    mode='markers',
                    marker=dict(symbol='circle', size=7, color='red'),
                    name='Isolation Forest',
                    customdata=np.stack((iso_pts['RVOL'], iso_pts['Volatility'] * 100, iso_pts['Daily_Return']), axis=-1),
                    hovertemplate="Tarih: %{x|%Y-%m-%d}<br>Fiyat: %{y:.2f} TL<br>RVOL: %{customdata[0]:.2f}x<br>Volatilite: %%%{customdata[1]:.2f}<br>Getiri: %{customdata[2]:.2f}%<extra></extra>"
                ),
                row=1, col=1
            )

    colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Hacim",
            marker_color=colors,
            opacity=0.7,
            customdata=df['RVOL'],
            hovertemplate="Hacim: %{y:,.0f}<br>RVOL: %{customdata:.2f}x<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Volume_SMA20'],
            name="SMA20",
            line=dict(color='blue', width=1.2)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=640,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def scan_bist30(z_thr: float, cont: float, rvol_thr: float) -> pd.DataFrame:
    rows = []
    pbar = st.progress(0, text="Taraniyor...")

    for i, sym in enumerate(BIST30_TICKERS):
        raw = fetch_stock_data(sym, period="6mo")
        if raw is not None and len(raw) >= 25:
            d = compute_features(raw)
            d = detect_zscore_anomalies(d, z_threshold=z_thr, rvol_threshold=rvol_thr)
            d = detect_isolation_forest(d, contamination=cont)

            last7 = d.tail(7)
            z_cnt = int(last7['Anomali_ZScore'].sum())
            iso_cnt = int(last7['Anomali_IsoForest'].sum())
            total = z_cnt + iso_cnt

            last_close = float(last7['Close'].iloc[-1])
            last_ret = float(last7['Daily_Return'].iloc[-1])
            last_rvol = float(last7['RVOL'].iloc[-1])

            if total > 0 or last_rvol >= rvol_thr:
                sub = last7[last7['Anomali_ZScore'] | last7['Anomali_IsoForest']]
                last_dt = sub.index[-1].strftime("%Y-%m-%d") if not sub.empty else "-"

                rows.append({
                    "Hisse": sym.replace(".IS", ""),
                    "Son Fiyat": last_close,
                    "Gunluk Getiri": last_ret,
                    "Son RVOL": last_rvol,
                    "Z-Score Alarmi": z_cnt,
                    "Iso Forest Alarmi": iso_cnt,
                    "Toplam Alarm": total,
                    "Son Alarm Tarihi": last_dt
                })

        pbar.progress((i + 1) / len(BIST30_TICKERS), text=f"Taraniyor: {sym}")

    pbar.empty()
    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    return res.sort_values(by=["Toplam Alarm", "Son RVOL"], ascending=[False, False]).reset_index(drop=True)

def main():
    st.title("BIST Anomali ve RVOL Tarayici")
    st.caption("BIST 30 Endeksi Istatistiki ve Cok Degiskenli Anomali Tespit Paneli")

    st.sidebar.subheader("Veri Secimi")
    selected_ticker = st.sidebar.selectbox("Hisse Kodu", BIST30_TICKERS, index=BIST30_TICKERS.index("THYAO.IS"))
    selected_period = st.sidebar.radio("Zaman Araligi", ["6mo", "1y"], format_func=lambda x: "6 Ay" if x == "6mo" else "1 Yil")

    st.sidebar.subheader("Esik Degerleri")
    rvol_thr = st.sidebar.slider("Minimum RVOL Kat Sayisi", 1.5, 4.0, 2.0, 0.1)
    z_thr = st.sidebar.slider("Z-Score Standart Sapma", 2.0, 3.5, 2.5, 0.1)
    contamination = st.sidebar.slider("Isolation Forest Kontaminasyon", 0.01, 0.10, 0.05, 0.01)

    display_mode = st.sidebar.selectbox("Model Filtresi", ["Karsilastirma", "Z-Score / RVOL", "Isolation Forest"])
    trigger_scan = st.sidebar.button("BIST 30 Toplu Tara", use_container_width=True)

    if trigger_scan:
        st.subheader("BIST 30 Son 7 Gun Alarm Ozeti")
        scan_df = scan_bist30(z_thr, contamination, rvol_thr)
        if not scan_df.empty:
            st.dataframe(
                scan_df,
                use_container_width=True,
                column_config={
                    "Son Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                    "Gunluk Getiri": st.column_config.NumberColumn(format="%.2f %%"),
                    "Son RVOL": st.column_config.NumberColumn(format="%.2f x"),
                }
            )

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=scan_df['Hisse'],
                    y=scan_df['Toplam Alarm'],
                    marker_color='royalblue'
                )
            ])
            fig_bar.update_layout(
                title="Son 7 Gunde Tespit Edilen Anomali Sayilari",
                xaxis_title="Hisse Kodu",
                yaxis_title="Toplam Alarm",
                template="plotly_dark",
                height=300,
                margin=dict(l=30, r=30, t=35, b=25)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Secilen parametrelere gore aktif alarm bulunamadi.")
        st.divider()

    raw_df = fetch_stock_data(selected_ticker, period=selected_period)
    if raw_df is None:
        st.error(f"{selected_ticker} verisi saglanamadi.")
        return

    feat_df = compute_features(raw_df)
    z_df = detect_zscore_anomalies(feat_df, z_threshold=z_thr, rvol_threshold=rvol_thr)
    final_df = detect_isolation_forest(z_df, contamination=contamination)

    last_close = float(final_df['Close'].iloc[-1])
    last_ret = float(final_df['Daily_Return'].iloc[-1])
    last_rvol = float(final_df['RVOL'].iloc[-1])
    z_total = int(final_df['Anomali_ZScore'].sum())
    iso_total = int(final_df['Anomali_IsoForest'].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Son Fiyat", f"{last_close:.2f} TL")
    c2.metric("Gunluk Getiri", f"{last_ret:+.2f}%")
    c3.metric("Son RVOL", f"{last_rvol:.2f}x")
    c4.metric("Z-Score / RVOL Alarm", f"{z_total}")
    c5.metric("Isolation Forest Alarm", f"{iso_total}")

    fig_main = build_figure(final_df, selected_ticker.replace('.IS', ''), display_mode)
    st.plotly_chart(fig_main, use_container_width=True)

    anomaly_rows = final_df[final_df['Anomali_ZScore'] | final_df['Anomali_IsoForest']].copy()
    st.subheader("Anomali Tespit Edilen Islem Gunleri")
    if not anomaly_rows.empty:
        table_df = pd.DataFrame({
            "Tarih": anomaly_rows.index.strftime("%Y-%m-%d"),
            "Kapanis": anomaly_rows['Close'],
            "Getiri": anomaly_rows['Daily_Return'],
            "RVOL": anomaly_rows['RVOL'],
            "Volatilite": anomaly_rows['Volatility'] * 100,
            "Z-Score": anomaly_rows['Anomali_ZScore'].map(lambda x: "Pozitif" if x else "-"),
            "Iso Forest": anomaly_rows['Anomali_IsoForest'].map(lambda x: "Pozitif" if x else "-")
        }).sort_values(by="Tarih", ascending=False).reset_index(drop=True)

        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "Kapanis": st.column_config.NumberColumn(format="%.2f TL"),
                "Getiri": st.column_config.NumberColumn(format="%.2f %%"),
                "RVOL": st.column_config.NumberColumn(format="%.2f x"),
                "Volatilite": st.column_config.NumberColumn(format="%.2f %%"),
            }
        )
    else:
        st.info("Secilen parametrelere gore anomali kaydi bulunmuyor.")

if __name__ == "__main__":
    main()
