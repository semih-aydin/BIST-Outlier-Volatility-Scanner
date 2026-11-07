# BIST Outlier & RVOL Scanner

Borsa İstanbul (BIST 30) hisselerinde gerçekleşen olağan dışı hacim sıçramalarını (RVOL) ve sert fiyat hareketlerini klasik istatistik (Z-Score) ve gözetimsiz makine öğrenmesi (Isolation Forest) ile tespit edip görselleştiren analiz aracı.

## Özellikler
- **yfinance** ile günlük BIST 30 verisi
- **RVOL (Relative Volume):** Günlük hacim / 20 günlük ortalama hacim
- **Z-Score & RVOL Eşiği:** İstatistiki aykırı değer tespiti
- **Isolation Forest:** Çok boyutlu (RVOL, volatilite, getiri) makine öğrenmesi tespiti
- **Plotly:** Mum grafik ve hacim paneli üzerinde anomali işaretleri
- **BIST 30 Toplu Tarayıcı:** Son 7 günde en çok alarm üreten hisseler tablosu

## Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```
