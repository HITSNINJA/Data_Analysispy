import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memuat data (gunakan cache agar lebih cepat)
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders_dataset.csv")
    order_payments = pd.read_csv("data/order_payments_dataset.csv")
    customers = pd.read_csv("data/customers_dataset.csv")
    geolocation = pd.read_csv("data/geolocation_dataset.csv")

    # Konversi tanggal ke format datetime
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")

    # Menghitung waktu pengiriman (dalam hari)
    orders["delivery_time"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days

    return orders, order_payments, customers, geolocation 

# Memuat dataset
orders, order_payments, customers, geolocation = load_data()

# Sidebar navigasi
st.sidebar.header("📌 Pilih Halaman")
page = st.sidebar.radio("Navigasi", ["📊 EDA", "📈 Visualization & Explanatory", "🗺️ Geospatial Analysis"])

# Tambahkan filter untuk memilih tahun
st.sidebar.header("📆 Pilih Tahun")
orders["year"] = orders["order_purchase_timestamp"].dt.year  # Tambahkan kolom tahun
available_years = sorted(orders["year"].dropna().unique())   # Ambil daftar tahun yang tersedia
selected_year = st.sidebar.selectbox("Pilih Tahun", available_years, index=len(available_years)-1)

# Filter data berdasarkan tahun yang dipilih
filtered_orders = orders[orders["year"] == selected_year]

# ====================== BAGIAN EDA ====================== #
if page == "📊 EDA":
    st.title(f"📊 Exploratory Data Analysis ({selected_year})")

    eda_option = st.selectbox("Pilih EDA:", [
        "Distribusi Waktu Pengiriman",
        "Distribusi Status Pesanan",
        "Distribusi Customer Berdasarkan State",
    ])
    
    if eda_option == "Distribusi Waktu Pengiriman":
        st.subheader(f"Distribusi Waktu Pengiriman ({selected_year})")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_orders["delivery_time"].dropna(), bins=30, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Hari")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Distribusi Waktu Pengiriman (Hari)")
        st.pyplot(fig)

    elif eda_option == "Distribusi Status Pesanan":
        st.subheader(f"Distribusi Status Pesanan ({selected_year})")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=filtered_orders["order_status"], order=filtered_orders["order_status"].value_counts().index, palette="coolwarm", ax=ax)
        ax.set_xlabel("Jumlah Pesanan")
        ax.set_ylabel("Status Pesanan")
        ax.set_title("Distribusi Status Pesanan")
        st.pyplot(fig)

    elif eda_option == "Distribusi Customer Berdasarkan State":
        st.subheader(f"Distribusi Customer Berdasarkan State ({selected_year})")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=customers["customer_state"], order=customers["customer_state"].value_counts().index, ax=ax)
        ax.set_xlabel("Jumlah Customer")
        ax.set_ylabel("State")
        ax.set_title("Distribusi Customer Berdasarkan State")
        st.pyplot(fig)

# ====================== BAGIAN VISUALIZATION & EXPLANATORY ====================== #
elif page == "📈 Visualization & Explanatory":
    st.title(f"📈 Visualization & Explanatory Analysis ({selected_year})")

    viz_option = st.selectbox("Pilih Analisis:", ["Rata-rata Waktu Pengiriman per Wilayah", "Bagaimana Hari Libur dan Event Tahunan Mempengaruhi Transaksi"])

    if viz_option == "Rata-rata Waktu Pengiriman per Wilayah":
        st.subheader(f"Rata-rata Waktu Pengiriman per Wilayah ({selected_year})")

        # Gabungkan dengan data pelanggan
        orders_customers = filtered_orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")

        # Hitung rata-rata waktu pengiriman per wilayah
        avg_delivery_time = orders_customers.groupby("customer_state")["delivery_time"].mean().dropna().reset_index()

        # Visualisasi
        fig = px.bar(avg_delivery_time, x="customer_state", y="delivery_time",
                     labels={"customer_state": "Wilayah (State)", "delivery_time": "Rata-rata Waktu Pengiriman (Hari)"},
                     title="Rata-rata Waktu Pengiriman per Wilayah", color="delivery_time")
        st.plotly_chart(fig)

    elif viz_option == "Bagaimana Hari Libur dan Event Tahunan Mempengaruhi Transaksi":
        st.subheader(f"Tren Jumlah Pesanan dan Pengaruh Event Tahunan ({selected_year})")

        # Agregasi jumlah pesanan per hari
        orders_daily = filtered_orders.groupby(filtered_orders["order_purchase_timestamp"].dt.date).size().reset_index(name="jumlah_pesanan")
        orders_daily["order_purchase_timestamp"] = pd.to_datetime(orders_daily["order_purchase_timestamp"])

        # List hari libur besar di Brasil
        holiday_events = {
            "2017-01-01": "Tahun Baru",
            "2017-04-14": "Paskah",
            "2017-09-07": "Hari Kemerdekaan Brasil",
            "2017-11-24": "Black Friday",
            "2017-12-25": "Natal",
            "2018-01-01": "Tahun Baru",
            "2018-04-01": "Paskah",
            "2018-09-07": "Hari Kemerdekaan Brasil",
            "2018-12-25": "Natal",
        }

        # Filter hanya libur yang sesuai dengan tahun yang dipilih
        holiday_events = {date: event for date, event in holiday_events.items() if date.startswith(str(selected_year))}

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(orders_daily["order_purchase_timestamp"], orders_daily["jumlah_pesanan"], label="Jumlah Pesanan", color="blue")

        for date, event in holiday_events.items():
            ax.axvline(pd.to_datetime(date), color="red", linestyle="--", alpha=0.7)
            ax.text(pd.to_datetime(date), max(orders_daily["jumlah_pesanan"]) * 0.9, event, rotation=90, color="red", fontsize=10, verticalalignment='bottom')

        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Pesanan")
        ax.set_title("Tren Jumlah Pesanan dan Pengaruh Event Tahunan")
        ax.legend()
        st.pyplot(fig)

# ====================== BAGIAN GEOSPATIAL ANALYSIS ====================== #
elif page == "🗺️ Geospatial Analysis":
    st.title("🗺️ Geospatial Analysis")

    geo_sample = geolocation.sample(n=10000, random_state=42)

    fig = px.scatter_mapbox(geo_sample, lat="geolocation_lat", lon="geolocation_lng", color="geolocation_state",
                            mapbox_style="carto-positron", zoom=3, title="Sebaran Lokasi Pengguna E-Commerce")
    st.plotly_chart(fig)
