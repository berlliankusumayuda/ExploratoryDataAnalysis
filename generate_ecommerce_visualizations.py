import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ------------------------------------------------------------
# 1. Membuat Dataset Sintetis E-Commerce
# ------------------------------------------------------------
np.random.seed(42)
n = 1000

product_categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
shipping_methods = ["Standard", "Express", "Same Day"]
customer_segments = ["Regular", "Premium", "VIP"]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]

data = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "customer_id": np.random.randint(1000, 2000, n),
    "product_id": np.random.randint(2000, 3000, n),
    "product_category": np.random.choice(product_categories, n, p=[0.3, 0.25, 0.15, 0.2, 0.1]),
    "product_price": np.round(np.random.normal(100, 20, n), 2),
    "quantity": np.random.randint(1, 8, n),
    "discount_percentage": np.random.choice([0, 5, 10, 15, 20], n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
    "customer_age": np.random.randint(18, 65, n),
    "customer_segment": np.random.choice(customer_segments, n, p=[0.6, 0.3, 0.1]),
    "payment_method": np.random.choice(payment_methods, n, p=[0.4, 0.25, 0.25, 0.1]),
    "shipping_method": np.random.choice(shipping_methods, n, p=[0.7, 0.25, 0.05]),
    "customer_satisfaction": np.random.choice([1,2,3,4,5], n, p=[0.1,0.15,0.25,0.3,0.2]),
    "order_date": pd.date_range("2024-01-01", periods=n, freq="D").to_list()
})

data["total_amount"] = np.round(
    data["product_price"] * data["quantity"] * (1 - data["discount_percentage"] / 100), 2
)
data["is_weekend"] = data["order_date"].dt.dayofweek >= 5
data["hour"] = np.random.randint(6, 22, n)

print("✅ Dataset sintetis e-commerce berhasil dibuat dengan", len(data), "baris.\n")

# ------------------------------------------------------------
# 2. Membuat Visualisasi Eksploratif (EDA)
# ------------------------------------------------------------
fig, axes = plt.subplots(6, 3, figsize=(24, 24))
axes = axes.flatten()

# Distribusi harga produk
sns.histplot(data["product_price"], kde=True, color="salmon", ax=axes[0])
axes[0].axvline(data["product_price"].mean(), color="red", linestyle="--", label=f"Mean: {data['product_price'].mean():.2f}")
axes[0].legend()
axes[0].set_title("Distribusi Harga Produk")

# Distribusi kategori produk
data["product_category"].value_counts().plot.pie(autopct="%.1f%%", ax=axes[1])
axes[1].set_ylabel("")
axes[1].set_title("Distribusi Kategori Produk")

# Total penjualan per kategori
sns.barplot(x="product_category", y="total_amount", data=data, ax=axes[2], estimator=np.sum)
axes[2].set_title("Total Penjualan per Kategori")

# Distribusi usia pelanggan
sns.histplot(data["customer_age"], bins=20, kde=True, color="green", ax=axes[3])
axes[3].axvline(data["customer_age"].mean(), color="red", linestyle="--", label=f"Mean: {data['customer_age'].mean():.1f}")
axes[3].legend()
axes[3].set_title("Distribusi Usia Pelanggan")

# Distribusi segmen pelanggan
sns.countplot(x="customer_segment", data=data, ax=axes[4], palette="pastel")
axes[4].set_title("Distribusi Segmen Pelanggan")

# Metode pembayaran
sns.countplot(y="payment_method", data=data, ax=axes[5])
axes[5].set_title("Metode Pembayaran yang Digunakan")

# Distribusi diskon
sns.countplot(x="discount_percentage", data=data, ax=axes[6], color="purple")
axes[6].set_title("Distribusi Diskon")

# Kepuasan pelanggan
sns.countplot(x="customer_satisfaction", data=data, ax=axes[7], color="orange")
axes[7].set_title("Distribusi Kepuasan Pelanggan")

# Jumlah item per transaksi
sns.histplot(data["quantity"], bins=7, color="saddlebrown", ax=axes[8])
axes[8].set_title("Distribusi Jumlah Item per Transaksi")

# Total nilai transaksi
sns.histplot(data["total_amount"], bins=30, kde=True, ax=axes[9])
axes[9].set_title("Distribusi Total Nilai Transaksi")

# Heatmap korelasi
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=axes[10])
axes[10].set_title("Heatmap Korelasi Variabel Numerik")

# Harga per kategori
sns.boxplot(x="product_category", y="product_price", data=data, ax=axes[11])
axes[11].set_title("Distribusi Harga per Kategori")

# Total transaksi vs kepuasan
sns.boxplot(x="customer_satisfaction", y="total_amount", data=data, ax=axes[12])
axes[12].set_title("Total Transaksi vs Kepuasan")

# Nilai transaksi per metode pengiriman
sns.barplot(x="shipping_method", y="total_amount", data=data, estimator=np.mean, ax=axes[13])
axes[13].set_title("Rata-rata Nilai Transaksi per Metode Pengiriman")

# Revenue per segmen
segment_revenue = data.groupby("customer_segment")["total_amount"].sum()
segment_revenue.plot.pie(autopct="%.1f%%", ax=axes[14])
axes[14].set_ylabel("")
axes[14].set_title("Kontribusi Revenue per Segmen Pelanggan")

# Tren penjualan harian
daily_sales = data.groupby("order_date")["total_amount"].sum()
axes[15].plot(daily_sales.index, daily_sales.values, color="green")
axes[15].set_title("Tren Penjualan Harian")
axes[15].set_xlabel("Tanggal")
axes[15].set_ylabel("Total Penjualan")

# Pola pesanan per jam
sns.lineplot(x="hour", y="quantity", data=data, marker="o", ax=axes[16], color="red")
axes[16].set_title("Pola Pesanan per Jam")

# Efektivitas diskon
discount_eff = data.groupby("discount_percentage")["quantity"].mean().reset_index()
sns.lineplot(x="discount_percentage", y="quantity", data=discount_eff, marker="o", ax=axes[17], color="purple")
axes[17].set_title("Efektivitas Diskon terhadap Kuantitas Pembelian")

# Layout dan simpan
plt.tight_layout()
plt.savefig("eda_visualizations.png", dpi=150)
plt.close()

print("📊 File visualisasi eksploratif disimpan sebagai 'eda_visualizations.png'")
