# UTS Machine Learning - Customer Segmentation using K-Means Clustering

## Data Diri

| Kategori | Detail |
| :--- | :--- |
| Nama | RIFQI MUHAMMAD|
| Kelas | TK-46-GAB |
| NIM | [1103223138] |

---

## 1. Tujuan Repositori
Repository ini dirancang untuk mengimplementasikan *end-to-end unsupervised machine learning pipeline* guna melakukan segmentasi pelanggan (*customer segmentation*) berdasarkan perilaku penggunaan kartu kredit. Proyek ini dibuat untuk memenuhi tugas UTS Machine Learning dengan memanfaatkan metode **K-Means Clustering**, optimasi jumlah klaster menggunakan kombinasi **Elbow Method** dan **Silhouette Score**, serta reduksi dimensi menggunakan **Principal Component Analysis (PCA)** untuk visualisasi spasial 2D.

## 2. Gambaran Singkat Proyek
Proyek ini berfokus pada pengelompokan nasabah berdasarkan pola finansial mereka (seperti saldo, total pembelian, limit kredit, dan intensitas pembayaran). Alur kerja data mencakup *data profiling*, penanganan nilai yang hilang (*missing values*), mitigasi pencilan (*outlier handling*) menggunakan batas persentil, standardisasi skala fitur (*feature scaling*), pencarian parameter $K$ optimal, pemodelan klaster, hingga pembongkaran karakteristik profil tiap segmen (*cluster profiling*).

## 3. Dataset
Dataset yang dianalisis dalam proyek ini termuat dalam file `clusteringmidterm.csv`. Dataset ini menyimpan berbagai metrik historis aktivitas pengguna kartu kredit, seperti:
* `BALANCE`: Saldo yang tersisa di rekening untuk melakukan pembelian.
* `PURCHASES`: Total nominal pembelian yang dilakukan.
* `CREDIT_LIMIT`: Batas maksimum penggunaan kartu kredit pelanggan.
* `MINIMUM_PAYMENTS`: Minimum pembayaran yang wajib disetorkan, dan atribut transaksional lainnya.

## 4. Data Preprocessing & Profiling
Tahapan pembersihan data dilakukan secara ketat untuk menjamin validitas metrik jarak pada model:
1. **Inspeksi Struktur Data:** Mengidentifikasi tipe data dan sebaran data numerik melalui fungsi rangkuman statistik deskriptif.
2. **Imputasi Nilai Hilang (*Missing Values*):** Fitur `CREDIT_LIMIT` dan `MINIMUM_PAYMENTS` terdeteksi memiliki nilai kosong. Pembersihan dilakukan menggunakan teknik *Dictionary-based Imputation* dengan nilai **Median** dari masing-masing kolom untuk meminimalkan bias pencilan.
3. **Eliminasi Atribut Tidak Relevan:** Menghapus fitur `CUST_ID` menggunakan penanganan *defensive programming* (`errors='ignore'`) karena kolom teks tersebut bersifat unik-identifikasi dan tidak membawa makna matematis bagi algoritma pengklasteran.

## 5. Handling Outlier (Winsorization Capping)
Data transaksional finansial sangat rentan terhadap nilai ekstrem (*outliers*). Oleh karena itu, diterapkan metode *vectorized clipping* (Winsorization) batas atas pada **persentil ke-99**. Setiap nilai yang melampaui ambang batas persentil 99 dipangkas secara otomatis agar selaras dengan nilai batas tersebut. Hal ini krusial dilakukan agar titik pusat klaster (*centroid*) tidak bergeser atau terdistorsi oleh data anomali.

## 6. Standardisasi Fitur (Z-Score Normalization)
Algoritma K-Means berbasis pada kalkulasi jarak geometris (seperti *Euclidean Distance*). Agar kolom dengan nominal besar tidak mendominasi fitur berangka kecil, dilakukan standardisasi menggunakan **StandardScaler**. Seluruh fitur diubah skalanya secara simultan hingga memiliki nilai rata-rata (*mean*) = 0 dan variansi (*variance*) = 1.

## 7. Optimasi Penentuan Jumlah Klaster (K)
Pencarian jumlah kelompok ($K$) paling ideal diteliti secara objektif menggunakan kombinasi dua metode ilmiah:
* **Metode Elbow:** Menghitung nilai inersia atau *Within-Cluster Sum of Squares* (WCSS) dari rentang $K=1$ hingga $K=10$. Titik persendian kurva yang melandai secara konsisten (membentuk siku) dianalisis untuk menentukan retensi varians kelompok.
* **Silhouette Score:** Menguji ketajaman pemisahan antarklaster pada rentang $K=2$ hingga $K=10$. Berdasarkan pengujian, konfigurasi **$K=3$** mencatatkan rata-rata *Silhouette Score* tertinggi yaitu **0.2510**, membuktikan pemisahan spasial antarkelompok yang paling tegas dan padat.

## 8. Implementasi & Distribusi Klaster
Model K-Means final dieksekusi menggunakan konfigurasi $K=3$ dengan inisialisasi pusat data berbasis `k-means++`. Distribusi pembagian total nasabah pada ketiga klaster hasil pemodelan adalah sebagai berikut:

| ID Klaster | Jumlah Nasabah | Persentase Proporsi (%) |
| :---: | :---: | :---: |
| **0** | [Isi Jumlah Akhir Klaster 0] | [Isi % Klaster 0] |
| **1** | [Isi Jumlah Akhir Klaster 1] | [Isi % Klaster 1] |
| **2** | [Isi Jumlah Akhir Klaster 2] | [Isi % Klaster 2] |

## 9. Profiling & Interpretasi Klaster
Analisis karakteristik perilaku nasabah dilakukan dengan menelaah rata-rata nilai riil dari tiap klaster menggunakan visualisasi tabel ter-transpose dengan aksen gradasi warna (*heatmap style*). 

*(Opsional: Tuliskan interpretasi singkat 2-3 baris di sini setelah menjalankan kode Anda, misal: Klaster 0 = Pengguna Hemat, Klaster 1 = Pengguna Transaksional Tinggi/Sangat Konsumtif, Klaster 2 = Pengguna Saldo Menengah).*

## 10. Visualisasi Menggunakan PCA 2D
Mengingat dataset kartu kredit ini memiliki dimensi fitur yang tinggi, **Principal Component Analysis (PCA)** diterapkan untuk mereduksi matriks fitur terstandardisasi menjadi 2 Komponen Utama (`PC1` dan `PC2`). Proyeksi spasial 2D dalam bentuk *Scatter Plot* (menggunakan palet warna kontras tinggi dan efek transparansi data) memperlihatkan batasan pemisahan kelompok nasabah yang jelas dan terdistribusi dengan baik secara geometris.

## 11. Struktur Repositori
```text
├── Midterm_ML_2.ipynb         # Jupyter Notebook (Eksplorasi, Preprocessing, & Pemodelan)
├── clusteringmidterm.csv      # File dataset mentah riwayat finansial kartu kredit
└── README.md                  # Dokumentasi teknis proyek clustering
