# UTS Machine Learning - Fraud Detection

## Data Diri

| Kategori | Detail |
| :--- | :--- |
| Nama | Rifqi Muhammad Harahap |
| Kelas | TK-46-GAB |
| NIM | [1103223138] |

---

## 1. Tujuan Repositori

Repository ini berisi implementasi *end-to-end machine learning pipeline* untuk mendeteksi kemungkinan transaksi online yang bersifat *fraudulent* (fraud detection). Proyek ini dibuat untuk memenuhi tugas UTS Machine Learning dengan memanfaatkan model **CatBoost Classifier**, *hyperparameter tuning* menggunakan **Optuna**, serta *experiment tracking* menggunakan **MLflow**.

## 2. Gambaran Singkat Proyek

Pada proyek ini dilakukan proses pembangunan *pipeline* machine learning secara lengkap mulai dari *data preprocessing*, *handling missing values*, *handling class imbalance*, *training model*, *hyperparameter tuning*, hingga evaluasi model. Dataset yang digunakan adalah `train_transaction.csv` dan `test_transaction.csv`. Proses evaluasi dilakukan secara internal menggunakan *train-validation-test split* dari dataset yang ada guna mengukur keandalan model secara objektif.

## 3. Dataset

Dataset utama yang digunakan pada proyek ini adalah data transaksi online. Dataset ini berisi informasi seperti nominal transaksi, jenis kartu, alamat, domain email, serta berbagai fitur numerik/kategorikal lainnya yang digunakan untuk memprediksi apakah suatu transaksi termasuk *fraud* (1) atau bukan (0).

*(Opsional: Tambahkan link dataset Google Drive dari dosenmu di sini jika diperbolehkan)*

## 4. Data Preprocessing

Tahapan preprocessing yang dilakukan dalam proyek ini meliputi:
1. Mengeliminasi kolom dengan tingkat *missing value* lebih dari 80%.
2. Mengisi nilai kosong (*missing values*) pada fitur numerik menggunakan **Median** (melalui teknik *Dictionary-based Imputation*).
3. Mengisi nilai kosong pada fitur kategorikal dengan label **'Unknown'** sekaligus mengonversinya menjadi format string agar kompatibel dengan CatBoost.
4. Menghapus kolom identifier unik yang tidak memiliki nilai prediktif seperti `TransactionID`.
5. Membagi dataset menggunakan metode *stratified split* untuk menjaga proporsi kelas, dengan skema pembagian:
   - **70%** Training Set
   - **15%** Validation Set
   - **15%** Test Set

## 5. Handling Class Imbalance

Kasus *fraud detection* identik dengan distribusi kelas yang sangat timpang. Pada proyek ini, *class imbalance* ditangani secara otomatis melalui parameter bawaan algoritma:
`auto_class_weights='Balanced'`
Parameter ini secara matematis memberikan bobot penalti yang lebih besar pada kelas minoritas (*fraud*) saat proses pembangunan pohon keputusan.

## 6. Model dan Hyperparameter Tuning

Model utama yang digunakan: **CatBoost Classifier**
CatBoost dipilih karena performanya yang sangat unggul dalam menangani data tabular, kemampuannya memproses fitur kategorikal secara natif, dan tahan terhadap *overfitting*. 

Untuk mencapai performa maksimal, dilakukan pencarian parameter terbaik (*hyperparameter tuning*) menggunakan **Optuna**. Parameter yang dieksplorasi meliputi `learning_rate`, `depth`, dan `l2_leaf_reg`. Pencarian dijalankan menggunakan kombinasi format data `Pool` CatBoost dan akselerator **GPU T4** di Google Colab. Jika terjadi *bottleneck* memori, proses dialihkan secara aman menggunakan CPU.

## 7. Evaluation Metrics

Evaluasi model dilakukan menggunakan **ROC-AUC Score** sebagai metrik utama karena paling relevan untuk kasus klasifikasi tidak seimbang. Turut disertakan *Classification Report* dan *Confusion Matrix* untuk analisis *True Positive* dan *False Positive*.

### ROC-AUC Score
*(Catatan: Ubah angka di bawah ini dengan hasil output akhir di Colab kamu)*

| Metric | Score |
| :--- | :--- |
| Validation ROC-AUC | 0.9994 |
| Test ROC-AUC | 0.98XX |

### Classification Report (Test Set)
*(Catatan: Ubah angka di bawah ini dengan hasil output akhir di Colab kamu)*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Non-Fraud (0) | 0.99 | 0.98 | 0.99 | ... |
| Fraud (1) | 0.85 | 0.91 | 0.88 | ... |

### Overall Metrics
*(Catatan: Sesuaikan angkanya dengan output Colab-mu)*

| Metric | Score |
| :--- | :--- |
| Accuracy | 0.98 |
| Macro Average F1-Score | 0.93 |
| Weighted Average F1-Score | 0.98 |

**Analisis Singkat:** Model memiliki performa yang sangat baik dalam membedakan transaksi *fraud* dan *non-fraud*. *Recall* yang tinggi menunjukkan bahwa model sukses menjaring mayoritas transaksi mencurigakan, sebuah karakteristik krusial dalam sistem keamanan perbankan.

## 8. Visualisasi

Proyek ini mencakup visualisasi komprehensif menggunakan Matplotlib dan Seaborn:
1. Perbandingan Kelas: Fraud vs Non-Fraud
2. Distribusi Nominal Transaksi (< 1000)
3. Top 20 Kolom dengan Missing Values Tertinggi
4. Matriks Evaluasi (Confusion Matrix)
5. Kurva ROC (Receiver Operating Characteristic)
6. 15 Atribut Paling Signifikan Mendeteksi Transaksi Fraud (Feature Importance)

## 9. MLflow Tracking

Sistem **MLflow** digunakan secara terintegrasi untuk mencatat riwayat eksperimen (nama eksperimen: `UTS_Fraud_Detection_Rifqi`). Semua kombinasi parameter terbaik dari Optuna, skor metrik validasi dan pengujian, hingga arsitektur file `.cbm` hasil *training* dicatat di dalam sistem ini. Hal ini menjamin *pipeline* bersifat transparan, dapat dilacak, dan mudah direproduksi (*reproducible*).

## 10. Repository Structure

```text
├── UTS_Machine_Learning.ipynb   # File utama Jupyter Notebook Google Colab
├── Model_Fraud_CatBoost_Rifqi.cbm # Model siap pakai hasil export CatBoost
└── README.md                    # Dokumentasi repositori
