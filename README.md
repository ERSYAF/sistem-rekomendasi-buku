# Proyek Akhir : Membuat Model Sistem Rekomendasi

- **Nama:** [Era Syafina]
- **Email:** [erasyafina025@gmail.com]
- **ID Dicoding:** [Ersyafin]

---

# Project Overview

Membaca buku merupakan kegiatan penting dalam pengembangan wawasan, kemampuan berpikir kritis, serta kesehatan mental. Namun, di tengah kemajuan era digital, pengguna dihadapkan pada begitu banyak pilihan buku di platform daring, yang menimbulkan fenomena “kelebihan pilihan” (choice overload).

Untuk mengatasi tantangan ini, dibutuhkan sebuah sistem rekomendasi yang mampu membantu pembaca menemukan buku-buku yang sesuai dengan minat mereka berdasarkan riwayat interaksi, seperti penilaian (rating) dan ulasan yang telah diberikan.

# Mengapa Masalah Ini Perlu Diselesaikan?

- Banyak pengguna merasa kewalahan ketika harus memilih buku karena banyaknya opsi yang tersedia.
- Sistem rekomendasi berpotensi meningkatkan keterlibatan dan loyalitas pengguna terhadap platform digital buku.
- Penulis dan penerbit dapat menjangkau target audiens yang lebih tepat sasaran berdasarkan genre atau kategori bukunya.

# Pendekatan Penyelesaian

Solusi dirancang dengan membangun sistem rekomendasi menggunakan pendekatan berikut:

- **Content-Based Filtering**: Memanfaatkan kemiripan konten, khususnya judul buku, menggunakan metode TF-IDF dan Cosine Similarity.
- **User-Based Collaborative Filtering (Memory-Based)**: Menganalisis pola rating antar pengguna untuk menemukan kemiripan preferensi.
- **Model-Based Collaborative Filtering**: Menerapkan pembelajaran representasi (embedding) dengan TensorFlow/Keras untuk menangkap pola kompleks dalam data interaksi.

Dataset yang digunakan diambil dari [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) yang tersedia di Kaggle.

Referensi:

- Alkaff, M., Khatimi, H., & Eriadi, A. (2020). Sistem Rekomendasi Buku Menggunakan Weighted Tree Similarity dan Content Based Filtering. MATRIK J. Manajemen, Tek. Inform. dan Rekayasa Komput, 20(1), 193-202.
- Pratama, R. V., & Hasrullah, H. (2025). Pengembangan Sistem Rekomendasi Buku untuk Meningkatkan Minat Baca dengan Pendekatan Hybrid Filtering. Jurnal Inovasi Global, 3(1), 2182-2191.

---

# Business Understanding

Bagian ini menjelaskan proses identifikasi masalah yang menjadi dasar pengembangan sistem rekomendasi berbasis machine learning.

# Problem Statements:

- Banyaknya pilihan buku di platform daring membuat pengguna kesulitan menemukan bacaan yang sesuai dengan selera mereka.
- Pengguna baru atau yang jarang berinteraksi sering kali tidak memberikan cukup data (rating), sehingga sistem sulit memahami preferensi mereka.
- Platform perlu memberikan pengalaman yang lebih personal agar pengguna merasa lebih terlibat dan setia menggunakan layanan.

# Goals:

- Membantu pengguna menemukan buku yang cocok dengan minat mereka berdasarkan interaksi sebelumnya.
- Mengurangi efek kelebihan pilihan dengan menyediakan rekomendasi buku _top-N_ yang dipersonalisasi.
- Meningkatkan keterlibatan dan waktu yang dihabiskan pengguna di platform melalui sistem rekomendasi yang efektif.
- Mendorong pengguna untuk mengeksplorasi buku-buku dari genre yang mungkin mereka sukai tetapi belum mereka coba.

# Solution Statements:

- **Content-Based Filtering (CBF)**
  Menggunakan informasi dari buku seperti judul dan kesamaan antar judul dengan pendekatan TF-IDF dan Cosine Similarity.
  Kelebihan: Efektif untuk pengguna baru yang minim interaksi.
  Kekurangan: Cenderung merekomendasikan buku yang mirip saja, sehingga eksplorasi terbatas.

- **User-Based Collaborative Filtering (Memory-Based)**
  Memanfaatkan pola rating antar pengguna untuk memberikan rekomendasi berdasarkan kemiripan preferensi.
  Kelebihan: Dapat menangkap variasi selera pengguna.
  Kekurangan: Mengalami kesulitan dalam menghadapi pengguna baru (cold-start) dengan data interaksi yang minim.

- **Model-Based Collaborative Filtering (Deep Learning dengan Keras)**
  Menggunakan embedding untuk mempelajari preferensi laten pengguna dan buku, sehingga bisa memprediksi rating dengan lebih akurat.
  Kelebihan: Skalabilitas tinggi dan mampu menangkap pola kompleks.
  Kekurangan: Memerlukan data dalam jumlah besar dan waktu pelatihan lebih lama dibanding metode lain.

---

# Data Understanding

Pada bagian ini, kami menjelaskan jumlah data, kondisi data, dan informasi mengenai dataset yang digunakan. Dataset yang digunakan dalam proyek ini adalah [Book Recommendation Dataset - Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Dataset ini terdiri dari tiga file utama:

- **Books.csv**: berisi informasi lengkap tentang buku, mencakup ISBN, judul, penulis, tahun terbit, penerbit, dan link gambar sampul.
- **Users.csv**: memuat data pengguna, seperti ID, lokasi tempat tinggal, dan usia mereka.
- **Ratings.csv**: mencatat interaksi antara pengguna dan buku dalam bentuk rating yang diberikan.

---

## Informasi Jumlah Data

- Books.csv: 271.360 entri, 8 kolom
- Users.csv: 278.858 entri, 3 kolom
- Ratings.csv: 1.149.780 entri, 3 kolom

---

## Uraian Variabel

**Uraian Variabel** - `Books.csv`

- **ISBN**: Merupakan identifikasi unik untuk setiap buku. Bertipe data _object_ dengan jumlah data tidak kosong sebanyak 271.360.
- **Book-Title**: Menyimpan judul buku. Bertipe _object_, seluruh data terisi (271.360 entri).
- **Book-Author**: Berisi nama penulis buku. Tipe data _object_. Terdapat 271.358 data terisi dan 2 nilai yang hilang.
- **Year-Of-Publication**: Tahun buku diterbitkan. Bertipe _object_ dan tidak memiliki nilai kosong (271.360 entri).
- **Publisher**: Nama penerbit buku. Tipe _object_, dengan 271.358 entri terisi dan 2 data yang hilang.
- **Image-URL-S**: Link menuju gambar sampul kecil buku. Tipe data _object_ dengan 271.360 data lengkap.
- **Image-URL-M**: Menyimpan URL gambar sampul ukuran sedang. Bertipe _object_, seluruh entri terisi lengkap.
- **Image-URL-L**: URL gambar sampul besar buku. Tipe _object_, namun terdapat 3 data yang hilang (271.357 entri terisi).

**Uraian Variabel** – `Ratings.csv`

- **User-ID**: Merupakan identitas unik dari pengguna yang memberikan rating. Bertipe data _int64_ dengan total **1.149.780** entri yang lengkap (tidak ada nilai kosong).
- **ISBN**: Menunjukkan kode buku (ISBN) yang diberikan rating oleh pengguna. Bertipe _object_, dan seluruhnya terisi sebanyak **1.149.780** baris.
- **Book-Rating**: Nilai rating yang diberikan oleh pengguna terhadap buku, dengan skala **0 hingga 10**. Tipe data _int64_, tanpa missing value (**1.149.780 entri**).

**Uraian Variabel** – `Users.csv`

- **User-ID**: ID unik yang mengidentifikasi setiap pengguna. Bertipe data _int64_, dengan total **168.096** data yang lengkap.
- **Location**: Informasi lokasi pengguna yang mencakup **kota, provinsi, dan negara**. Bertipe _object_ dan tidak memiliki data yang hilang (**168.096 entri**).
- **Age**: Usia pengguna, disimpan dalam tipe data _float64_. Seluruhnya berjumlah **168.096**, meskipun kemungkinan terdapat nilai yang tidak valid seperti 0 atau lebih dari 100.

---

## Statistik Ringkasan

**Statistik Ringkasan Books**

**Beberapa insight penting dari dataset books:**

- Terdapat sebanyak **271.360 ISBN unik**, yang menunjukkan bahwa setiap buku memiliki kode ISBN yang berbeda.
- Jumlah **judul buku unik mencapai 242.135**, menandakan adanya beberapa duplikasi judul dengan edisi atau versi berbeda.
- **Agatha Christie** tercatat sebagai penulis paling produktif dalam dataset ini, dengan total **632 buku**.
- **Tahun terbit terbanyak adalah 2002**, di mana **17.627 buku** diterbitkan pada tahun tersebut.
- **Penerbit paling dominan adalah Harlequin**, dengan **7.535 buku** yang diterbitkan.
- Untuk kolom **Image-URL**, sebagian besar URL bersifat unik, namun terdapat sejumlah URL yang digunakan berulang kali untuk beberapa buku.

**Statistik Ringkasan Ratings**

**Insight Utama:**

- **Rata-rata rating:** 2.87
- **Standar deviasi:** 3.85
- **Rentang nilai rating:** dari **0** hingga **10**
- **Median (50%) rating:** 0 → Menunjukkan bahwa sebagian besar pengguna memberikan rating **kosong** atau **default**
- **ISBN terbanyak muncul:** `0971880107`, muncul sebanyak **2.502 kali**
- **Jumlah ISBN unik:** 340.556

**Statistik Ringkasan Users**

Beberapa insight penting:

- **Rata-rata usia pengguna** adalah sekitar **34,75 tahun**.
- **Usia terendah** tercatat **0 tahun**, sementara **usia tertinggi** mencapai **244 tahun**, yang kemungkinan besar merupakan data tidak valid.
- Lokasi yang paling sering dicantumkan pengguna adalah **"london, england, united kingdom"**, dengan total **2.506 pengguna**.
- Total terdapat **57.339 lokasi unik** dalam dataset.

---

## Informasi Kondisi Data

- Dataset Books:
  - Memiliki beberapa missing value di kolom `Book-Author`, `Publisher`, dan `Image-URL-L`.
- Dataset Users:
  - Terdapat sekitar 110.762 missing value pada kolom `Age`.
  - Ditemukan usia tidak valid, seperti 0 dan lebih dari 100 tahun.
- Dataset Ratings:
  - Tidak terdapat missing value.
  - Banyak pengguna memberikan rating 0.

---

## Cek Duplikat Data

Pada tahap ini, kami memeriksa apakah terdapat data duplikat pada masing-masing dataset.

Hasil pemeriksaan:

- Dataset Books: 0 duplikat
- Dataset Users: 0 duplikat
- Dataset Ratings: 0 duplikat

Tidak ditemukan data duplikat, sehingga tidak diperlukan tindakan pembersihan untuk duplikasi.

--- 

## Exploratory Data Analysis (EDA)

1. **10 Buku Rating Terbanyak**

Berikut adalah tabel yang merangkum 10 buku dengan jumlah rating terbanyak di dataset:

| No | ISBN       | Judul Buku                                        | Jumlah Rating |
| -- | ---------- | ------------------------------------------------- | ------------- |
| 1  | 0971880107 | *Wild Animus*                                     | 2502          |
| 2  | 0316666343 | *The Lovely Bones: A Novel*                       | 1295          |
| 3  | 0385504209 | *The Da Vinci Code*                               | 883           |
| 4  | 0060928336 | *Divine Secrets of the Ya-Ya Sisterhood: A Novel* | 732           |
| 5  | 0312195516 | *The Red Tent (Bestselling Backlist)*             | 723           |
| 6  | 044023722X | *A Painted House*                                 | 647           |
| 7  | 0679781587 | *Tidak tersedia (missing title)*                  | 639           |
| 8  | 0142001740 | *The Secret Life of Bees*                         | 615           |
| 9  | 067976402X | *Snow Falling on Cedars*                          | 614           |
| 10 | 0671027360 | *Angels & Demons*                                 | 586           |


![10 Buku Rating Terbanyak](gambar/Buku%20Rating%20Terbanyak.png)


**Insight**:

- Buku Wild Animus jauh lebih sering dirating dibandingkan buku lain, kemungkinan besar karena faktor popularitas atau faktor marketing tertentu.
- Terdapat ISBN yang tidak memiliki `Book-Title` (NaN), perlu perhatian lebih dalam tahap berikutnya.

2. **Distribusi Nilai Rating**

Distribusi nilai rating menunjukkan bahwa sebagian besar rating yang diberikan adalah 0.
Rating tinggi seperti 8–10 juga cukup banyak, menunjukkan adanya bias positif pengguna terhadap buku favorit mereka.

![Distribusi Nilai Rating Buku](gambar/Distribusi%20Nilai%20Rating%20Buku.png)

**Insight**:
- Dominasi rating 0 mengisyaratkan adanya banyak data yang bersifat kosong atau merupakan nilai default, yang tidak merepresentasikan preferensi pengguna secara akurat.
- Hal ini perlu mendapat perhatian khusus dalam tahap pemodelan sistem rekomendasi, karena bisa menyebabkan bias atau distorsi pada hasil prediksi rekomendasi.

3. **Distribusi Usia Pengguna**

Visualisasi data menunjukkan bahwa sebagian besar pengguna berada dalam rentang usia 20 hingga 50 tahun. Namun, terdapat kejanggalan data pada usia 0 tahun dan di atas 100 tahun, yang mengindikasikan keberadaan outlier.

![Distribusi Usia Pengguna](gambar/Distribusi%20Usia%20Pengguna.png)

**Insight**:
- Kelompok usia terbanyak berasal dari rentang 30 hingga 40 tahun.
- Nilai usia ekstrem seperti 0 dan lebih dari 100 tahun perlu dipertimbangkan untuk dibersihkan.

---

# Data Preparation

Pada bagian ini, kami menerapkan dan menyebutkan teknik data preparation yang dilakukan.
Teknik yang digunakan pada notebook dan laporan disusun secara berurutan sesuai proses.

---

### 1. Penanganan Nilai Hilang pada Kolom Age

**Teknik yang digunakan**:
- Menghilangkan missing value pada kolom Age dengan menggunakan fungsi dropna(subset=['Age']).

**Proses dan Alasan**:
Setelah pemeriksaan, ditemukan sekitar 110.762 data pengguna tidak memiliki nilai usia. Karena proporsi nilai hilang ini cukup besar (sekitar 40%), kami memutuskan untuk menghapus baris data pengguna yang kolom Age-nya kosong agar proses pemodelan tidak bias.

**Hasil**: Awalnya, dataset Users memiliki 278.858 baris data. Setelah proses penghapusan baris yang memiliki nilai kosong pada kolom Age, jumlah data berkurang menjadi 168.096 entri dengan 3 kolom.

---

### 2. Penanganan Outlier pada Kolom Age

**Teknik yang digunakan**:
- Memfilter data dengan kondisi usia >= 5 dan <= 100.

**Proses dan Alasan**:
Dari eksplorasi data awal, ditemukan beberapa usia yang tidak masuk akal, seperti 0 tahun dan 244 tahun. Untuk meningkatkan kualitas data, kami melakukan filtering sehingga hanya menyertakan pengguna dengan usia di antara 5 hingga 100 tahun.

**Hasil**: Setelah menghapus nilai kosong di kolom Age, dataset Users memiliki 168.096 entri. Setelah proses pembersihan outlier dengan menyaring usia antara 5 hingga 100 tahun, jumlah data berkurang menjadi 166.848 entri dengan 3 kolom.


---

### 3.  Penyaringan (Filtering) Data Ratings

**Teknik yang digunakan**:
- Menghapus baris data dengan kondisi Book-Rating == 0 menggunakan teknik filtering.

**Proses dan Alasan**:
- Pada tahap ini, kami memfilter dataset Ratings dengan menghapus semua data yang memiliki nilai rating 0. Rating 0 menunjukkan bahwa pengguna tidak secara aktif memberikan penilaian terhadap buku tersebut, sehingga data ini tidak memberikan informasi preferensi yang berarti untuk sistem rekomendasi.

**Hasil**: Dataset Ratings awalnya memiliki 1.149.780 entri. Setelah menghapus data dengan nilai rating 0, jumlah entri berkurang menjadi 433.671 entri dengan 3 kolom.


---

### 4. Filtering Pengguna (users) dan Buku Berdasarkan Aktivitas Minimum

**Teknik yang digunakan**:
- Penyaringan Pengguna: Mempertahankan hanya pengguna yang telah memberikan minimal 3 rating.
- Penyaringan Buku: Mempertahankan hanya buku yang menerima minimal 3 rating.
- Melakukan penyaringan berdasarkan jumlah interaksi menggunakan fungsi value_counts() dan isin().

**Proses dan Alasan**:
- Memastikan hanya pengguna yang aktif memberikan rating yang diproses.
- Memastikan buku yang dianalisis memiliki cukup banyak penilaian untuk memberikan informasi yang valid bagi sistem rekomendasi.

**Hasil**: Setelah menghapus rating dengan nilai 0, dataset Ratings memiliki 433.671 entri. Setelah dilakukan filtering untuk mempertahankan pengguna dan buku yang memiliki minimal 3 rating, jumlah entri berkurang menjadi 203.851 dengan 3 kolom.

---

### 5. Konversi Kolom Menjadi List

**Teknik yang digunakan**:
- Mengubah kolom-kolom utama dari dataset Ratings menjadi format list menggunakan fungsi .tolist()

**Proses dan Alasan**:
- Kolom yang dikonversi meliputi: User-ID menjadi list user_id, ISBN menjadi list isbn, Book-Rating menjadi list book_rating

---

### 6. Membuat DataFrame Bersih untuk Modeling

**Teknik yang digunakan**:
- Membuat sebuah DataFrame baru bernama ratings_clean yang berisi tiga kolom utama hasil dari konversi list: user_id : ID pengguna, isbn : ISBN buku, book_rating : Nilai rating yang diberikan pengguna pada buku

**Proses dan Alasan**:
Pembuatan DataFrame ini bertujuan untuk memudahkan proses pemodelan sistem rekomendasi pada tahap berikutnya, baik menggunakan metode Collaborative Filtering maupun Content-Based Filtering.

**Hasil**: DataFrame ratings_clean yang dibuat memiliki 203.851 baris dan 3 kolom.

---

### 7. Membuat Matriks User-Item untuk Collaborative Filtering

**Teknik yang digunakan**:
- Pada tahap ini, kami menyusun User-Item Matrix dari dataset ratings_clean.

Metode yang digunakan:
- Membuat pivot table dengan fungsi pivot_table dari pandas.

**Proses dan Alasan**:
- **Baris**: Merepresentasikan setiap pengguna unik (`user_id`).
- **Kolom**: Merepresentasikan setiap buku unik berdasarkan kode `ISBN`.
- **Isi sel**: Nilai `book_rating` yang diberikan pengguna terhadap buku tertentu.

**Hasil**:
- Matriks User-Item yang terbentuk memiliki ukuran (20.908, 25.790).
- Terdapat banyak nilai missing (NaN) karena sebagian besar pengguna hanya memberikan rating pada sebagian kecil dari total buku yang tersedia.

---

### 8. Membuat Daftar User-ID dan ISBN Unik

**Teknik yang digunakan**:
- Mengambil semua nilai unik dari `user_id` dan `isbn` di `ratings_clean` menggunakan `.unique()` dan mengubahnya ke list.

**Proses dan Alasan**:
- Daftar ini nantinya akan digunakan untuk proses encoding data.

**Hasil**:
- Terdapat 20.908 User-ID unik dan 25.790 ISBN unik dalam dataset setelah pembersihan.

---

### 9. Encoding User-ID dan ISBN ke Bentuk Integer dan Memetakan (Mapping) User dan ISBN yang Telah Di-encode ke DataFrame

**Teknik yang digunakan**:
- Menggunakan `LabelEncoder` untuk mengubah `user_id` dan `isbn` dari string menjadi angka integer.

**Proses dan Alasan**:
- Model berbasis embedding di TensorFlow hanya menerima input numerik, sehingga encoding perlu dilakukan.
- Hasil encoding disimpan dalam kolom `user` dan `book`.

**Hasil**:
- Setiap `user_id` dan `isbn` berhasil dipetakan ke nilai integer unik.
- Contoh mapping:
  
| user_id | isbn       | book_rating | user | book |
|---------|------------|-------------|------|------|
| 276747  | 0060517794 | 9           | 0    | 0    |
| 276747  | 0671537458 | 9           | 0    | 1    |
| 276747  | 0679776818 | 8           | 0    | 2    |
| 276762  | 0380711524 | 5           | 1    | 3    |
| 276772  | 0553572369 | 7           | 2    | 4    |

---

### 10. Konversi Nilai Rating ke Tipe Data float32

**Teknik yang digunakan**:
- Menggunakan `.astype('float32')` untuk mengubah nilai `book_rating`.

**Proses dan Alasan**:
- TensorFlow membutuhkan input numerik bertipe float untuk training model neural network.

**Hasil**:
- Kolom `book_rating` pada dataset `ratings_clean` berhasil dikonversi ke format `float32`.

---

### 11. Check Jumlah User dan Buku Setelah Encoding

**Teknik yang digunakan**:
- Menggunakan fungsi `nunique()` untuk menghitung jumlah user dan buku unik setelah encoding.

**Proses dan Alasan**:
- Tahapan ini digunakan untuk verifikasi bahwa proses encoding berhasil dilakukan secara konsisten dan tidak ada data yang hilang.
- Menjadi acuan juga untuk menentukan input dimension pada embedding layer model deep learning.

**Hasil**: Setelah proses encoding, ditemukan bahwa terdapat 20.908 user unik dan 25.790 buku unik dalam dataset.

---

### 12. Membuat TF-IDF Matrix dari Judul Buku

**1. Filtering ISBN Buku:**
Dataset `books` difilter agar hanya menyertakan ISBN yang terdapat dalam `ratings_clean`.
* Tujuan: Menyelaraskan data buku dengan buku-buku yang telah dirating secara aktif.

**2. Menangani Missing Value:**
Kolom `Book-Title` yang kosong diisi dengan string kosong (`''`) untuk mencegah error saat proses vektorisasi teks.
* Teknik: `fillna('')`

**3. TF-IDF Vectorization:**
Digunakan **`TfidfVectorizer`** dari `scikit-learn` dengan parameter `stop_words='english'`.
* Tujuan: Mengubah teks judul buku menjadi representasi numerik berdasarkan pentingnya kata (frekuensi term invers dokumen).

**4. Fit dan Transform:**
TF-IDF Vectorizer di-*fit* dan diterapkan (`transform`) pada kolom `Book-Title`.
* Hasil: TF-IDF Matrix yang merepresentasikan setiap judul buku dalam bentuk vektor.

**Hasil:**

* **Ukuran Matrix:** `(24.253, 16.052)`

  * **24.253** baris = jumlah judul buku yang dianalisis.
  * **16.052** kolom = jumlah kata unik (fitur) dari semua judul buku setelah stopwords dihapus.
* **Makna Matrix:**

  * Setiap baris merepresentasikan **sebuah buku** dalam bentuk vektor fitur.
  * Setiap kolom merepresentasikan **kata unik** yang digunakan dalam judul-judul buku.
  * Nilai pada matrix menunjukkan **seberapa penting** kata tersebut bagi suatu judul (berdasarkan frekuensi term dan seberapa umum kata tersebut dalam seluruh koleksi judul).

---

### 13. Membagi (Split) Dataset untuk Pelatihan dan Validasi

**Teknik yang digunakan**:
- Menggunakan `train_test_split` dengan rasio 80:20.

**Proses dan Alasan**:
- Dataset diacak dan dibagi menjadi dua bagian:
  - **Data pelatihan (training)**: 80%
  - **Data validasi (validation)**: 20%
  
- **Fitur (X)**: Berupa pasangan `user` dan `book` yang merepresentasikan interaksi pengguna dengan buku.
- **Target (y)**: Nilai `book_rating` yang telah dinormalisasi ke dalam rentang **0 hingga 1**.

**Tujuan pemisahan data**:
Pemisahan ini dilakukan untuk menghindari tumpang tindih antara data pelatihan dan data validasi, sehingga model:
- Dapat **belajar dengan optimal** dari data pelatihan.
- Dapat **dilakukan evaluasi secara akurat** menggunakan data validasi.

**Hasil**:
- Bentuk data pelatihan: x_train (163.080, 2), y_train (163.080,)
- Bentuk data validasi: x_val (40.771, 2), y_val (40.771,)

---

# Modeling

Pada bagian ini, kami membangun tiga pendekatan sistem rekomendasi untuk menyelesaikan permasalahan prediksi buku yang relevan untuk pengguna, yaitu:

1. Content-Based Filtering
2. Collaborative Filtering (Memory-Based)
3. Collaborative Filtering (Model-Based menggunakan TensorFlow)

---

### Content-Based Filtering

Pendekatan ini merekomendasikan buku berdasarkan kemiripan konten, khususnya pada judul buku. Berikut adalah tahapan modeling yang dilakukan:

---

### 1. Cosine Similarity Antar Judul Buku

Setelah representasi teks judul buku dikonversi ke bentuk vektor melalui proses **TF-IDF**, langkah selanjutnya adalah menghitung **kemiripan antar judul buku**. Dalam penelitian ini, digunakan teknik **Cosine Similarity** untuk mengukur sejauh mana dua judul buku memiliki kemiripan makna berdasarkan distribusi kata-kata dalam vektor TF-IDF.

---

#### Teknik yang Digunakan: Cosine Similarity

**Cosine Similarity** merupakan metode populer untuk mengukur kemiripan antara dua vektor dalam ruang berdimensi tinggi. Alih-alih menggunakan jarak Euclidean, Cosine Similarity menghitung **nilai cosinus dari sudut** antara dua vektor.

* **Interpretasi Nilai Cosine Similarity**:

  * **1.0** → Vektor sejajar sempurna (judul buku sangat mirip atau identik secara teks).
  * **0.0** → Vektor saling tegak lurus (tidak ada kesamaan sama sekali).
  * Nilai antara 0 dan 1 menggambarkan derajat kemiripan relatif antar judul.

* **Mengapa Cosine Similarity?**
  Karena Cosine Similarity tidak dipengaruhi oleh panjang vektor (frekuensi absolut), tetapi hanya memperhatikan **arah vektor**, sehingga cocok untuk analisis teks yang menggunakan representasi seperti TF-IDF.

---

#### Proses Perhitungan

1. **Input**:

   * Matriks TF-IDF hasil vektorisasi judul buku dengan dimensi `(24.253 x 16.052)`, di mana:

     * 24.253 = jumlah judul buku unik.
     * 16.052 = jumlah kata unik yang digunakan sebagai fitur (setelah filtering dan stopword removal).

2. **Fungsi yang Digunakan**:

   * `cosine_similarity` dari modul `sklearn.metrics.pairwise`.

3. **Output**:

   * Sebuah **matriks similarity** berdimensi `(24.253 x 24.253)`, di mana setiap entri `[i][j]` menunjukkan skor kemiripan antara buku ke-*i* dan buku ke-*j* berdasarkan judulnya.

**Hasil:**
Matriks cosine similarity yang diperoleh memiliki dimensi (24.253, 24.253).
Ini menunjukkan bahwa sebanyak 24.253 judul buku saling dibandingkan untuk menilai seberapa mirip masing-masing judul satu dengan yang lain.

---
### 2. Cosine Similarity Antar Pengguna

Tahap ini bertujuan untuk mengukur sejauh mana dua pengguna memiliki preferensi yang serupa terhadap buku-buku yang mereka rating. Pendekatan ini umum digunakan dalam sistem rekomendasi **collaborative filtering berbasis user-based**, di mana pengguna akan direkomendasikan item berdasarkan kesamaan preferensinya dengan pengguna lain.

---

#### Teknik yang Digunakan: Cosine Similarity

**Cosine Similarity** digunakan untuk menghitung **kemiripan antara dua vektor pengguna** dalam sebuah **User-Item Matrix**. Setiap baris pada matriks merepresentasikan seorang pengguna, dan setiap kolom merepresentasikan sebuah buku. Nilai pada sel matriks menunjukkan rating yang diberikan oleh pengguna terhadap buku tersebut.

* **Interpretasi Nilai Cosine Similarity**:

  * **1.0** → Kedua pengguna memiliki pola rating yang sangat mirip.
  * **0.0** → Kedua pengguna tidak memiliki kesamaan preferensi sama sekali.
  * Nilai antara 0 dan 1 menunjukkan tingkat kemiripan secara bertahap.

* **Mengapa Cosine Similarity?**
  Cosine similarity tidak mempertimbangkan perbedaan skala (misalnya, pengguna yang cenderung memberi rating tinggi vs. rendah), tetapi lebih fokus pada **pola** rating relatif terhadap item yang sama, sehingga cocok digunakan untuk analisis perilaku pengguna.

---

#### Proses Perhitungan

1. **Membangun User-Item Matrix**:

   * Matriks dibuat dengan baris sebagai pengguna dan kolom sebagai buku.
   * Nilai dalam matriks adalah rating yang diberikan oleh pengguna terhadap buku.
   * **Missing values** (jika pengguna belum merating buku tertentu) diisi dengan **0**, karena 0 dianggap mewakili tidak adanya interaksi.

2. **Fungsi yang Digunakan**:

   * `cosine_similarity` dari `sklearn.metrics.pairwise` digunakan untuk menghitung kemiripan antara setiap pasangan pengguna.

3. **Output**:

   * Sebuah **matriks kemiripan pengguna** (misalnya ukuran `N x N`, di mana N adalah jumlah pengguna), dengan setiap entri `[i][j]` menunjukkan skor kemiripan antara pengguna ke-*i* dan ke-*j* berdasarkan pola rating mereka.

**Hasil**: Matriks cosine similarity yang diperoleh memiliki dimensi (20.908, 20.908).
Ini menunjukkan bahwa kemiripan dihitung antar 20.908 pengguna dengan membandingkan setiap pasangan pengguna satu sama lain.
teks yang dimiringkan

---
### 3. Membuat Fungsi Rekomendasi Berdasarkan Judul Buku
Pada tahap ini, kami mengembangkan fungsi rekomendasi yang bertujuan untuk menyajikan Top-N buku yang memiliki kemiripan tinggi berdasarkan judul buku.

**Langkah-langkah yang diterapkan meliputi:**
* Memanfaatkan matriks Cosine Similarity yang sudah tersedia.
* Mencari indeks buku berdasarkan judul yang diinput.
* Mengurutkan skor kemiripan dan memilih Top-N buku dengan nilai tertinggi.

Fungsi ini bertujuan memberikan rekomendasi buku yang serupa sesuai dengan preferensi pengguna terhadap sebuah buku tertentu.

**Contoh Output: Rekomendasi Top-5 Buku**  
*(Menggunakan Content-Based Filtering)*

Untuk menguji fungsi rekomendasi berbasis *Content-Based Filtering* yang telah dibuat, kami menggunakan judul buku berikut sebagai input:

- *Buku Input*: **Harry Potter and the Chamber of Secrets (Book 2)**

Sistem kemudian mencari 5 buku dengan tingkat kemiripan tertinggi berdasarkan analisis *TF-IDF* dan *Cosine Similarity* antar judul buku.

---

**Top-5 Rekomendasi**:

| No  | Book-Title                                                | Book-Author   |
| :-- | :-------------------------------------------------------- | :------------ |
| 1   | Harry Potter and the Chamber of Secrets (Book 2)          | J. K. Rowling |
| 2   | Harry Potter and the Chamber of Secrets (Book 2)          | J. K. Rowling |
| 3   | Harry Potter and the Chamber of Secrets (Harry Potter)    | J. K. Rowling |
| 4   | Harry Potter and the Chamber of Secrets Postcard Book     | J. K. Rowling |
| 5   | Harry Potter and the Chamber of Secrets (Book 2 Audio CD) | J. K. Rowling |


**Interpretasi Hasil**:

Seluruh buku yang direkomendasikan merupakan variasi dari *"Harry Potter and the Chamber of Secrets"*, seperti edisi berbeda atau format yang beragam (buku cetak, audio CD, kartu pos).  
Hal ini menunjukkan bahwa metode *Content-Based Filtering* yang menggunakan kemiripan judul cukup efektif dalam menangkap kesamaan isi atau tema antar buku.

---

### Model Development dengan Collaborative Filtering

### Pengembangan Fungsi Rekomendasi Berbasis Kemiripan Pengguna

Pada tahap ini, dikembangkan sebuah fungsi rekomendasi yang bertujuan untuk memberikan saran buku secara personal kepada pengguna berdasarkan **kesamaan preferensi** dengan pengguna lain. Pendekatan ini dikenal sebagai **User-Based Collaborative Filtering (UBCF)**, di mana sistem belajar dari pola rating pengguna lain yang serupa untuk memprediksi preferensi pengguna target.

---

#### Metode yang Digunakan: User-Based Collaborative Filtering

**User-Based Collaborative Filtering** merupakan pendekatan dalam sistem rekomendasi yang menggunakan prinsip bahwa *"pengguna yang memiliki preferensi serupa di masa lalu kemungkinan besar akan menyukai item yang sama di masa depan."* Dalam konteks ini, kesamaan preferensi antar pengguna dihitung menggunakan **Cosine Similarity** berdasarkan data rating buku.

* **Prinsip Dasar:**

  * Cari pengguna lain yang mirip dengan pengguna target.
  * Lihat buku-buku yang disukai oleh pengguna-pengguna tersebut.
  * Rekomendasikan buku-buku tersebut kepada pengguna target, terutama yang belum pernah ia nilai sebelumnya.

---

#### Langkah-Langkah Fungsi Rekomendasi

1. **Menghitung Skor Kemiripan Pengguna:**

   * Menggunakan **Cosine Similarity** untuk menghitung tingkat kemiripan antara pengguna target dan seluruh pengguna lainnya berdasarkan **User-Item Matrix**.

2. **Memilih K-Nearest Neighbors:**

   * Sistem memilih **5 pengguna paling mirip** (selain pengguna itu sendiri) dengan skor cosine tertinggi sebagai *neighborhood*.

3. **Menghitung Rata-Rata Rating Buku:**

   * Untuk setiap buku yang telah dirating oleh pengguna-pengguna serupa, sistem menghitung **rata-rata rating** sebagai prediksi minat terhadap buku tersebut.

4. **Menyaring Buku Baru bagi Pengguna Target:**

   * Sistem mengecek buku-buku yang belum pernah dirating oleh pengguna target agar hanya merekomendasikan buku baru baginya.

5. **Memilih Top-N Rekomendasi:**

   * Dari daftar buku yang telah disaring, sistem memilih sejumlah **Top-N buku dengan rata-rata rating tertinggi** sebagai hasil rekomendasi.

6. **Menggabungkan Data Tambahan Buku:**

   * Untuk memperkaya hasil rekomendasi, data buku digabungkan dengan metadata seperti **judul buku** (`Book-Title`) dan **penulis** (`Book-Author`) dari dataset `books`.

7. **Menambahkan Nilai Rata-Rata Rating:**

   * Disisipkan informasi tambahan berupa **rata-rata rating prediksi** untuk setiap buku, sehingga pengguna dapat mempertimbangkan seberapa tinggi tingkat kesukaan pengguna serupa terhadap buku tersebut.

---

#### Tujuan

Fungsi ini bertujuan untuk memberikan rekomendasi buku yang **relevan dan personal** kepada pengguna berdasarkan perilaku pengguna lain yang memiliki pola preferensi serupa. Dengan pendekatan ini, sistem diharapkan dapat meningkatkan kepuasan pengguna dalam menemukan buku-buku yang menarik, meskipun pengguna tersebut belum memiliki banyak riwayat interaksi.

---

#### Rekomendasi Berdasarkan User Similarity

**Contoh Input User:**

> **User ID:** 8

**Top-3 Rekomendasi Buku:**

| No  |    ISBN    | Book-Title            | Book-Author    | Average-Rating |
| :-: | :--------: | :-------------------- | :------------- | :------------: |
|  1  | 0446310786 | To Kill a Mockingbird | Harper Lee     |      10.0      |
|  2  | 0684874350 | ANGELA'S ASHES        | Frank McCourt  |      10.0      |
|  3  | 0440212561 | Outlander             | Diana Gabaldon |      10.0      |

**Catatan:**  
Hanya 3 rekomendasi yang berhasil ditampilkan karena hanya 3 ISBN yang cocok dengan data buku yang tersedia pada `books_filtered`.

---

### Model Development dengan Collaborative Filtering menggunakan Keras

**Membangun Kelas Model RecommenderNet**

Kami mengembangkan model deep learning kustom menggunakan TensorFlow/Keras dengan pendekatan subclassing API.

**Detail Arsitektur:**

- **User Embedding Layer:** Membuat representasi vektor laten yang mewakili setiap pengguna.
- **Book Embedding Layer:** Membuat representasi vektor laten yang mewakili setiap buku.
- **Bias Layer:** Menambahkan bias khusus untuk masing-masing pengguna dan buku.
- **Dot Product:** Menghitung skor kecocokan antara vektor embedding pengguna dan buku.
- **Fungsi Aktivasi:** Sigmoid, untuk memastikan nilai prediksi berada dalam rentang \[0, 1\].

Model ini dirancang untuk secara efisien mempelajari pola interaksi antara pengguna dan item melalui embedding yang terlatih.


#### Proses Model:

**Inisialisasi dan Kompilasi Model RecommenderNet**

Setelah menyelesaikan desain arsitektur model, kami melakukan proses inisialisasi dan kompilasi dengan pengaturan sebagai berikut:

- **Fungsi Loss:** Binary Crossentropy  
- **Optimizer:** Adam dengan learning rate 0.001  
- **Metrik Evaluasi:** Root Mean Squared Error (RMSE)  

Pemilihan fungsi loss binary crossentropy didasarkan pada karakter output model yang memprediksi probabilitas keterkaitan antara pengguna dan buku dalam rentang nilai \[0,1\].

Setelah tahap kompilasi selesai, model siap untuk dilatih.

**Pelatihan Model RecommenderNet**

Model dilatih menggunakan data yang telah dibagi menjadi set pelatihan dan validasi.

Pengaturan pelatihan meliputi:

- **Batch Size:** 8  
- **Jumlah Epoch:** 100  
- **Optimizer:** Adam  
- **Fungsi Loss:** Binary Crossentropy  
- **Metrik Evaluasi:** Root Mean Squared Error (RMSE)  

Konfigurasi ini bertujuan untuk mengoptimalkan performa model dalam mempelajari pola interaksi antara pengguna dan buku secara efektif.

#### Contoh Rekomendasi:
Pada tahap ini, kami menguji performa model Collaborative Filtering yang telah dilatih dengan memberikan rekomendasi buku untuk pengguna tertentu, yaitu User-ID 95301.

Proses:

* Model menerima input berupa ID user dan seluruh daftar buku (ISBN).
* Model memprediksi skor kecocokan atau preferensi user terhadap setiap buku dalam skala $0,1$.
* Berdasarkan skor prediksi, sistem memilih buku dengan skor tertinggi yang belum pernah dirating oleh user tersebut.
* Hasil rekomendasi berupa daftar buku yang diprediksi paling sesuai dengan minat dan preferensi user tersebut.

| No | Book-Title                                             | Book-Author      |
| :- | :----------------------------------------------------- | :--------------- |
| 1  | The Return of the King (The Lord of the Rings, Part 3) | J.R.R. Tolkien   |
| 2  | The Two Towers (The Lord of the Rings, Part 2)         | J.R.R. Tolkien   |
| 3  | The Giving Tree                                        | Shel Silverstein |
| 4  | Dilbert: A Book of Postcards                           | Scott Adams      |
| 5  | Harry Potter and the Chamber of Secrets Postcard Book  | J.K. Rowling     |


**Interpretasi Hasil:**
Model rekomendasi berhasil mengidentifikasi preferensi pengguna dengan cukup baik. Berdasarkan hasil rekomendasi, pengguna ini tampaknya memiliki ketertarikan yang kuat terhadap genre **fantasi dan fiksi klasik**, sebagaimana terlihat dari munculnya buku *The Return of the King* dan *The Two Towers* karya **J.R.R. Tolkien**, serta *Harry Potter and the Chamber of Secrets* karya **J.K. Rowling**.

Selain itu, keberadaan buku *The Giving Tree* dari **Shel Silverstein** menunjukkan bahwa pengguna juga menyukai **buku-buku bernuansa emosional atau bermakna dalam**, yang sering ditemukan dalam buku anak-anak dengan pesan moral yang kuat. Adapun *Dilbert: A Book of Postcards* dari **Scott Adams** mencerminkan minat pengguna terhadap **humor dan karya visual ringan**, yang biasanya disukai oleh pembaca yang ingin hiburan santai.

Secara keseluruhan, model mampu memberikan rekomendasi yang tidak hanya relevan dengan **tema atau genre utama** yang disukai pengguna, tetapi juga menawarkan **variasi konten** yang bisa memperluas wawasan bacaan mereka.

---

### Top-N Recommendation Output

- **Content-Based Filtering:** Berdasarkan kemiripan judul buku.  
- **Memory-Based Collaborative Filtering:** Berdasarkan user lain yang mirip.  
- **Model-Based Collaborative Filtering:** Berdasarkan representasi laten dari interaksi user-buku.

---

### Perbandingan Pendekatan

| **Pendekatan**                                          | **Kelebihan**                                                                                                                               | **Kekurangan**                                                                                                                            |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Content-Based Filtering**                             | - Tidak memerlukan data interaksi (rating).<br>- Mampu merekomendasikan item baru yang belum pernah dinilai.                                | - Rekomendasi terbatas pada item yang mirip saja.<br>- Tidak mampu menangkap pola preferensi antar pengguna.                              |
| **Collaborative Filtering (User-Based)**                | - Dapat mengenali pola kesamaan dalam komunitas pengguna.<br>- Rekomendasi lebih bervariasi karena mempertimbangkan selera banyak pengguna. | - Membutuhkan data interaksi user-item dalam jumlah cukup.<br>- Mengalami kesulitan jika menghadapi pengguna atau buku baru (cold-start). |
| **Model-Based Collaborative Filtering (Deep Learning)** | - Mampu mempelajari pola preferensi yang kompleks dan tersembunyi.<br>- Tidak bergantung pada kemiripan eksplisit antar item.               | - Membutuhkan waktu pelatihan (training) yang lebih lama.<br>- Risiko overfitting jika data yang tersedia terbatas.                       |

Dengan mengombinasikan ketiga pendekatan ini, sistem rekomendasi dapat menjadi lebih tangguh dan adaptif. Pendekatan gabungan memungkinkan sistem memberikan rekomendasi yang relevan, baik dari sisi kesamaan konten maupun dari pola interaksi dan preferensi pengguna.

---
# Evaluation

Pada bagian ini, kami mengevaluasi kinerja sistem rekomendasi yang telah dibangun menggunakan dua pendekatan: Content-Based Filtering dan Collaborative Filtering (baik Memory-Based maupun Model-Based).

## Visualisasi Learning Curve 

Grafik berikut memperlihatkan tren perubahan nilai Root Mean Squared Error (RMSE) pada data pelatihan dan validasi sepanjang proses pelatihan model.

![Learning Curve](gambar/foto%20visual.png)

**Insight:**  
Grafik pelatihan model menunjukkan penurunan bertahap pada nilai *Root Mean Squared Error (RMSE)* baik pada data pelatihan maupun data validasi sepanjang 15 epoch. RMSE pada data pelatihan menurun hingga mendekati angka **0.1511**, sementara RMSE pada data validasi stabil di kisaran **0.1835** hingga **0.1845**.

Model menunjukkan proses konvergensi yang baik dan stabil, terutama dalam 10–15 epoch pertama. Tidak ditemukan indikasi **overfitting yang signifikan**, karena perbedaan nilai RMSE antara pelatihan dan validasi relatif kecil. Hal ini menunjukkan bahwa model mampu belajar pola dari data dengan efektif dan tetap mempertahankan kemampuan generalisasi terhadap data baru.

---

## Metrik Evaluasi: Root Mean Squared Error (RMSE)

Untuk menilai kinerja model Collaborative Filtering yang dibangun menggunakan TensorFlow Keras, kami menggunakan metrik Root Mean Squared Error (RMSE). RMSE merupakan metrik populer dalam permasalahan regresi, termasuk prediksi nilai rating, karena dapat menunjukkan sejauh mana hasil prediksi menyimpang dari nilai aktual.

**Alasan Penggunaan RMSE:**

- Data yang digunakan merupakan rating berskala kontinu, bukan klasifikasi.  
- RMSE memberikan informasi yang jelas mengenai besarnya kesalahan prediksi dalam satuan rating.  
- Nilai RMSE yang lebih kecil mengindikasikan model yang lebih akurat.


## Hasil Evaluasi Model

| Dataset        | RMSE       |
| -------------- | ---------- |
| Training Data  | ± 0.1511   |
| Validation Data| ± 0.1835 – 0.1845  |

#### Hasil Output:
- Nilai RMSE terbaik pada training tercapai pada **epoch ke-15** dengan nilai sekitar **0.1511**.
- RMSE validasi menunjukkan nilai stabil sejak **epoch ke-10**, menandakan model telah mencapai kondisi optimal. 
- Tidak ada fluktuasi besar pada kurva validasi, yang menandakan model tidak mengalami overfitting.

**Interpretasi:**
Model berhasil melakukan pembelajaran secara efektif dan stabil. Nilai RMSE yang rendah baik pada data training maupun validasi menunjukkan bahwa model memiliki **akurasi prediksi yang baik** dalam mempelajari hubungan atau pola dari data pengguna dan item (misalnya, rating buku). Perbedaan yang kecil antara training dan validation error mengindikasikan bahwa model **tidak hanya hafal data training**, tetapi juga mampu **mengeneralisasi dengan baik** terhadap data yang belum pernah dilihat sebelumnya.

---

## Content-Based Filtering (Precision@5)

- **Rata-rata Precision@5:** 0.0633 atau 6,33%

**Interpretasi:**

- Dari lima rekomendasi teratas, rata-rata hanya sekitar 6,3% yang benar-benar sesuai dengan preferensi pengguna berdasarkan histori rating.  
- Sistem cenderung merekomendasikan buku dengan kemiripan judul, namun belum tentu relevan secara personal.

**Insight Tambahan:**

- Rendahnya tingkat presisi disebabkan oleh keterbatasan fitur konten yang digunakan, yakni hanya mengandalkan judul buku.  
- Meskipun performanya kurang optimal dari sisi presisi, pendekatan Content-Based Filtering tetap memiliki nilai guna, terutama dalam mengatasi permasalahan *cold-start*, baik untuk pengguna baru maupun item (buku) yang belum banyak diberi rating.

---
## Hubungan Evaluasi dengan Business Understanding

Evaluasi terhadap model rekomendasi ini mengacu kembali pada tujuan bisnis dan rumusan masalah yang mendasari pembangunan sistem, guna memastikan solusi yang dikembangkan relevan dan aplikatif.

---

### Masalah 1:  
Pengguna mengalami kesulitan dalam menentukan pilihan buku di tengah banyaknya opsi yang tersedia.

**Solusi & Refleksi Evaluasi:**  
Sistem rekomendasi mampu mempermudah proses pencarian dengan menyajikan daftar buku teratas yang sesuai dengan preferensi masing-masing pengguna. Rekomendasi yang ditampilkan terbukti sejalan dengan minat dan genre favorit user, baik melalui pendekatan berbasis konten maupun perilaku pengguna lain yang serupa.

---

### Masalah 2:  
Minimnya data interaksi dari pengguna baru atau pasif menyulitkan dalam memahami preferensinya.

**Solusi & Refleksi Evaluasi:**  
Dengan mengandalkan pendekatan berbasis konten, sistem tetap dapat memberikan saran buku meskipun histori interaksi sangat terbatas. Hal ini menjadi solusi awal yang efektif dalam menghadapi tantangan *cold-start*, terutama saat pengguna belum aktif memberikan rating.

---

### Masalah 3:  
Platform membutuhkan pendekatan personalisasi untuk meningkatkan pengalaman pengguna dan mendorong keterlibatan lebih dalam.

**Solusi & Refleksi Evaluasi:**  
Melalui pendekatan Collaborative Filtering, baik berbasis pengguna maupun model, sistem dapat mengidentifikasi preferensi tersembunyi dan menyusun rekomendasi secara personal. Model deep learning juga berkontribusi besar dalam menghasilkan prediksi akurat berkat representasi laten (embedding) yang dibentuk dari interaksi user-buku, seperti tercermin dari nilai RMSE yang rendah.

---

### Tujuan Bisnis yang Tercapai:

- Sistem membantu pengguna dalam menemukan buku yang relevan dengan minat dan kebiasaan bacanya.  
- Rekomendasi yang diberikan mampu mengurangi kebingungan pengguna saat dihadapkan dengan banyak pilihan.  
- Pengalaman pengguna meningkat karena sistem memberikan saran yang terasa lebih personal dan kontekstual.  
- Pengguna terdorong untuk menjelajahi genre-genre baru berkat kemampuan model dalam mengenali pola minat yang tidak langsung terlihat.

---

**Kesimpulan:**  
Model telah berhasil menjawab seluruh *problem statements*, memenuhi goals bisnis, dan memberikan solusi teknis yang tepat sesuai dengan karakteristik data dan kebutuhan pengguna.

---

## Kesimpulan Evaluasi Hasil Rekomendasi

**Kesimpulan Evaluasi Sistem:**  

- Pendekatan Content-Based Filtering terbukti cukup efektif sebagai langkah awal, namun keterbatasannya terletak pada fokus yang hanya mengandalkan judul buku. Nilai Precision@5 yang hanya mencapai 6,33% menandakan bahwa sistem ini belum sepenuhnya mampu menangkap preferensi personal secara mendalam.

- User-Based Collaborative Filtering menawarkan rekomendasi yang lebih sesuai dengan referensi pengguna lain yang serupa, meskipun performanya menurun ketika diterapkan pada pengguna baru atau yang belum aktif.

- Pendekatan Model-Based Collaborative Filtering menunjukkan performa paling optimal. Nilai RMSE yang rendah dan kurva pelatihan yang stabil mengindikasikan bahwa model ini dapat secara efektif mengenali dan memetakan pola interaksi pengguna terhadap buku.

Secara keseluruhan, kombinasi dari ketiga pendekatan ini telah berhasil memberikan solusi yang relevan terhadap permasalahan yang ada dan mendukung pencapaian target bisnis secara menyeluruh.
