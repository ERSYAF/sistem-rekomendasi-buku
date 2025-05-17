{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Gd6dXkQXxTA7"
   },
   "source": [
    "# Proyek Akhir : Membuat Model Sistem Rekomendasi\n",
    "- **Nama:** [Era Syafina]\n",
    "- **Email:** [erasyafina025@gmail.com]\n",
    "- **ID Dicoding:** [Ersyafin]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sq5XaI7F76K2"
   },
   "source": [
    "# Project Overview\n",
    "\n",
    "Membaca buku merupakan kegiatan penting dalam pengembangan wawasan, kemampuan berpikir kritis, serta kesehatan mental. Namun, di tengah kemajuan era digital, pengguna dihadapkan pada begitu banyak pilihan buku di platform daring, yang menimbulkan fenomena “kelebihan pilihan” (choice overload).\n",
    "\n",
    "Untuk mengatasi tantangan ini, dibutuhkan sebuah sistem rekomendasi yang mampu membantu pembaca menemukan buku-buku yang sesuai dengan minat mereka berdasarkan riwayat interaksi, seperti penilaian (rating) dan ulasan yang telah diberikan.\n",
    "\n",
    "# Mengapa Masalah Ini Perlu Diselesaikan?\n",
    "\n",
    "* Banyak pengguna merasa kewalahan ketika harus memilih buku karena banyaknya opsi yang tersedia.\n",
    "* Sistem rekomendasi berpotensi meningkatkan keterlibatan dan loyalitas pengguna terhadap platform digital buku.\n",
    "* Penulis dan penerbit dapat menjangkau target audiens yang lebih tepat sasaran berdasarkan genre atau kategori bukunya.\n",
    "\n",
    "# Pendekatan Penyelesaian\n",
    "\n",
    "Solusi dirancang dengan membangun sistem rekomendasi menggunakan pendekatan berikut:\n",
    "\n",
    "* **Content-Based Filtering**: Memanfaatkan kemiripan konten, khususnya judul buku, menggunakan metode TF-IDF dan Cosine Similarity.\n",
    "* **User-Based Collaborative Filtering (Memory-Based)**: Menganalisis pola rating antar pengguna untuk menemukan kemiripan preferensi.\n",
    "* **Model-Based Collaborative Filtering**: Menerapkan pembelajaran representasi (embedding) dengan TensorFlow/Keras untuk menangkap pola kompleks dalam data interaksi.\n",
    "\n",
    "Dataset yang digunakan diambil dari [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) yang tersedia di Kaggle.\n",
    "\n",
    "Referensi:\n",
    "\n",
    "* Alkaff, M., Khatimi, H., & Eriadi, A. (2020). Sistem Rekomendasi Buku Menggunakan Weighted Tree Similarity dan Content Based Filtering. MATRIK J. Manajemen, Tek. Inform. dan Rekayasa Komput, 20(1), 193-202.\n",
    "* Pratama, R. V., & Hasrullah, H. (2025). Pengembangan Sistem Rekomendasi Buku untuk Meningkatkan Minat Baca dengan Pendekatan Hybrid Filtering. Jurnal Inovasi Global, 3(1), 2182-2191.\n",
    "\n",
    "---\n",
    "\n",
    "# Business Understanding\n",
    "\n",
    "Bagian ini menjelaskan proses identifikasi masalah yang menjadi dasar pengembangan sistem rekomendasi berbasis machine learning.\n",
    "\n",
    "# Problem Statements:\n",
    "\n",
    "* Banyaknya pilihan buku di platform daring membuat pengguna kesulitan menemukan bacaan yang sesuai dengan selera mereka.\n",
    "* Pengguna baru atau yang jarang berinteraksi sering kali tidak memberikan cukup data (rating), sehingga sistem sulit memahami preferensi mereka.\n",
    "* Platform perlu memberikan pengalaman yang lebih personal agar pengguna merasa lebih terlibat dan setia menggunakan layanan.\n",
    "\n",
    "# Goals:\n",
    "\n",
    "* Membantu pengguna menemukan buku yang cocok dengan minat mereka berdasarkan interaksi sebelumnya.\n",
    "* Mengurangi efek kelebihan pilihan dengan menyediakan rekomendasi buku *top-N* yang dipersonalisasi.\n",
    "* Meningkatkan keterlibatan dan waktu yang dihabiskan pengguna di platform melalui sistem rekomendasi yang efektif.\n",
    "* Mendorong pengguna untuk mengeksplorasi buku-buku dari genre yang mungkin mereka sukai tetapi belum mereka coba.\n",
    "\n",
    "# Solution Statements:\n",
    "\n",
    "* **Content-Based Filtering (CBF)**\n",
    "  Menggunakan informasi dari buku seperti judul dan kesamaan antar judul dengan pendekatan TF-IDF dan Cosine Similarity.\n",
    "  Kelebihan: Efektif untuk pengguna baru yang minim interaksi.\n",
    "  Kekurangan: Cenderung merekomendasikan buku yang mirip saja, sehingga eksplorasi terbatas.\n",
    "\n",
    "* **User-Based Collaborative Filtering (Memory-Based)**\n",
    "  Memanfaatkan pola rating antar pengguna untuk memberikan rekomendasi berdasarkan kemiripan preferensi.\n",
    "  Kelebihan: Dapat menangkap variasi selera pengguna.\n",
    "  Kekurangan: Mengalami kesulitan dalam menghadapi pengguna baru (cold-start) dengan data interaksi yang minim.\n",
    "\n",
    "* **Model-Based Collaborative Filtering (Deep Learning dengan Keras)**\n",
    "  Menggunakan embedding untuk mempelajari preferensi laten pengguna dan buku, sehingga bisa memprediksi rating dengan lebih akurat.\n",
    "  Kelebihan: Skalabilitas tinggi dan mampu menangkap pola kompleks.\n",
    "  Kekurangan: Memerlukan data dalam jumlah besar dan waktu pelatihan lebih lama dibanding metode lain.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "me3g8kNf3XeE"
   },
   "source": [
    "# Data Loading"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_oRS98LE3pZl"
   },
   "source": [
    "Pada tahap ini, sistem akan melakukan proses pengunduhan dataset secara otomatis dari Kaggle.\n",
    "Dataset yang digunakan adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "# Set path ke kaggle.json\n",
    "os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: kaggle in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (1.7.4.5)\n",
      "Requirement already satisfied: bleach in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (6.1.0)\n",
      "Requirement already satisfied: certifi>=14.05.14 in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (2024.2.2)\n",
      "Requirement already satisfied: charset-normalizer in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (3.3.2)\n",
      "Requirement already satisfied: idna in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (3.6)\n",
      "Requirement already satisfied: protobuf in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (3.20.3)\n",
      "Requirement already satisfied: python-dateutil>=2.5.3 in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (2.9.0.post0)\n",
      "Requirement already satisfied: python-slugify in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (8.0.4)\n",
      "Requirement already satisfied: requests in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (2.31.0)\n",
      "Requirement already satisfied: setuptools>=21.0.0 in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (65.5.0)\n",
      "Requirement already satisfied: six>=1.10 in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (1.16.0)\n",
      "Requirement already satisfied: text-unidecode in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (1.3)\n",
      "Requirement already satisfied: tqdm in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (4.67.1)\n",
      "Requirement already satisfied: urllib3>=1.15.1 in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (2.2.1)\n",
      "Requirement already satisfied: webencodings in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from kaggle) (0.5.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\ersyaf\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from tqdm->kaggle) (0.4.6)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 24.3.1 -> 25.1.1\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "!pip install kaggle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset URL: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset\n",
      "License(s): CC0-1.0\n",
      "Downloading book-recommendation-dataset.zip to C:\\Users\\ersyaf\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "  0%|          | 0.00/24.3M [00:00<?, ?B/s]\n",
      "100%|##########| 24.3M/24.3M [00:00<00:00, 478MB/s]\n"
     ]
    }
   ],
   "source": [
    "!kaggle datasets download -d arashnic/book-recommendation-dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "id": "oPWLV2yF34sa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done\n"
     ]
    }
   ],
   "source": [
    "# Import library zipfile\n",
    "import zipfile\n",
    "\n",
    "# Mengekstrak file ZIP ke folder BookRecommendation\n",
    "with zipfile.ZipFile('book-recommendation-dataset.zip', 'r') as zip_ref:\n",
    "    zip_ref.extractall('BookRecommendation')\n",
    "\n",
    "print(\"Done\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LMfxEWis38aG"
   },
   "source": [
    "# Data Understanding"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jlFLEUge4AD_"
   },
   "source": [
    "Dataset ini terdiri dari tiga file utama:\n",
    "\n",
    "* **Books.csv**: berisi informasi lengkap tentang buku, mencakup ISBN, judul, penulis, tahun terbit, penerbit, dan link gambar sampul.\n",
    "* **Users.csv**: memuat data pengguna, seperti ID, lokasi tempat tinggal, dan usia mereka.\n",
    "* **Ratings.csv**: mencatat interaksi antara pengguna dan buku dalam bentuk rating yang diberikan.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "XSZgPKcI39PA",
    "outputId": "303a79a4-328f-4bde-fd7d-12f64eb565aa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset berhasil dimuat!\n"
     ]
    }
   ],
   "source": [
    "# Load dataset menggunakan pandas\n",
    "import pandas as pd\n",
    "\n",
    "books = pd.read_csv('BookRecommendation/Books.csv', low_memory=False)\n",
    "users = pd.read_csv('BookRecommendation/Users.csv')\n",
    "ratings = pd.read_csv('BookRecommendation/Ratings.csv')\n",
    "\n",
    "print(\"Dataset berhasil dimuat!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xS6oeXW94IjG"
   },
   "source": [
    "Pada tahap ini, kita akan memeriksa semua data-data pada Books.csv, Users.csv, dan Ratings.csv supaya Memahami struktur, isi, kualitas, dan karakteristik data sebelum melakukan analisis lebih lanjut."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0BZglxVx4Lo6"
   },
   "source": [
    "## Dataset Books"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XEe7eXZp4SRR"
   },
   "source": [
    "**Disini kita akan menampilkan tampilan dataset books**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 434
    },
    "id": "tM4ClK1C4JQF",
    "outputId": "ceaa3fce-9edd-4b96-f9c9-d2505637c4f0"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Title</th>\n",
       "      <th>Book-Author</th>\n",
       "      <th>Year-Of-Publication</th>\n",
       "      <th>Publisher</th>\n",
       "      <th>Image-URL-S</th>\n",
       "      <th>Image-URL-M</th>\n",
       "      <th>Image-URL-L</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0195153448</td>\n",
       "      <td>Classical Mythology</td>\n",
       "      <td>Mark P. O. Morford</td>\n",
       "      <td>2002</td>\n",
       "      <td>Oxford University Press</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0002005018</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0060973129</td>\n",
       "      <td>Decision in Normandy</td>\n",
       "      <td>Carlo D'Este</td>\n",
       "      <td>1991</td>\n",
       "      <td>HarperPerennial</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0374157065</td>\n",
       "      <td>Flu: The Story of the Great Influenza Pandemic...</td>\n",
       "      <td>Gina Bari Kolata</td>\n",
       "      <td>1999</td>\n",
       "      <td>Farrar Straus Giroux</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0393045218</td>\n",
       "      <td>The Mummies of Urumchi</td>\n",
       "      <td>E. J. W. Barber</td>\n",
       "      <td>1999</td>\n",
       "      <td>W. W. Norton &amp;amp; Company</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         ISBN                                         Book-Title  \\\n",
       "0  0195153448                                Classical Mythology   \n",
       "1  0002005018                                       Clara Callan   \n",
       "2  0060973129                               Decision in Normandy   \n",
       "3  0374157065  Flu: The Story of the Great Influenza Pandemic...   \n",
       "4  0393045218                             The Mummies of Urumchi   \n",
       "\n",
       "            Book-Author Year-Of-Publication                   Publisher  \\\n",
       "0    Mark P. O. Morford                2002     Oxford University Press   \n",
       "1  Richard Bruce Wright                2001       HarperFlamingo Canada   \n",
       "2          Carlo D'Este                1991             HarperPerennial   \n",
       "3      Gina Bari Kolata                1999        Farrar Straus Giroux   \n",
       "4       E. J. W. Barber                1999  W. W. Norton &amp; Company   \n",
       "\n",
       "                                         Image-URL-S  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0060973129.0...   \n",
       "3  http://images.amazon.com/images/P/0374157065.0...   \n",
       "4  http://images.amazon.com/images/P/0393045218.0...   \n",
       "\n",
       "                                         Image-URL-M  \\\n",
       "0  http://images.amazon.com/images/P/0195153448.0...   \n",
       "1  http://images.amazon.com/images/P/0002005018.0...   \n",
       "2  http://images.amazon.com/images/P/0060973129.0...   \n",
       "3  http://images.amazon.com/images/P/0374157065.0...   \n",
       "4  http://images.amazon.com/images/P/0393045218.0...   \n",
       "\n",
       "                                         Image-URL-L  \n",
       "0  http://images.amazon.com/images/P/0195153448.0...  \n",
       "1  http://images.amazon.com/images/P/0002005018.0...  \n",
       "2  http://images.amazon.com/images/P/0060973129.0...  \n",
       "3  http://images.amazon.com/images/P/0374157065.0...  \n",
       "4  http://images.amazon.com/images/P/0393045218.0...  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "books.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YKF43SZN4XLP"
   },
   "source": [
    "**Pemeriksaan Awal Dataset Books**\n",
    "\n",
    "Pada tahap ini, kami menggunakan fungsi `.head()` untuk menampilkan **lima baris pertama** dari dataset **Books**. Tujuan utamanya adalah untuk mendapatkan gambaran awal mengenai struktur data serta jenis informasi yang tersedia.\n",
    "\n",
    "**Contoh Data Lima Buku Pertama:**\n",
    "\n",
    "1. **\"Classical Mythology\"** oleh *Mark P. O. Morford* — diterbitkan tahun **2002** oleh *Oxford University Press*\n",
    "2. **\"Clara Callan\"** oleh *Richard Bruce Wright* — diterbitkan tahun **2001** oleh *HarperFlamingo Canada*\n",
    "3. **\"Decision in Normandy\"** oleh *Carlo D'Este* — diterbitkan tahun **1991** oleh *HarperPerennial*\n",
    "4. **\"Flu: The Story of the Great Influenza Pandemic of 1918...\"** oleh *Gina Bari Kolata* — diterbitkan tahun **1999** oleh *Farrar Straus Giroux*\n",
    "5. **\"The Mummies of Urumchi\"** oleh *E. J. W. Barber* — diterbitkan tahun **1999** oleh *W. W. Norton & Company*\n",
    "\n",
    "Dataset ini tidak hanya mencakup informasi utama seperti **ISBN**, **judul buku**, **penulis**, **tahun terbit**, dan **penerbit**, tetapi juga menyediakan **URL gambar sampul buku** dalam tiga ukuran berbeda: kecil, sedang, dan besar.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iFQw8z-q4dLG"
   },
   "source": [
    "**Menampilkan jumlah baris dan kolom dari dataset books**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ny15gzxU4YVy",
    "outputId": "e095299d-3dec-4b3c-c2bb-d04a5a266539"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Books: (271360, 8)\n"
     ]
    }
   ],
   "source": [
    "print(f\"Jumlah data Books: {books.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_e1aJTGC4knz"
   },
   "source": [
    "Terdapat data buku dengan shape 271.360 baris, 8 kolom"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9CKDvURD4mhl"
   },
   "source": [
    "**Menampilkan struktur dataset books**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "8vhi7LeM4k9p",
    "outputId": "94b236ef-d134-4552-abe1-8f909e271c58"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 271360 entries, 0 to 271359\n",
      "Data columns (total 8 columns):\n",
      " #   Column               Non-Null Count   Dtype \n",
      "---  ------               --------------   ----- \n",
      " 0   ISBN                 271360 non-null  object\n",
      " 1   Book-Title           271360 non-null  object\n",
      " 2   Book-Author          271358 non-null  object\n",
      " 3   Year-Of-Publication  271360 non-null  object\n",
      " 4   Publisher            271358 non-null  object\n",
      " 5   Image-URL-S          271360 non-null  object\n",
      " 6   Image-URL-M          271360 non-null  object\n",
      " 7   Image-URL-L          271357 non-null  object\n",
      "dtypes: object(8)\n",
      "memory usage: 16.6+ MB\n"
     ]
    }
   ],
   "source": [
    "books.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Vhh-fKsZ4qFe"
   },
   "source": [
    "**Uraian Variabel** - `Books.csv`\n",
    "\n",
    "* **ISBN**: Merupakan identifikasi unik untuk setiap buku. Bertipe data *object* dengan jumlah data tidak kosong sebanyak 271.360.\n",
    "* **Book-Title**: Menyimpan judul buku. Bertipe *object*, seluruh data terisi (271.360 entri).\n",
    "* **Book-Author**: Berisi nama penulis buku. Tipe data *object*. Terdapat 271.358 data terisi dan 2 nilai yang hilang.\n",
    "* **Year-Of-Publication**: Tahun buku diterbitkan. Bertipe *object* dan tidak memiliki nilai kosong (271.360 entri).\n",
    "* **Publisher**: Nama penerbit buku. Tipe *object*, dengan 271.358 entri terisi dan 2 data yang hilang.\n",
    "* **Image-URL-S**: Link menuju gambar sampul kecil buku. Tipe data *object* dengan 271.360 data lengkap.\n",
    "* **Image-URL-M**: Menyimpan URL gambar sampul ukuran sedang. Bertipe *object*, seluruh entri terisi lengkap.\n",
    "* **Image-URL-L**: URL gambar sampul besar buku. Tipe *object*, namun terdapat 3 data yang hilang (271.357 entri terisi).\n",
    "\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "\n",
    "Pada tahap ini, struktur data Buku ditampilkan menggunakan fungsi .info(). Dataset ini berisi sebanyak 271.360 baris data dengan 8 kolom, di mana seluruh kolom bertipe data object atau string.\n",
    "\n",
    "Rincian jumlah data yang tidak kosong (non-null) pada setiap kolom adalah sebagai berikut:\n",
    "\n",
    "- ISBN: 271.360 non-null\n",
    "- Book-Title: 271.360 non-null\n",
    "- Book-Author: 271.358 non-null\n",
    "- Year-Of-Publication: 271.360 non-null\n",
    "- Publisher: 271.358 non-null\n",
    "- Image-URL-S: 271.360 non-null\n",
    "- Image-URL-M: 271.360 non-null\n",
    "- Image-URL-L: 271.357 non-null\n",
    "\n",
    "Beberapa kolom memiliki nilai yang hilang (missing value), khususnya pada `Book-Author`, `Publisher`, dan `Image-URL-L`. Kondisi ini perlu ditangani lebih lanjut pada proses pembersihan data berikutnya."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zM8KVjxd42nf"
   },
   "source": [
    "**Menampilkan statistik ringkasan untuk dataset books**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 229
    },
    "id": "yS7pLceo4qcC",
    "outputId": "57e49638-523f-49e9-ef95-4f5d95a8f34b"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Title</th>\n",
       "      <th>Book-Author</th>\n",
       "      <th>Year-Of-Publication</th>\n",
       "      <th>Publisher</th>\n",
       "      <th>Image-URL-S</th>\n",
       "      <th>Image-URL-M</th>\n",
       "      <th>Image-URL-L</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>271360</td>\n",
       "      <td>271360</td>\n",
       "      <td>271358</td>\n",
       "      <td>271360</td>\n",
       "      <td>271358</td>\n",
       "      <td>271360</td>\n",
       "      <td>271360</td>\n",
       "      <td>271357</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>271360</td>\n",
       "      <td>242135</td>\n",
       "      <td>102022</td>\n",
       "      <td>118</td>\n",
       "      <td>16807</td>\n",
       "      <td>271044</td>\n",
       "      <td>271044</td>\n",
       "      <td>271041</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>0195153448</td>\n",
       "      <td>Selected Poems</td>\n",
       "      <td>Agatha Christie</td>\n",
       "      <td>2002</td>\n",
       "      <td>Harlequin</td>\n",
       "      <td>http://images.amazon.com/images/P/185326119X.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/185326119X.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/225307649X.0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>1</td>\n",
       "      <td>27</td>\n",
       "      <td>632</td>\n",
       "      <td>17627</td>\n",
       "      <td>7535</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              ISBN      Book-Title      Book-Author Year-Of-Publication  \\\n",
       "count       271360          271360           271358              271360   \n",
       "unique      271360          242135           102022                 118   \n",
       "top     0195153448  Selected Poems  Agatha Christie                2002   \n",
       "freq             1              27              632               17627   \n",
       "\n",
       "        Publisher                                        Image-URL-S  \\\n",
       "count      271358                                             271360   \n",
       "unique      16807                                             271044   \n",
       "top     Harlequin  http://images.amazon.com/images/P/185326119X.0...   \n",
       "freq         7535                                                  2   \n",
       "\n",
       "                                              Image-URL-M  \\\n",
       "count                                              271360   \n",
       "unique                                             271044   \n",
       "top     http://images.amazon.com/images/P/185326119X.0...   \n",
       "freq                                                    2   \n",
       "\n",
       "                                              Image-URL-L  \n",
       "count                                              271357  \n",
       "unique                                             271041  \n",
       "top     http://images.amazon.com/images/P/225307649X.0...  \n",
       "freq                                                    2  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "books.describe(include='all')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XXYV66Cj47a5"
   },
   "source": [
    "Pada tahap ini, kami menerapkan fungsi .describe(include='all') untuk memperoleh ringkasan statistik dari dataset books. Statistik ini memberikan gambaran umum mengenai data, seperti total entri, jumlah nilai unik, nilai yang paling sering muncul (top), serta frekuensi kemunculannya (freq).\n",
    "\n",
    "\n",
    "**Beberapa insight penting dari dataset books:**\n",
    "\n",
    "* Terdapat sebanyak **271.360 ISBN unik**, yang menunjukkan bahwa setiap buku memiliki kode ISBN yang berbeda.\n",
    "* Jumlah **judul buku unik mencapai 242.135**, menandakan adanya beberapa duplikasi judul dengan edisi atau versi berbeda.\n",
    "* **Agatha Christie** tercatat sebagai penulis paling produktif dalam dataset ini, dengan total **632 buku**.\n",
    "* **Tahun terbit terbanyak adalah 2002**, di mana **17.627 buku** diterbitkan pada tahun tersebut.\n",
    "* **Penerbit paling dominan adalah Harlequin**, dengan **7.535 buku** yang diterbitkan.\n",
    "* Untuk kolom **Image-URL**, sebagian besar URL bersifat unik, namun terdapat sejumlah URL yang digunakan berulang kali untuk beberapa buku.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vas-ISYa4-K3"
   },
   "source": [
    "**Mengecek jumlah missing value pada setiap kolom di data Buku**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 335
    },
    "id": "BVhQUbBE471U",
    "outputId": "cfd47f2b-a434-459e-fafd-1395886fc4c7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ISBN                   0\n",
       "Book-Title             0\n",
       "Book-Author            2\n",
       "Year-Of-Publication    0\n",
       "Publisher              2\n",
       "Image-URL-S            0\n",
       "Image-URL-M            0\n",
       "Image-URL-L            3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "books.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DekmpfV55B_2"
   },
   "source": [
    "Pada tahap ini, kami melakukan pengecekan jumlah nilai yang hilang (*missing value*) di setiap kolom dalam data Buku menggunakan fungsi `.isnull().sum()`. Langkah ini penting untuk mengidentifikasi kolom-kolom yang membutuhkan perlakuan khusus pada proses **data preparation**.\n",
    "\n",
    "Berdasarkan hasil pemeriksaan, ditemukan hal-hal berikut:\n",
    "\n",
    "* Kolom **Book-Author** memiliki **2 nilai yang hilang**.\n",
    "* Kolom **Publisher** juga memiliki **2 nilai yang hilang**.\n",
    "* Kolom **Image-URL-L** mengandung **3 nilai yang hilang**.\n",
    "* Sementara itu, kolom lain seperti **ISBN**, **Book-Title**, **Year-Of-Publication**, **Image-URL-S**, dan **Image-URL-M** tidak memiliki missing value.\n",
    "\n",
    "Dengan demikian, perlu disiapkan langkah penanganan khusus terhadap missing value, terutama pada kolom **Book-Author**, **Publisher**, dan **Image-URL-L**, pada tahap pembersihan data selanjutnya.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "guSj3PTT5E0F"
   },
   "source": [
    "**Memeriksa jumlah buku unik dan jumlah duplikat di dataset books**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "mXloJdfv5CaY",
    "outputId": "2f36efd1-dfdf-451b-a0ad-6248b6407a6e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah ISBN unik: 340556\n",
      "Jumlah duplikat di Books: 0\n"
     ]
    }
   ],
   "source": [
    "print('Jumlah ISBN unik:', ratings['ISBN'].nunique())\n",
    "print('Jumlah duplikat di Books:', books.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3NgIcDJ35Ia4"
   },
   "source": [
    "**Jumlah ISBN Unik dalam Dataset Ratings**\n",
    "Untuk memahami seberapa luas cakupan buku yang telah diberi rating oleh pengguna, dilakukan perhitungan jumlah ISBN unik menggunakan fungsi `.nunique()` pada kolom **ISBN** dalam dataset **Ratings**.\n",
    "\n",
    "**Hasil:**\n",
    "Terdapat **340.556 buku unik**, menunjukkan bahwa dataset ini memiliki koleksi buku yang sangat beragam yang telah diberi penilaian oleh berbagai pengguna.\n",
    "\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "**Pemeriksaan Duplikasi pada Dataset Books**\n",
    "\n",
    "Langkah selanjutnya adalah memastikan integritas data dengan memeriksa adanya **data duplikat** pada dataset **Books**. Duplikasi dapat menyebabkan bias dalam analisis dan mempengaruhi kualitas hasil akhir.\n",
    "\n",
    "Pemeriksaan dilakukan dengan metode identifikasi duplikasi berdasarkan keseluruhan baris data.\n",
    "\n",
    "**Hasil:**\n",
    "**Tidak ditemukan data duplikat** (jumlah duplikat = 0), sehingga tidak diperlukan tindakan penghapusan data pada tahap ini.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GkoLJ5b_5eDP"
   },
   "source": [
    "## Dataset Ratings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7pzhSWZJ5k2L"
   },
   "source": [
    "**Disini menampilkan tampilan data Ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "SmKkYylD5IvT",
    "outputId": "a693e613-ef2c-474d-d3a1-b856915c157a"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User-ID</th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>276725</td>\n",
       "      <td>034545104X</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>276726</td>\n",
       "      <td>0155061224</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>276727</td>\n",
       "      <td>0446520802</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>276729</td>\n",
       "      <td>052165615X</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>276729</td>\n",
       "      <td>0521795028</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User-ID        ISBN  Book-Rating\n",
       "0   276725  034545104X            0\n",
       "1   276726  0155061224            5\n",
       "2   276727  0446520802            0\n",
       "3   276729  052165615X            3\n",
       "4   276729  0521795028            6"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kTN8KyWa5oeH"
   },
   "source": [
    "**Menampilkan jumlah baris dan kolom dari data ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "SfIqBEFf5rb1",
    "outputId": "7eb8ca4c-818b-417d-dee9-34e87e892bd2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Ratings: (1149780, 3)\n"
     ]
    }
   ],
   "source": [
    "print(f\"Jumlah data Ratings: {ratings.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ky34j-c75yLK"
   },
   "source": [
    "Terdapat data ratings dengan shape 1.149.780 baris, 3 kolom"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RhMYrRW553O3"
   },
   "source": [
    "**Menampilkan struktur data ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ewCmzUWE5yb-",
    "outputId": "ccb00b1a-254c-4268-d107-faeef12a2654"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1149780 entries, 0 to 1149779\n",
      "Data columns (total 3 columns):\n",
      " #   Column       Non-Null Count    Dtype \n",
      "---  ------       --------------    ----- \n",
      " 0   User-ID      1149780 non-null  int64 \n",
      " 1   ISBN         1149780 non-null  object\n",
      " 2   Book-Rating  1149780 non-null  int64 \n",
      "dtypes: int64(2), object(1)\n",
      "memory usage: 26.3+ MB\n"
     ]
    }
   ],
   "source": [
    "ratings.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oOlIQrER58nP"
   },
   "source": [
    "**Uraian Variabel** – `Ratings.csv`\n",
    "\n",
    "* **User-ID**: Merupakan identitas unik dari pengguna yang memberikan rating. Bertipe data *int64* dengan total **1.149.780** entri yang lengkap (tidak ada nilai kosong).\n",
    "* **ISBN**: Menunjukkan kode buku (ISBN) yang diberikan rating oleh pengguna. Bertipe *object*, dan seluruhnya terisi sebanyak **1.149.780** baris.\n",
    "* **Book-Rating**: Nilai rating yang diberikan oleh pengguna terhadap buku, dengan skala **0 hingga 10**. Tipe data *int64*, tanpa missing value (**1.149.780 entri**).\n",
    "\n",
    "\n",
    "\n",
    "---\n",
    "Pada tahap ini, dilakukan peninjauan terhadap struktur dataset **Ratings** menggunakan fungsi `.info()`. Tujuannya adalah untuk memahami jumlah entri, tipe data pada masing-masing kolom, serta mengecek adanya *missing value*.\n",
    "\n",
    "**Insight Data:**\n",
    "* Tidak terdapat *missing value* pada ketiga kolom, karena seluruh kolom memiliki jumlah data non-null yang sama dengan jumlah total entri.\n",
    "\n",
    "Dengan demikian, dataset **Ratings** memiliki struktur data yang bersih dan siap untuk dianalisis lebih lanjut.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0YNVgE1C6GLM"
   },
   "source": [
    "**Menampilkan statistik ringkasan untuk data Ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "eofKByLp6BLM",
    "outputId": "0a7f3d9e-03c0-4d19-a2c3-180a4d526405"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User-ID</th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.149780e+06</td>\n",
       "      <td>1149780</td>\n",
       "      <td>1.149780e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>NaN</td>\n",
       "      <td>340556</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>NaN</td>\n",
       "      <td>0971880107</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>NaN</td>\n",
       "      <td>2502</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.403864e+05</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.866950e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.056228e+04</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.854184e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.034500e+04</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.410100e+05</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2.110280e+05</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2.788540e+05</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.000000e+01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             User-ID        ISBN   Book-Rating\n",
       "count   1.149780e+06     1149780  1.149780e+06\n",
       "unique           NaN      340556           NaN\n",
       "top              NaN  0971880107           NaN\n",
       "freq             NaN        2502           NaN\n",
       "mean    1.403864e+05         NaN  2.866950e+00\n",
       "std     8.056228e+04         NaN  3.854184e+00\n",
       "min     2.000000e+00         NaN  0.000000e+00\n",
       "25%     7.034500e+04         NaN  0.000000e+00\n",
       "50%     1.410100e+05         NaN  0.000000e+00\n",
       "75%     2.110280e+05         NaN  7.000000e+00\n",
       "max     2.788540e+05         NaN  1.000000e+01"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings.describe(include='all')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nFmU3VjM6KnF"
   },
   "source": [
    "Untuk mendapatkan gambaran umum terhadap karakteristik data, dilakukan analisis statistik deskriptif menggunakan fungsi `.describe(include='all')`.\n",
    "\n",
    "**Insight Utama:**\n",
    "\n",
    "* **Rata-rata rating:** 2.87\n",
    "* **Standar deviasi:** 3.85\n",
    "* **Rentang nilai rating:** dari **0** hingga **10**\n",
    "* **Median (50%) rating:** 0 → Menunjukkan bahwa sebagian besar pengguna memberikan rating **kosong** atau **default**\n",
    "\n",
    "**Informasi Tambahan:**\n",
    "\n",
    "* **ISBN terbanyak muncul:** `0971880107`, muncul sebanyak **2.502 kali**\n",
    "* **Jumlah ISBN unik:** 340.556\n",
    "\n",
    "Hasil ini memperlihatkan bahwa terdapat **sparsity** atau kelangkaan interaksi antara pengguna dan buku—ciri khas umum dalam dataset sistem rekomendasi.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JObrVvYX6NIu"
   },
   "source": [
    "**Mengecek jumlah missing value pada setiap kolom di data Ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 178
    },
    "id": "wUcDGsxC6K3Y",
    "outputId": "285e5623-ac9c-4176-fc23-adf7ce05f3e9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User-ID        0\n",
       "ISBN           0\n",
       "Book-Rating    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L-NH2_776R7B"
   },
   "source": [
    "Pada tahap ini, kami melakukan pengecekan terhadap jumlah *missing value* di setiap kolom dalam dataset **Ratings** menggunakan fungsi `.isnull().sum()`. Hasil pemeriksaan menunjukkan bahwa:\n",
    "\n",
    "* **User-ID**: 0 nilai yang hilang\n",
    "* **ISBN**: 0 nilai yang hilang\n",
    "* **Book-Rating**: 0 nilai yang hilang\n",
    "\n",
    "Dengan demikian, **dataset Ratings tidak memiliki missing value** dan siap untuk digunakan tanpa perlu dilakukan pembersihan terkait nilai yang hilang.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b-E_cIw26UrA"
   },
   "source": [
    "**Memeriksa jumlah duplikat di data Ratings**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "pEoyRmRv6SR6",
    "outputId": "e33e3001-6d2b-4a94-ca01-96ecb7727f1c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah duplikat di Ratings: 0\n"
     ]
    }
   ],
   "source": [
    "print('Jumlah duplikat di Ratings:', ratings.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RQT3L-JH6ZG7"
   },
   "source": [
    "Pada tahap ini, kami memeriksa kemungkinan adanya **data duplikat** dalam dataset **Ratings**. Mengingat bahwa dataset ini berisi data interaksi antara pengguna dan buku, keberadaan duplikat dapat mempengaruhi keakuratan model rekomendasi.\n",
    "\n",
    "**Hasil Pemeriksaan:**\n",
    "\n",
    "* **Jumlah duplikat:** 0\n",
    "* **Kesimpulan:** Tidak ditemukan data duplikat pada dataset **Ratings**, sehingga data dapat digunakan tanpa perlu penanganan lebih lanjut terkait duplikasi.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bHdj7TvP6dMA"
   },
   "source": [
    "## Dataset Users"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PogG6niI6hNE"
   },
   "source": [
    "**Menampilkan tampilan data users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "G3kgs1Cn6ab-",
    "outputId": "94083072-c7ff-4ba5-c99e-6867624e1b4a"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User-ID</th>\n",
       "      <th>Location</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>nyc, new york, usa</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>stockton, california, usa</td>\n",
       "      <td>18.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>moscow, yukon territory, russia</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>porto, v.n.gaia, portugal</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>farnborough, hants, united kingdom</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User-ID                            Location   Age\n",
       "0        1                  nyc, new york, usa   NaN\n",
       "1        2           stockton, california, usa  18.0\n",
       "2        3     moscow, yukon territory, russia   NaN\n",
       "3        4           porto, v.n.gaia, portugal  17.0\n",
       "4        5  farnborough, hants, united kingdom   NaN"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "users.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Z4wQ9zC_6lYh"
   },
   "source": [
    "Tahap ini dilakukan untuk memahami struktur dan konten awal data pengguna dengan menampilkan lima baris pertama menggunakan fungsi `.head()`.\n",
    "\n",
    "**Informasi dari Lima Pengguna Pertama:**\n",
    "\n",
    "1. **User-ID 1**: Lokasi “nyc, new york, usa”, usia **tidak diketahui** *(NaN)*\n",
    "2. **User-ID 2**: Lokasi “stockton, california, usa”, usia **18 tahun**\n",
    "3. **User-ID 3**: Lokasi “moscow, yukon territory, russia”, usia **tidak diketahui** *(NaN)*\n",
    "4. **User-ID 4**: Lokasi “porto, v.n.gaia, portugal”, usia **17 tahun**\n",
    "5. **User-ID 5**: Lokasi “farnborough, hants, united kingdom”, usia **tidak diketahui** *(NaN)*\n",
    "\n",
    "**Insight:**\n",
    "\n",
    "* Kolom **Location** memuat informasi geografis pengguna (kota, negara bagian, negara).\n",
    "* Kolom **Age** menunjukkan banyak nilai kosong (*missing*), yang perlu diperhatikan pada tahap *data cleaning* selanjutnya.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hsgAaWj86oF8"
   },
   "source": [
    "**Menampilkan jumlah baris dan kolom dari data users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "P8PPbTiD6l3n",
    "outputId": "ff8463e2-fb02-48d8-c710-207b763ce072"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Users: (278858, 3)\n"
     ]
    }
   ],
   "source": [
    "print(f\"Jumlah data Users: {users.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iYprZ8po6t_1"
   },
   "source": [
    "Terdapat data rating dengan shape 278.858 baris, 3 kolom"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "w1-vJgw-6vp5"
   },
   "source": [
    "**Menampilkan struktur data users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "1ogFqMLA6uRN",
    "outputId": "6f151375-7662-4276-a75b-8c65c05abafe"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 278858 entries, 0 to 278857\n",
      "Data columns (total 3 columns):\n",
      " #   Column    Non-Null Count   Dtype  \n",
      "---  ------    --------------   -----  \n",
      " 0   User-ID   278858 non-null  int64  \n",
      " 1   Location  278858 non-null  object \n",
      " 2   Age       168096 non-null  float64\n",
      "dtypes: float64(1), int64(1), object(1)\n",
      "memory usage: 6.4+ MB\n"
     ]
    }
   ],
   "source": [
    "users.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yjhe6FUd63mU"
   },
   "source": [
    "**Uraian Variabel** – `Users.csv`\n",
    "\n",
    "* **User-ID**: ID unik yang mengidentifikasi setiap pengguna. Bertipe data *int64*, dengan total **168.096** data yang lengkap.\n",
    "* **Location**: Informasi lokasi pengguna yang mencakup **kota, provinsi, dan negara**. Bertipe *object* dan tidak memiliki data yang hilang (**168.096 entri**).\n",
    "* **Age**: Usia pengguna, disimpan dalam tipe data *float64*. Seluruhnya berjumlah **168.096**, meskipun kemungkinan terdapat nilai yang tidak valid seperti 0 atau lebih dari 100.\n",
    "\n",
    "\n",
    "---\n",
    "Tahap ini bertujuan untuk menampilkan struktur dataset **Users** menggunakan fungsi `.info()`. Dataset ini memuat **278.858 entri** dengan tiga kolom utama: **User-ID**, **Location**, dan **Age**.\n",
    "\n",
    "**Insight:**\n",
    "\n",
    "Ditemukan **missing value** pada kolom **Age**, yang terlihat dari jumlah nilai non-null yang lebih sedikit dibandingkan total baris data. Hal ini menunjukkan perlunya penanganan khusus terhadap kolom usia dalam tahap *data preparation* berikutnya.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8V3mgns768dF"
   },
   "source": [
    "**Menampilkan statistik ringkasan untuk data users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "btwHOytX66Fx",
    "outputId": "ab7ca647-0d85-4ad8-aeb7-f38616fd3073"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User-ID</th>\n",
       "      <th>Location</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>278858.00000</td>\n",
       "      <td>278858</td>\n",
       "      <td>168096.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>NaN</td>\n",
       "      <td>57339</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>NaN</td>\n",
       "      <td>london, england, united kingdom</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>NaN</td>\n",
       "      <td>2506</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>139429.50000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>34.751434</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>80499.51502</td>\n",
       "      <td>NaN</td>\n",
       "      <td>14.428097</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.00000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>69715.25000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>24.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>139429.50000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>32.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>209143.75000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>44.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>278858.00000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>244.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             User-ID                         Location            Age\n",
       "count   278858.00000                           278858  168096.000000\n",
       "unique           NaN                            57339            NaN\n",
       "top              NaN  london, england, united kingdom            NaN\n",
       "freq             NaN                             2506            NaN\n",
       "mean    139429.50000                              NaN      34.751434\n",
       "std      80499.51502                              NaN      14.428097\n",
       "min          1.00000                              NaN       0.000000\n",
       "25%      69715.25000                              NaN      24.000000\n",
       "50%     139429.50000                              NaN      32.000000\n",
       "75%     209143.75000                              NaN      44.000000\n",
       "max     278858.00000                              NaN     244.000000"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "users.describe(include='all')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Gi7pAUnz7Ahi"
   },
   "source": [
    "Pada tahap ini, dilakukan analisis statistik ringkas terhadap dataset **Users** menggunakan fungsi `.describe(include='all')`.\n",
    "\n",
    "* **Rata-rata usia pengguna** adalah sekitar **34,75 tahun**.\n",
    "* **Usia terendah** tercatat **0 tahun**, sementara **usia tertinggi** mencapai **244 tahun**, yang kemungkinan besar merupakan data tidak valid.\n",
    "* Lokasi yang paling sering dicantumkan pengguna adalah **\"london, england, united kingdom\"**, dengan total **2.506 pengguna**.\n",
    "* Total terdapat **57.339 lokasi unik** dalam dataset.\n",
    "\n",
    "**Insight**:\n",
    "\n",
    "* Nilai usia ekstrem seperti 0 dan 244 perlu ditinjau lebih lanjut dan mungkin harus **dibersihkan atau diimputasi** pada tahap **data preparation**.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MTm7A8ta7NsM"
   },
   "source": [
    "**Mengecek jumlah missing value pada setiap kolom di data Users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 178
    },
    "id": "fIwAEJOH7G5L",
    "outputId": "835b4894-fefb-49be-8584-a0ba64a08b45"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User-ID          0\n",
       "Location         0\n",
       "Age         110762\n",
       "dtype: int64"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "users.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MHu3hu5r85I4"
   },
   "source": [
    "Pada tahap ini, dilakukan pemeriksaan terhadap jumlah nilai yang hilang (missing value) di setiap kolom pada dataset *Users* dengan menggunakan fungsi `.isnull().sum()`.\n",
    "\n",
    "**Hasil pemeriksaan menunjukkan:**\n",
    "\n",
    "* **User-ID:** Tidak terdapat nilai yang hilang\n",
    "* **Location:** Tidak terdapat nilai yang hilang\n",
    "* **Age:** Terdapat sekitar **110.762** entri yang kosong\n",
    "\n",
    "Dengan demikian, sekitar **40%** data pada kolom **Age** tidak tersedia. Hal ini menunjukkan perlunya penerapan strategi penanganan data hilang, seperti menggunakan teknik **imputasi** atau melakukan **penghapusan data** yang tidak lengkap, agar kualitas analisis dan performa model tetap optimal.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fhV5_FPT9Bta"
   },
   "source": [
    "**Memeriksa jumlah buku unik dan jumlah duplikat di data users**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "opS0zpvk887e",
    "outputId": "4078d82a-ece4-4df3-c5f7-9b04207829d7"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah id User unik: 105283\n",
      "Duplikat di Users: 0\n"
     ]
    }
   ],
   "source": [
    "print('Jumlah id User unik:', ratings['User-ID'].nunique())\n",
    "print('Duplikat di Users:', users.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nDckaCbr9Gn5"
   },
   "source": [
    "**Jumlah id User Unik**\n",
    "\n",
    "Pada tahap ini, dilakukan penghitungan jumlah pengguna unik pada dataset **Ratings** menggunakan fungsi `.nunique()` pada kolom **User-ID**.\n",
    "**Hasilnya:** Terdapat **105.283** pengguna unik, yang menunjukkan banyaknya individu berbeda yang memberikan penilaian terhadap buku.\n",
    "\n",
    "---\n",
    "\n",
    "**Pemeriksaan Duplikasi pada Dataset Users**\n",
    "\n",
    "Untuk memastikan kualitas data, dilakukan pengecekan terhadap duplikasi pada dataset **Users**.\n",
    "**Hasilnya:** Tidak ditemukan data duplikat (jumlah duplikat = 0), sehingga data pengguna dapat digunakan langsung tanpa perlu pembersihan lebih lanjut pada aspek ini.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SDOojp4c9K_H"
   },
   "source": [
    "## Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 564
    },
    "id": "hcBF27t99G-U",
    "outputId": "28dd7678-b97b-45ea-f495-f21b02eebad0"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABCsAAAIjCAYAAAAqSJEGAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8ekN5oAAAACXBIWXMAAA9hAAAPYQGoP6dpAACbcElEQVR4nOzdeVxV1f7/8fdhBhFERMAR5ykVw1S0HBLFnMvS1ELJIcuZezUxZ80hzag0zVKzUnPWMidyLskxNCstx64DOIuCAsL5/eGP/fUIKJTdc87t9Xw8eNhZ+7PX/uy9l+n5uPbaJrPZbBYAAAAAAICNcLB2AgAAAAAAAPeiWAEAAAAAAGwKxQoAAAAAAGBTKFYAAAAAAACbQrECAAAAAADYFIoVAAAAAADAplCsAAAAAAAANoViBQAAAAAAsCkUKwAAAAAAgE2hWAEAwCO2bds2mUwmLV++3GrH3rZt23/92P8k3bt3V1BQkLXT+Ft8+umnMplM2rdvn7VT+csexe/FPXv2yMXFRadPn36Emf19hg0bprp161o7DQD4yyhWAABszs2bNzV69Gi1aNFChQsXlslk0qeffppr/K+//qoWLVrI09NThQsX1ssvv6yLFy/m6Vgmk8nip0CBAqpataomTJiglJSUR3RG+LuNGTPG4j46OzsrKChIAwYM0LVr1/5Un+fOndOYMWMUHx//SHP9s7KKCA/7+V8toljLm2++qc6dO6t06dJGW+PGjfXYY49ZxKWlpem9995TrVq15OXlpUKFCqlatWrq3bu3jhw5YsTldB+LFi2qJk2aaP369dmOnxXzzjvvZNuWU2Fp0KBBOnjwoL766qtHcfoAYDVO1k4AAID7Xbp0SePGjVOpUqVUs2bNB84SOHPmjBo2bChvb29NnDhRN2/e1LRp0/TTTz8Z/yL6MM2aNVNERISku4WSnTt3auTIkTp48KCWLVv2qE4L/wWzZs2Sp6enkpOTtXnzZn3wwQc6cOCAvvvuu3z3de7cOY0dO1ZBQUEKDg622Pbxxx8rMzPzEWWdNw0bNtTnn39u0dazZ0/VqVNHvXv3Nto8PT3/q3n9L4uPj9e3336rXbt2PTS2Q4cOWr9+vTp37qxevXopPT1dR44c0dq1a1W/fn1VrlzZIn7cuHEqU6aMzGazEhMT9emnn6ply5b6+uuv1bp162z9T506Va+99po8PDwemEdAQIDatWunadOmqW3btvk7YQCwIRQrAAA2JzAwUOfPn1dAQID27dunJ554ItfYiRMnKjk5Wfv371epUqUkSXXq1FGzZs306aefWnyJy03FihX10ksvGZ/79OmjtLQ0rVy5Urdv35abm9tfPyn8Vzz//PMqUqSIJOnVV1/Viy++qCVLlmjPnj2qU6fOIzuOs7PzI+srr8qWLauyZctatPXp00dly5a1GL9/1u3bt/NU3PsnmT9/vkqVKqV69eo9MG7v3r1au3at3nrrLQ0fPtxi24wZM3Kc3fPMM8+odu3axucePXrI399fixcvzlasCA4OVnx8vGbPnq2oqKiH5t2xY0e98MILOnHiRLYxAwD2gsdAAAA2x9XVVQEBAXmKXbFihVq3bm0UKiQpLCxMFStW1NKlS/90DgEBATKZTHJy+r+6flBQkLp3754ttnHjxmrcuPED+0tNTVXr1q3l7e2tXbt26dSpU7k+3mIymTRmzJiH5njmzBm1b99eBQoUUNGiRTV48GClpqbmGLt79261aNFC3t7e8vDwUKNGjfT9999bxGQ9SnHs2DF1795dhQoVkre3tyIjI7M9EnPr1i0NGDBARYoUUcGCBdW2bVudPXs2W+6nT5/W66+/rkqVKsnd3V2+vr564YUXdOrUKYv+sqazf//994qKipKfn58KFCigZ599Ns+P9OTkqaeekiQdP37caLty5Yr+/e9/q3r16vL09JSXl5eeeeYZHTx40IjZtm2bUSSLjIw0puJn3a/716zIup/Tpk3TnDlzVK5cObm6uuqJJ57Q3r17s+W1bNkyVa1aVW5ubnrssce0atWqR7YOxtmzZ/XKK6/I399frq6uqlatmubNm2cRk7WWw5dffqkRI0aoePHi8vDwUFJSkhGTkpKiV199Vb6+vvLy8lJERISuXr1q0c+aNWvUqlUrFStWTK6uripXrpzGjx+vjIwMi7isxyZ++eUXNWnSRB4eHipevLjefvttI+bmzZsqUKCABg4cmO2czpw5I0dHR02aNElS3u5hbu7/vfggq1ev1tNPPy2TyfTAuKzx1aBBg2zbHB0d5evr+9C8ChUqJHd3d4v/52Rp0KCBnn76ab399tu6devWQ/sKCwuTdPf+AIC9YmYFAMBunT17VhcuXLD418ksderU0bp16/LUz+3bt3Xp0iVJUnJysr7//nstWLBAXbp0yfGLQ37dunVL7dq10759+/Ttt9/qiSeeyPZl/c/02bRpU/3xxx8aMGCAihUrps8//1xbtmzJFrtlyxY988wzCgkJ0ejRo+Xg4KD58+fr6aef1s6dO7PNOOjYsaPKlCmjSZMm6cCBA/rkk09UtGhRTZkyxYjp3r27li5dqpdffln16tXT9u3b1apVq2zH3rt3r3bt2qUXX3xRJUqU0KlTpzRr1iw1btxYv/zyS7Yp7f3795ePj49Gjx6tU6dOKSYmRv369dOSJUv+1HXKus4+Pj5G24kTJ7R69Wq98MILKlOmjBITE/XRRx+pUaNG+uWXX1SsWDFVqVJF48aN06hRo9S7d2+j6FG/fv0HHm/RokW6ceOGXn31VZlMJr399tt67rnndOLECWM2xjfffKNOnTqpevXqmjRpkq5evaoePXqoePHif+oc75WYmKh69erJZDKpX79+8vPz0/r169WjRw8lJSVp0KBBFvHjx4+Xi4uL/v3vfys1NdViZkW/fv1UqFAhjRkzRkePHtWsWbN0+vRpo9Ah3S0yeXp6KioqSp6entqyZYtGjRqlpKQkTZ061eJYV69eVYsWLfTcc8+pY8eOWr58ud544w1Vr15dzzzzjDw9PfXss89qyZIlmj59uhwdHY19Fy9eLLPZrK5du0rK2z3MSU6/F3Nz9uxZ/fHHH3r88ccfet2z1rNYuHChGjRokKf/b1y/fl2XLl2S2WzWhQsX9MEHH+jmzZu5zpIZM2aMGjZsqFmzZj10doW3t7fKlSun77//XoMHD35oLgBgk8wAANiwvXv3miWZ58+fn+u2zz77LNu2IUOGmCWZb9++/cD+JeX40759+2z7li5d2tytW7dsfTRq1MjcqFEj4/PWrVvNkszLli0z37hxw9yoUSNzkSJFzD/++KMRc/LkyVzPS5J59OjRD8w7JibGLMm8dOlSoy05Odlcvnx5syTz1q1bzWaz2ZyZmWmuUKGCOTw83JyZmWnEpqSkmMuUKWNu1qyZ0TZ69GizJPMrr7xicaxnn33W7Ovra3zev3+/WZJ50KBBFnHdu3fPlntKSkq23OPi4rLdt/nz55slmcPCwizyHDx4sNnR0dF87dq1B16PrNyPHj1qvnjxovnUqVPmefPmmd3d3c1+fn7m5ORkI/b27dvmjIwMi/1PnjxpdnV1NY8bN85oe9DY69atm7l06dIW+0sy+/r6mq9cuWK0r1mzxizJ/PXXXxtt1atXN5coUcJ848YNo23btm1mSRZ95kWBAgUsxmSPHj3MgYGB5kuXLlnEvfjii2Zvb2/jfmSN0bJly2a7R1n3IiQkxJyWlma0v/3222ZJ5jVr1hhtOd3fV1991ezh4WHx+6dRo0bZ7nlqaqo5ICDA3KFDB6Nt48aNZknm9evXW/RZo0YNi99jeb2Hefm9mJtvv/02272793yqVatmfM7MzDTO0d/f39y5c2fzzJkzzadPn862b9b1vf/H1dXV/Omnn2aLl2Tu27ev2Ww2m5s0aWIOCAgwrntWX3v37s22X/Pmzc1VqlR56HkCgK3iMRAAgN3Kmg7t6uqabVvWOhN5mTLdrl07xcbGKjY2VmvWrFF0dLQ2bNigLl26yGw2/+n8rl+/rubNm+vIkSPatm1btkUa/4p169YpMDBQzz//vNHm4eGRbY2O+Ph4/f777+rSpYsuX76sS5cu6dKlS0pOTlbTpk21Y8eObAtF9unTx+LzU089pcuXLxuPCGzYsEGS9Prrr1vE9e/fP1ue7u7uxn+np6fr8uXLKl++vAoVKqQDBw5ki+/du7fFlPunnnpKGRkZeX5tZKVKleTn56egoCC98sorKl++vNavX28xg8PV1VUODnf/CpSRkaHLly/L09NTlSpVyjGn/OjUqZPFLI6sGRknTpyQdHfRzp9++kkREREWC2E2atRI1atX/0vHNpvNWrFihdq0aSOz2Wzc60uXLik8PFzXr1/Pdn7dunWzuEf36t27t8XaHK+99pqcnJwsZizdu++NGzd06dIlPfXUU0pJSbF4A4Z0d+HPe2cNuLi4qE6dOsa1ke4+vlCsWDEtXLjQaDt8+LAOHTpksW9+7+Gf+b14+fJlSZazcnJjMpm0ceNGTZgwQT4+Plq8eLH69u2r0qVLq1OnTjmuWTFz5kzj/ztffPGFmjRpop49e2rlypW5HmfMmDFKSEjQ7NmzH5qTj4+PMWMMAOwRj4EAAOxW1helnNZpuH37tkXMg5QoUcJ4xluS2rZtK19fX/373//W2rVr1aZNmz+V36BBg3T79m39+OOPqlat2p/qIzenT59W+fLlsz1LX6lSJYvPv//+u6S7X0pzc/36dYsvZPeu/yH935e1q1evysvLS6dPn5aDg4PKlCljEVe+fPlsfd+6dUuTJk3S/PnzdfbsWYviz/Xr17PFP+jYebFixQp5eXnp4sWLev/993Xy5MlsYyAzM1PvvfeePvzwQ508edJifYW8rC3wIA/LP6voktO1Kl++/F8qlly8eFHXrl3TnDlzNGfOnBxjLly4YPH5/nt4rwoVKlh89vT0VGBgoMUjTD///LNGjBihLVu2WKx3IWW/vyVKlMg2Xn18fHTo0CHjs4ODg7p27apZs2YpJSVFHh4eWrhwodzc3PTCCy8Ycfm9h3/l92JeC5aurq5688039eabb+r8+fPavn273nvvPS1dulTOzs764osvLOLr1Klj8Qhb586dVatWLfXr10+tW7fOcbHThg0bqkmTJnr77bezFRVzyvtha20AgC1jZgUAwG4FBgZKks6fP59t2/nz51W4cOEcZ13kRdOmTSVJO3bsMNpy+4v//YsJZmnXrp3MZrMmT56cbfZCfvv6s7KOO3XqVONfce//uf9Vl/euFXCvPzPLpH///nrrrbfUsWNHLV26VJs2bVJsbKx8fX1zfPXnXz12w4YNFRYWps6dOys2Nlbu7u7q2rWrxbEmTpyoqKgoNWzYUF988YU2btyo2NhYVatW7S+/jvRRXrv8ysr9pZdeyvVe378AZF6Kebm5du2aGjVqpIMHD2rcuHH6+uuvFRsba6xtcv+1zOu1iYiI0M2bN7V69WqZzWYtWrTIWBAzS37v4YN+L+Ymq+iR10LZvQIDA/Xiiy9qx44dqlChgpYuXao7d+48cB8HBwc1adJE58+fN4qMORk9erQSEhL00UcfPbC/q1evGm/GAQB7xMwKAIDdKl68uPz8/LRv375s2/bs2fOXHrvI+mJx8+ZNo83HxyfH6dynT5/O8fWA7du3V/PmzdW9e3cVLFhQs2bNsuhLUrb+8vq4Q+nSpXX48OFs/3p69OhRi7hy5cpJkry8vCxmj/wVpUuXVmZmpk6ePGnxr+/Hjh3LFrt8+XJ169ZN77zzjtF2+/btHK/jo+bp6anRo0crMjJSS5cu1Ysvvmjk1KRJE82dO9ci/tq1axZf7v6Of5XOWogxp2uVU1t++Pn5qWDBgsrIyHgk9/r3339XkyZNjM83b97U+fPn1bJlS0l33yhy+fJlrVy5Ug0bNjTiTp48+ZeO+9hjj6lWrVpauHChSpQooT/++EMffPCBRUxe72GWB/1ezE3lypUl/bXzcXZ2Vo0aNfT777/r0qVLD33LUU7/37lfo0aN1LhxY02ZMkWjRo3KNe7kyZOqWbPmn0scAGwAMysAAHatQ4cOWrt2rf7zn/8YbZs3b9Zvv/1mMW08v77++mtJsvjLfrly5fTDDz8oLS3NaLv/2PeLiIjQ+++/r9mzZ+uNN94w2r28vFSkSBGLmRuS9OGHH+Ypv5YtW+rcuXNavny50ZaSkpJt+n9ISIjKlSunadOm5fgF6M+8FjQ8PDzHXO//Qind/df0+//l/IMPPnjkM0hy07VrV5UoUcLiTSY55bRs2TKdPXvWoq1AgQKSsheU/opixYrpscce02effWZxP7Zv366ffvrpL/Xt6OioDh06aMWKFTp8+HC27fm913PmzFF6errxedasWbpz546eeeYZ43iS5cyItLS0PI/hB3n55Ze1adMmxcTEyNfX1zhmlrzew3vl9nsxN8WLF1fJkiVzLIbe7/fff9cff/yRrf3atWuKi4uTj4+P/Pz8HthHenq6Nm3aJBcXF1WpUuWBsVlrV+T2uM/169d1/Pjxh769BgBsGTMrAAA2acaMGbp27ZrOnTsn6W7x4MyZM5LuPlqQNSV8+PDhWrZsmZo0aaKBAwfq5s2bmjp1qqpXr67IyMg8Heu3334znidPSUnRDz/8oAULFqh8+fJ6+eWXjbiePXtq+fLlatGihTp27Kjjx4/riy++MGYv5KZfv35KSkrSm2++KW9vbw0fPtzob/LkyerZs6dq166tHTt26LfffstTzr169dKMGTMUERGh/fv3KzAwUJ9//nm2V4E6ODjok08+0TPPPKNq1aopMjJSxYsX19mzZ7V161Z5eXkZhZm8CgkJUYcOHRQTE6PLly8bry7Nyv3eGQmtW7fW559/Lm9vb1WtWlVxcXH69ttv//LaEHnl7OysgQMHasiQIdqwYYNatGih1q1ba9y4cYqMjFT9+vX1008/aeHChdlmx5QrV06FChXS7NmzVbBgQRUoUEB169Z94DoPeTFx4kS1a9dODRo0UGRkpK5evaoZM2bosccee+C/qOfF5MmTtXXrVtWtW1e9evVS1apVdeXKFR04cEDffvutrly5kue+0tLS1LRpU3Xs2FFHjx7Vhx9+qCeffFJt27aVdPc1rj4+PurWrZsGDBggk8mkzz///JE88tKlSxcNHTpUq1at0muvvWax0KekPN/D++X2ezE37dq106pVqx66/sPBgwfVpUsXPfPMM3rqqadUuHBhnT17VgsWLNC5c+cUExOT7TGY9evXG4uQXrhwQYsWLdLvv/+uYcOGycvL64F5NWrUSI0aNdL27dtz3P7tt9/KbDarXbt2D+wHAGzaf/XdIwAA5FHp0qVzfa3oyZMnLWIPHz5sbt68udnDw8NcqFAhc9euXc0JCQl5Os79fTs6OppLlChh7t27tzkxMTFb/DvvvGMuXry42dXV1dygQQPzvn37Hvjq0nsNHTrULMk8Y8YMs9l897WPPXr0MHt7e5sLFixo7tixo/nChQt5enWp2Ww2nz592ty2bVuzh4eHuUiRIuaBAweaN2zYYPHq0iw//vij+bnnnjP7+vqaXV1dzaVLlzZ37NjRvHnzZiMm6/WfFy9etNg36/WI91735ORkc9++fc2FCxc2e3p6mtu3b28+evSoWZJ58uTJRtzVq1fNkZGR5iJFipg9PT3N4eHh5iNHjmR7DWxur2DMupb3n8/9csvdbDabr1+/bvb29jbu0e3bt83/+te/zIGBgWZ3d3dzgwYNzHFxcdnuo9l899WjVatWNTs5OVm8xjS3V5dOnTo12/Fzup9ffvmluXLlymZXV1fzY489Zv7qq6/MHTp0MFeuXPmB53m/+19dajabzYmJiea+ffuaS5YsaXZ2djYHBASYmzZtap4zZ44Rk9sYNZv/715s377d3Lt3b7OPj4/Z09PT3LVrV/Ply5ctYr///ntzvXr1zO7u7uZixYqZhw4darx+9N57dv+rPrPcfx3v1bJlS7Mk865du7Jty+s9zOvvxdwcOHDALMm8c+dOi/b7zycxMdE8efJkc6NGjcyBgYFmJycns4+Pj/npp582L1++3GLfnF5d6ubmZg4ODjbPmjXL4tW9ZrPlq0vvlXVuOf2+6dSpk/nJJ5984LkBgK0zmc3/hRWfAADA/7z4+HjVqlVLX3zxhbp27WrtdOxOcHCw/Pz8FBsba+1UbMKzzz6rn3766S+v5fFXNW3aVMWKFdPnn39u1TzyKiEhQWXKlNGXX37JzAoAdo01KwAAQL7dunUrW1tMTIwcHBwsFltEdunp6dneDLFt2zYdPHhQjRs3tk5SNub8+fP65ptvLB7DspaJEydqyZIleV781tpiYmJUvXp1ChUA7B4zKwAAQL6NHTtW+/fvV5MmTeTk5KT169dr/fr16t2790NfqfhPd+rUKYWFhemll15SsWLFdOTIEc2ePVve3t46fPjwf209D1t08uRJff/99/rkk0+0d+9eHT9+/KFv0AAA/G9igU0AAJBv9evXV2xsrMaPH6+bN2+qVKlSGjNmjN58801rp2bzfHx8FBISok8++UQXL15UgQIF1KpVK02ePPkfXaiQ7r4VJTIyUqVKldKCBQsoVADAPxgzKwAAAAAAgE1hzQoAAAAAAGBTKFYAAAAAAACbwpoV/2CZmZk6d+6cChYsKJPJZO10AAAAAAD/48xms27cuKFixYrJwSH3+RMUK/7Bzp07p5IlS1o7DQAAAADAP8x//vMflShRItftFCv+wQoWLCjp7mvCChcubOVsgPxLT0/Xpk2b1Lx5czk7O1s7HSDfGMOwd4xh2DvGMOydPY7hpKQklSxZ0vg+mhuKFf9gWY9+FCxYUF5eXlbOBsi/9PR0eXh4yMvLy27+5wzcizEMe8cYhr1jDMPe2fMYfthSBCywCQAAAAAAbAozK6AWE5Yr08nd2mkA+ebiKA0L9VbDkYuVlmHtbID8YwzD3jGGYe8Yw7B3WWP4fxEzKwAAAAAAgE2hWAEAAAAAAGwKxQoAAAAAAGBTKFYAAAAAAACbQrECAAAAAADYFIoVAAAAAADAplCsAAAAAAAANoViBQAAAAAAsCkUKwAAAAAAgE2hWAEAAAAAAGwKxQoAAAAAAGBTKFYAAAAAAACbQrECAAAAAADYFIoVAAAAAADApthEsWLmzJkKCgqSm5ub6tatqz179hjbjh8/rmeffVZ+fn7y8vJSx44dlZiYaGzftm2bTCZTjj979+6VJN2+fVvdu3dX9erV5eTkpPbt2+eYx8KFC1WzZk15eHgoMDBQr7zyii5fvmwRExMTo0qVKsnd3V0lS5bU4MGDdfv27TyfjyTNmTNHjRs3lpeXl0wmk65du5YtlytXrqhr167y8vJSoUKF1KNHD928edPYPmbMmBzPuUCBAnm65gAAAAAA2CqrFyuWLFmiqKgojR49WgcOHFDNmjUVHh6uCxcuKDk5Wc2bN5fJZNKWLVv0/fffKy0tTW3atFFmZqYkqX79+jp//rzFT8+ePVWmTBnVrl1bkpSRkSF3d3cNGDBAYWFhOebx/fffKyIiQj169NDPP/+sZcuWac+ePerVq5cRs2jRIg0bNkyjR4/Wr7/+qrlz52rJkiUaPnx4ns4nS0pKilq0aGGx3/26du2qn3/+WbGxsVq7dq127Nih3r17G9v//e9/ZzvvqlWr6oUXXvhzNwIAAAAAABvhZO0Epk+frl69eikyMlKSNHv2bH3zzTeaN2+eHn/8cZ06dUo//vijvLy8JEkLFiyQj4+PtmzZorCwMLm4uCggIMDoLz09XWvWrFH//v1lMpkkSQUKFNCsWbMk3S1K5DSTIS4uTkFBQRowYIAkqUyZMnr11Vc1ZcoUI2bXrl1q0KCBunTpIkkKCgpS586dtXv37jydz7BhwyRJgwYNknR3VkhOfv31V23YsEF79+41Ci4ffPCBWrZsqWnTpqlYsWLy9PSUp6ensc/Bgwf1yy+/aPbs2Xm46gAAAAAA2C6rFivS0tK0f/9+RUdHG20ODg4KCwtTXFycqlWrJpPJJFdXV2O7m5ubHBwc9N133+U4S+Krr77S5cuXjWJBXoWGhmr48OFat26dnnnmGV24cEHLly9Xy5YtjZj69evriy++0J49e1SnTh2dOHFC69at08svv5yn88mruLg4FSpUyChUSFJYWJgcHBy0e/duPfvss9n2+eSTT1SxYkU99dRTufabmpqq1NRU43NSUpIkycVRynTMc3qAzXBxtPwVsDeMYdg7xjDsHWMY9i5r7Kanp1s3kXzIa65WLVZcunRJGRkZ8vf3t2j39/fXkSNHVK9ePRUoUEBvvPGGJk6cKLPZrGHDhikjI0Pnz5/Psc+5c+cqPDxcJUqUyFcuDRo00MKFC9WpUyfdvn1bd+7cUZs2bTRz5kwjpkuXLrp06ZKefPJJmc1m3blzR3369DEe53jY+eRVQkKCihYtatHm5OSkwoULKyEhIVv87du3tXDhQmPmRm4mTZqksWPHZmvvF+IlDw+PPOcH2JqoOt7WTgH4SxjDsHeMYdg7xjDsXWxsrLVTyLOUlJQ8xVn9MZAH8fPz07Jly/Taa6/p/fffl4ODgzp37qzHH39cDg7Zl9s4c+aMNm7cqKVLl+b7WL/88osGDhyoUaNGKTw8XOfPn9eQIUPUp08fzZ07V9LdxzYmTpyoDz/8UHXr1tWxY8c0cOBAjR8/XiNHjvzL5/tnrVq1Sjdu3FC3bt0eGBcdHa2oqCjjc1JSkkqWLKkZ+5OU6WQ/lTggi4vj3b9cTN9zXWkZ1s4GyD/GMOwdYxj2jjEMe5c1hps1ayZnZ2drp5MnWTP8H8aqxYoiRYrI0dHR4u0ekpSYmGisQ9G8eXMdP35cly5dkpOTkwoVKqSAgACVLVs2W3/z58+Xr6+v2rZtm+9cJk2apAYNGmjIkCGSpBo1aqhAgQJ66qmnNGHCBAUGBmrkyJF6+eWX1bNnT0lS9erVlZycrN69e+vNN9/M0/nkRUBAgMWCnJJ0584dXblyJcd+PvnkE7Vu3TrbjI77ubq6WjxSkyUtQ8o05Tk9wOakZYi/YMCuMYZh7xjDsHeMYdg7Z2dnuylW5DVPq74NxMXFRSEhIdq8ebPRlpmZqc2bNys0NNQitkiRIipUqJC2bNmiCxcuZCtImM1mzZ8/XxEREX/qJqWkpGSbreHo6Gj0nZeY/JzPg4SGhuratWvav3+/0bZlyxZlZmaqbt26FrEnT57U1q1b1aNHjzz3DwAAAACALbP6YyBRUVHq1q2bateurTp16igmJkbJycnGApnz589XlSpV5Ofnp7i4OA0cOFCDBw9WpUqVLPrZsmWLTp48acx6uN8vv/yitLQ0XblyRTdu3FB8fLwkKTg4WJLUpk0b9erVS7NmzTIeAxk0aJDq1KmjYsWKGTHTp09XrVq1jMdARo4cqTZt2hhFi4edj3R3TYqEhAQdO3ZMkvTTTz+pYMGCKlWqlAoXLqwqVaqoRYsW6tWrl2bPnq309HT169dPL774opFLlnnz5ikwMFDPPPPMX7sRAAAAAADYCKsXKzp16qSLFy9q1KhRSkhIUHBwsDZs2GA80nD06FFFR0frypUrCgoK0ptvvqnBgwdn62fu3LmqX7++KleunONxWrZsqdOnTxufa9WqJen/Zk10795dN27c0IwZM/Svf/1LhQoV0tNPP23x6tIRI0bIZDJpxIgROnv2rPz8/NSmTRu99dZbeT4f6e7rTO9d6LJhw4aS7hZmunfvLklauHCh+vXrp6ZNm8rBwUEdOnTQ+++/b3FOmZmZ+vTTT9W9e3ejWAIAAAAAgL0zmbO+reMfJykpSd7e3qo9aLYyndytnQ6Qby6O0rBQb02OY1Es2CfGMOwdYxj2jjEMe5c1hlu2bGk3a1ZkfQ+9fv26vLy8co2z6poVAAAAAAAA96NYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJtCsQIAAAAAANgUihUAAAAAAMCmUKwAAAAAAAA2hWIFAAAAAACwKRQrAAAAAACATaFYAQAAAAAAbArFCgAAAAAAYFOcrJ0ArG/DiOfl6+tr7TSAfEtPT9e6deu0Y3xnOTs7WzsdIN8Yw7B3jGHYO8Yw7F3WGP5fxMwKAAAAAABgUyhWAAAAAAAAm0KxAgAAAAAA2BSKFQAAAAAAwKZQrAAAAAAAADaFYgUAAAAAALApFCsAAAAAAIBNoVgBAAAAAABsCsUKAAAAAABgU5ysnQCsr8WE5cp0crd2GkC+uThKw0K91XDkYqVlWDsbIP8Yw7Zv/9QIa6cAAMA/EjMrAAAAAACATaFYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJtCsQIAAAAAANgUihUAAAAAAMCmUKwAAAAAAAA2hWIFAAAAAACwKRQrAAAAAACATaFYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJtCsQIAAAAAANgUmyhWzJw5U0FBQXJzc1PdunW1Z88eY9urr76qcuXKyd3dXX5+fmrXrp2OHDlisf+AAQMUEhIiV1dXBQcH53gMs9msadOmqWLFinJ1dVXx4sX11ltvWcSkpqbqzTffVOnSpeXq6qqgoCDNmzfPIubatWvq27evAgMD5erqqooVK2rdunUWMWfPntVLL70kX19fubu7q3r16tq3b5+xfcyYMapcubIKFCggHx8fhYWFaffu3TnmnZqaquDgYJlMJsXHxxvtR48eVZMmTeTv7y83NzeVLVtWI0aMUHp6eq7XGQAAAAAAe+Bk7QSWLFmiqKgozZ49W3Xr1lVMTIzCw8N19OhRFS1aVCEhIeratatKlSqlK1euaMyYMWrevLlOnjwpR0dHo59XXnlFu3fv1qFDh3I8zsCBA7Vp0yZNmzZN1atX15UrV3TlyhWLmI4dOyoxMVFz585V+fLldf78eWVmZhrb09LS1KxZMxUtWlTLly9X8eLFdfr0aRUqVMiIuXr1qho0aKAmTZpo/fr18vPz0++//y4fHx8jpmLFipoxY4bKli2rW7du6d1331Xz5s117Ngx+fn5WeQ0dOhQFStWTAcPHrRod3Z2VkREhB5//HEVKlRIBw8eVK9evZSZmamJEyfm+z4AAAAAAGArrF6smD59unr16qXIyEhJ0uzZs/XNN99o3rx5GjZsmHr37m3EBgUFacKECapZs6ZOnTqlcuXKSZLef/99SdLFixdzLFb8+uuvmjVrlg4fPqxKlSpJksqUKWMRs2HDBm3fvl0nTpxQ4cKFjePda968ebpy5Yp27dolZ2fnHGOmTJmikiVLav78+Ubb/cfq0qVLtmswd+5cHTp0SE2bNjXa169fr02bNmnFihVav369xT5ly5ZV2bJljc+lS5fWtm3btHPnzmznDwAAAACAPbFqsSItLU379+9XdHS00ebg4KCwsDDFxcVli09OTtb8+fNVpkwZlSxZMs/H+frrr1W2bFmtXbtWLVq0kNlsVlhYmN5++22jMPHVV1+pdu3aevvtt/X555+rQIECatu2rcaPHy93d3cjJjQ0VH379tWaNWvk5+enLl266I033jBmeXz11VcKDw/XCy+8oO3bt6t48eJ6/fXX1atXr1yvwZw5c+Tt7a2aNWsa7YmJierVq5dWr14tDw+Ph57jsWPHtGHDBj333HO5xqSmpio1NdX4nJSUJElycZQyHXPbC7BdLo6WvwL2hjFs+3i88sGyrg/XCfaKMQx7Z49jOK+5WrVYcenSJWVkZMjf39+i3d/f32Jdig8//FBDhw5VcnKyKlWqpNjYWLm4uOT5OCdOnNDp06e1bNkyffbZZ8rIyNDgwYP1/PPPa8uWLUbMd999Jzc3N61atUqXLl3S66+/rsuXLxuzJE6cOKEtW7aoa9euWrdunY4dO6bXX39d6enpGj16tBEza9YsRUVFafjw4dq7d68GDBggFxcXdevWzchp7dq1evHFF5WSkqLAwEDFxsaqSJEiku6ur9G9e3f16dNHtWvX1qlTp3I9t/r16+vAgQNKTU1V7969NW7cuFxjJ02apLFjx2Zr7xfilaeCCGCroup4WzsF4C9hDNuu+9elQs5iY2OtnQLwlzCGYe/saQynpKTkKc7qj4HkRdeuXdWsWTOdP39e06ZNU8eOHfX999/Lzc0tT/tnZmYqNTVVn332mSpWrChJmjt3rkJCQnT06FFVqlRJmZmZMplMWrhwoby97/6lcfr06Xr++ef14Ycfyt3dXZmZmSpatKjmzJkjR0dHhYSE6OzZs5o6dapRrMjMzFTt2rWNdSNq1aqlw4cPa/bs2RbFiiZNmig+Pl6XLl3Sxx9/rI4dO2r37t0qWrSoPvjgA924ccNixklulixZohs3bujgwYMaMmSIpk2bpqFDh+YYGx0draioKONzUlKSSpYsqRn7k5TpZD+VOCCLi+PdL3nT91xXWoa1swHyjzFs+3aM72ztFGxaenq6YmNj1axZM+MRWcCeMIZh7+xxDGfN8H8YqxYrihQpIkdHRyUmJlq0JyYmKiAgwPjs7e0tb29vVahQQfXq1ZOPj49WrVqlzp3z9heIwMBAOTk5GYUKSapSpYok6Y8//lClSpUUGBio4sWLG4WKrBiz2awzZ86oQoUKCgwMlLOzs8XCnlWqVFFCQoLS0tLk4uKiwMBAVa1a1eL4VapU0YoVKyzaChQooPLly6t8+fKqV6+eKlSooLlz5yo6OlpbtmxRXFycXF1dLfapXbu2unbtqgULFhhtWY/DVK1aVRkZGerdu7f+9a9/WeSYxdXVNVufkpSWIWWaHnoZAZuVliG+6MGuMYZtl738xc/anJ2duVawa4xh2Dt7GsN5zdOqry51cXFRSEiINm/ebLRlZmZq8+bNCg0NzXEfs9kss9lssfbCwzRo0EB37tzR8ePHjbbffvtN0t2FKbNizp07p5s3b1rEODg4qESJEkbMsWPHLN4Q8ttvvykwMNB4LKVBgwY6evSoxfF/++034zi5yZr9Id1dMPTgwYOKj49XfHy8MQV1yZIl2V63en8f6enpFvkBAAAAAGBvrP4YSFRUlLp166batWurTp06iomJUXJysiIjI3XixAktWbJEzZs3l5+fn86cOaPJkyfL3d1dLVu2NPo4duyYbt68qYSEBN26dUvx8fGS7s42cHFxUVhYmB5//HG98soriomJUWZmpvr27atmzZoZsy26dOmi8ePHKzIyUmPHjtWlS5c0ZMgQvfLKK8YCm6+99ppmzJihgQMHqn///vr99981ceJEDRgwwMhl8ODBql+/viZOnKiOHTtqz549mjNnjubMmSPp7iKhb731ltq2bavAwEBdunRJM2fO1NmzZ/XCCy9IkkqVKmVxjTw9PSVJ5cqVMwonCxculLOzs6pXry5XV1ft27dP0dHR6tSpk91U1AAAAAAAyInVixWdOnXSxYsXNWrUKCUkJCg4OFgbNmyQv7+/zp07p507dyomJkZXr16Vv7+/GjZsqF27dqlo0aJGHz179tT27duNz7Vq1ZIknTx5UkFBQXJwcNDXX3+t/v37q2HDhipQoICeeeYZvfPOO8Y+np6eio2NVf/+/VW7dm35+vqqY8eOmjBhghFTsmRJbdy4UYMHD1aNGjVUvHhxDRw4UG+88YYR88QTT2jVqlWKjo7WuHHjVKZMGcXExKhr166SJEdHRx05ckQLFizQpUuX5OvrqyeeeEI7d+5UtWrV8nzdnJycNGXKFP32228ym80qXbq0+vXrp8GDB+f/JgAAAAAAYENMZrPZbO0kYB1JSUny9vZW7UGzlenkbu10gHxzcZSGhXprchyLE8I+MYZt3/6pEdZOwaalp6dr3bp1atmyJTM7YZcYw7B39jiGs76HXr9+XV5eXrnGWXXNCgAAAAAAgPtRrAAAAAAAADaFYgUAAAAAALApFCsAAAAAAIBNoVgBAAAAAABsCsUKAAAAAABgUyhWAAAAAAAAm0KxAgAAAAAA2BSKFQAAAAAAwKZQrAAAAAAAADaFYgUAAAAAALApFCsAAAAAAIBNcbJ2ArC+DSOel6+vr7XTAPItPT1d69at047xneXs7GztdIB8YwwDAADkjJkVAAAAAADAplCsAAAAAAAANoViBQAAAAAAsCkUKwAAAAAAgE2hWAEAAAAAAGwKxQoAAAAAAGBTKFYAAAAAAACbQrECAAAAAADYFCdrJwDrazFhuTKd3K2dBpBvLo7SsFBvNRy5WGkZ1s4Gtmb/1AhrpwAAAIA/iZkVAAAAAADAplCsAAAAAAAANoViBQAAAAAAsCkUKwAAAAAAgE2hWAEAAAAAAGwKxQoAAAAAAGBTKFYAAAAAAACbQrECAAAAAADYFIoVAAAAAADAplCsAAAAAAAANoViBQAAAAAAsCkUKwAAAAAAgE2hWAEAAAAAAGyKTRQrZs6cqaCgILm5ualu3bras2ePse3VV19VuXLl5O7uLj8/P7Vr105Hjhyx2H/v3r1q2rSpChUqJB8fH4WHh+vgwYPG9lOnTslkMmX7+eGHH4yYTz/9NNt2Nzc3i+OYzWaNGjVKgYGBcnd3V1hYmH7//fcczyk1NVXBwcEymUyKj4832rdt26Z27dopMDBQBQoUUHBwsBYuXJht/2XLlqly5cpyc3NT9erVtW7dOovtiYmJ6t69u4oVKyYPDw+1aNEi11wAAAAAALAnVi9WLFmyRFFRURo9erQOHDigmjVrKjw8XBcuXJAkhYSEaP78+fr111+1ceNGmc1mNW/eXBkZGZKkmzdvqkWLFipVqpR2796t7777TgULFlR4eLjS09MtjvXtt9/q/Pnzxk9ISIjFdi8vL4vtp0+fttj+9ttv6/3339fs2bO1e/duFShQQOHh4bp9+3a28xo6dKiKFSuWrX3Xrl2qUaOGVqxYoUOHDikyMlIRERFau3atRUznzp3Vo0cP/fjjj2rfvr3at2+vw4cPS7pbNGnfvr1OnDihNWvW6Mcff1Tp0qUVFham5OTkP3EXAAAAAACwHU7WTmD69Onq1auXIiMjJUmzZ8/WN998o3nz5mnYsGHq3bu3ERsUFKQJEyaoZs2aOnXqlMqVK6cjR47oypUrGjdunEqWLClJGj16tGrUqKHTp0+rfPnyxv6+vr4KCAjINReTyZTrdrPZrJiYGI0YMULt2rWTJH322Wfy9/fX6tWr9eKLLxqx69ev16ZNm7RixQqtX7/eop/hw4dbfB44cKA2bdqklStXqnXr1pKk9957Ty1atNCQIUMkSePHj1dsbKxmzJih2bNn6/fff9cPP/ygw4cPq1q1apKkWbNmKSAgQIsXL1bPnj0fcMUBAAAAALBtVi1WpKWlaf/+/YqOjjbaHBwcFBYWpri4uGzxycnJmj9/vsqUKWMUJipVqiRfX1/NnTtXw4cPV0ZGhubOnasqVaooKCjIYv+2bdvq9u3bqlixooYOHaq2bdtabL9586ZKly6tzMxMPf7445o4caJRDDh58qQSEhIUFhZmxHt7e6tu3bqKi4szihWJiYnq1auXVq9eLQ8Pjzxdh+vXr6tKlSrG57i4OEVFRVnEhIeHa/Xq1ZLuPmIiyeIxFQcHB7m6uuq7777LtViRmppq7CtJSUlJkiQXRynTMU+pAjbFxdHyV+Be98+us0VZOdpDrkBOGMOwd4xh2Dt7HMN5zdWqxYpLly4pIyND/v7+Fu3+/v4W61J8+OGHGjp0qJKTk1WpUiXFxsbKxcVFklSwYEFt27ZN7du31/jx4yVJFSpU0MaNG+XkdPf0PD099c4776hBgwZycHDQihUr1L59e61evdooWFSqVEnz5s1TjRo1dP36dU2bNk3169fXzz//rBIlSighIcHI7f5cs7aZzWZ1795dffr0Ue3atXXq1KmHXoOlS5dq7969+uijj4y2hISEBx6ncuXKKlWqlKKjo/XRRx+pQIECevfdd3XmzBmdP38+12NNmjRJY8eOzdbeL8Qrz4UVwBZF1fG2dgqwQfev9WPLYmNjrZ0C8JcwhmHvGMOwd/Y0hlNSUvIUZ/XHQPKia9euatasmc6fP69p06apY8eO+v777+Xm5qZbt26pR48eatCggRYvXqyMjAxNmzZNrVq10t69e+Xu7q4iRYpYzFR44okndO7cOU2dOtUoVoSGhio0NNSIqV+/vqpUqaKPPvrIKII8zAcffKAbN25YzBR5kK1btyoyMlIff/yxMYMjL5ydnbVy5Ur16NFDhQsXlqOjo8LCwvTMM8/IbDbnul90dLTFdUhKSlLJkiU1Y3+SMp3spxIHZHFxvFuomL7nutIyrJ0NbM2O8Z2tncJDpaenKzY2Vs2aNZOzs7O10wHyjTEMe8cYhr2zxzGcNcP/YaxarChSpIgcHR2VmJho0Z6YmGixdoS3t7e8vb1VoUIF1atXTz4+Plq1apU6d+6sRYsW6dSpU4qLi5ODw931QhctWiQfHx+tWbPGYi2Je9WtW/eB1SdnZ2fVqlVLx44dkyQjn8TERAUGBlrkGhwcLEnasmWL4uLi5OrqatFX7dq11bVrVy1YsMBo2759u9q0aaN3331XERERFvEBAQEPvSYhISGKj4/X9evXlZaWJj8/P9WtW1e1a9fO9ZxcXV2z5SZJaRlSpinX3QCbl5YhihXIxl7+wJbu5mpP+QL3YwzD3jGGYe/saQznNU+rvg3ExcVFISEh2rx5s9GWmZmpzZs3W8xyuJfZbJbZbDbWXkhJSZGDg4NMpv/7tp31OTMzM9djx8fHWxQd7peRkaGffvrJiClTpowCAgIsck1KStLu3buNXN9//30dPHhQ8fHxio+PN6YgL1myRG+99Zax37Zt29SqVStNmTLFYgHRLKGhoRbHke5O68npmnh7e8vPz0+///679u3bZyz+CQAAAACAvbL6YyBRUVHq1q2bateurTp16igmJkbJycmKjIzUiRMntGTJEjVv3lx+fn46c+aMJk+eLHd3d7Vs2VKS1KxZMw0ZMkR9+/ZV//79lZmZqcmTJ8vJyUlNmjSRJC1YsEAuLi6qVauWJGnlypWaN2+ePvnkEyOPcePGqV69eipfvryuXbumqVOn6vTp08ZilSaTSYMGDdKECRNUoUIFlSlTRiNHjlSxYsXUvn17SVKpUqUszs3T01OSVK5cOZUoUULS3Uc/WrdurYEDB6pDhw7GOhQuLi4qXLiwpLtvCGnUqJHeeecdtWrVSl9++aX27dunOXPmGH0vW7ZMfn5+KlWqlH766ScNHDhQ7du3V/PmzR/p/QEAAAAA4L/N6sWKTp066eLFixo1apQSEhIUHBysDRs2yN/fX+fOndPOnTsVExOjq1evyt/fXw0bNtSuXbtUtGhRSXcXm/z66681duxYhYaGysHBQbVq1dKGDRssZk6MHz9ep0+flpOTkypXrqwlS5bo+eefN7ZfvXpVvXr1UkJCgnx8fBQSEqJdu3apatWqRkzWIp+9e/fWtWvX9OSTT2rDhg0Wb+V4mAULFiglJUWTJk3SpEmTjPZGjRpp27Ztku6ul7Fo0SKNGDFCw4cPV4UKFbR69Wo99thjRvz58+cVFRVlPJYSERGhkSNH5vv6AwAAAABga0zmB63IiP9pSUlJ8vb2Vu1Bs5Xp5G7tdIB8c3GUhoV6a3IcC2wiu/1TIx4eZGXp6elat26dWrZsaTfPmQL3YgzD3jGGYe/scQxnfQ+9fv26vLy8co2z6poVAAAAAAAA96NYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJtCsQIAAAAAANgUihUAAAAAAMCmUKwAAAAAAAA2hWIFAAAAAACwKRQrAAAAAACATaFYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJviZO0EYH0bRjwvX19fa6cB5Ft6errWrVunHeM7y9nZ2drpAAAAAHhEmFkBAAAAAABsCsUKAAAAAABgUyhWAAAAAAAAm0KxAgAAAAAA2BSKFQAAAAAAwKZQrAAAAAAAADaFYgUAAAAAALApFCsAAAAAAIBNcbJ2ArC+FhOWK9PJ3dppAPnm4igNC/VWw5GLlZZh7Wzs2/6pEdZOAQAAADAwswIAAAAAANgUihUAAAAAAMCmUKwAAAAAAAA2hWIFAAAAAACwKRQrAAAAAACATaFYAQAAAAAAbArFCgAAAAAAYFMoVgAAAAAAAJtCsQIAAAAAANgUihUAAAAAAMCmUKwAAAAAAAA2hWIFAAAAAACwKRQrAAAAAACATaFYAQAAAAAAbIpNFCtmzpypoKAgubm5qW7dutqzZ4+x7fbt2+rbt698fX3l6empDh06KDExMVsfn376qWrUqCE3NzcVLVpUffv2tdh+6NAhPfXUU3Jzc1PJkiX19ttvW2xPT0/XuHHjVK5cObm5ualmzZrasGGDRcykSZP0xBNPqGDBgipatKjat2+vo0ePWsQkJCTo5ZdfVkBAgAoUKKDHH39cK1assIhp27atSpUqJTc3NwUGBurll1/WuXPnjO1Hjx5VkyZN5O/vLzc3N5UtW1YjRoxQenq6RT/Xrl1T3759FRgYKFdXV1WsWFHr1q3LwxUHAAAAAMB2Wb1YsWTJEkVFRWn06NE6cOCAatasqfDwcF24cEGSNHjwYH399ddatmyZtm/frnPnzum5556z6GP69Ol68803NWzYMP3888/69ttvFR4ebmxPSkpS8+bNVbp0ae3fv19Tp07VmDFjNGfOHCNmxIgR+uijj/TBBx/ol19+UZ8+ffTss8/qxx9/NGK2b9+uvn376ocfflBsbKzS09PVvHlzJScnGzERERE6evSovvrqK/3000967rnn1LFjR4t+mjRpoqVLl+ro0aNasWKFjh8/rueff97Y7uzsrIiICG3atElHjx5VTEyMPv74Y40ePdqISUtLU7NmzXTq1CktX75cR48e1ccff6zixYs/grsCAAAAAID1mMxms9maCdStW1dPPPGEZsyYIUnKzMxUyZIl1b9/f7322mvy8/PTokWLjC/zR44cUZUqVRQXF6d69erp6tWrKl68uL7++ms1bdo0x2PMmjVLb775phISEuTi4iJJGjZsmFavXq0jR45IkooVK6Y333zTYkZGhw4d5O7uri+++CLHfi9evKiiRYtq+/btatiwoSTJ09NTs2bN0ssvv2zE+fr6asqUKerZs2eO/Xz11Vdq3769UlNT5ezsnGNMVFSU9u7dq507d0qSZs+eralTp+rIkSO57nO/1NRUpaamGp+TkpJUsmRJ1f/XbGU6ueepD8CWuDhKUXW8NX3PdaVlWDsb+7ZjfGdrp/CPlJ6ertjYWDVr1izP/y8HbAljGPaOMQx7Z49jOCkpSUWKFNH169fl5eWVa5zTfzGnbNLS0rR//35FR0cbbQ4ODgoLC1NcXJzq1Kmj9PR0hYWFGdsrV66sUqVKGcWK2NhYZWZm6uzZs6pSpYpu3Lih+vXr65133lHJkiUlSXFxcWrYsKFRqJCk8PBwTZkyRVevXpWPj49SU1Pl5uZmkZ+7u7u+++67XPO/fv26JKlw4cJGW/369bVkyRK1atVKhQoV0tKlS3X79m01btw4xz6uXLmihQsXqn79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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import library visualisasi\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Menghitung jumlah rating per ISBN\n",
    "top_books = ratings['ISBN'].value_counts().head(10)\n",
    "\n",
    "# Visualisasi 10 ISBN dengan rating terbanyak\n",
    "plt.figure(figsize=(12,6))\n",
    "sns.barplot(x=top_books.values, y=top_books.index)\n",
    "plt.title('10 Buku dengan Rating Terbanyak (ISBN)')\n",
    "plt.xlabel('Jumlah Rating')\n",
    "plt.ylabel('ISBN Buku')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WbQ8eN1i9PXG"
   },
   "source": [
    "**Berikut adalah 10 buku yang menerima rating terbanyak dalam dataset:**\n",
    "\n",
    "1. **Wild Animus** – memperoleh **2.502 rating**\n",
    "2. **The Lovely Bones: A Novel** – sebanyak **1.295 rating**\n",
    "3. **The Da Vinci Code** – mendapat **883 rating**\n",
    "4. **Divine Secrets of the Ya-Ya Sisterhood: A Novel** – **732 rating**\n",
    "5. **The Red Tent** – **723 rating**\n",
    "6. **A Painted House** – **647 rating**\n",
    "7. **\\[Judul Tidak Tersedia/NaN]** – **639 rating**\n",
    "8. **The Secret Life of Bees** – **615 rating**\n",
    "9. **Snow Falling on Cedars** – **614 rating**\n",
    "10. **Angels & Demons** – **586 rating**\n",
    "\n",
    "**Insight:**\n",
    "\n",
    "* Buku **Wild Animus** menonjol dengan jumlah rating yang jauh melebihi buku lainnya. Hal ini kemungkinan dipengaruhi oleh tingkat **popularitas** atau **strategi promosi tertentu**.\n",
    "* Ditemukan satu entri dengan **judul buku kosong (NaN)** meskipun memiliki banyak rating. Hal ini memerlukan perhatian khusus saat proses **pembersihan data** atau pencocokan ulang dengan ISBN terkait.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 363
    },
    "id": "aNtaoUN49Pt4",
    "outputId": "261320da-ace5-4a37-85d9-9ac6b4d9236e"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Title</th>\n",
       "      <th>Jumlah Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0971880107</td>\n",
       "      <td>Wild Animus</td>\n",
       "      <td>2502</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0316666343</td>\n",
       "      <td>The Lovely Bones: A Novel</td>\n",
       "      <td>1295</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0385504209</td>\n",
       "      <td>The Da Vinci Code</td>\n",
       "      <td>883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0060928336</td>\n",
       "      <td>Divine Secrets of the Ya-Ya Sisterhood: A Novel</td>\n",
       "      <td>732</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0312195516</td>\n",
       "      <td>The Red Tent (Bestselling Backlist)</td>\n",
       "      <td>723</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>044023722X</td>\n",
       "      <td>A Painted House</td>\n",
       "      <td>647</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0679781587</td>\n",
       "      <td>NaN</td>\n",
       "      <td>639</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0142001740</td>\n",
       "      <td>The Secret Life of Bees</td>\n",
       "      <td>615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>067976402X</td>\n",
       "      <td>Snow Falling on Cedars</td>\n",
       "      <td>614</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0671027360</td>\n",
       "      <td>Angels &amp;amp; Demons</td>\n",
       "      <td>586</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         ISBN                                       Book-Title  Jumlah Rating\n",
       "0  0971880107                                      Wild Animus           2502\n",
       "1  0316666343                        The Lovely Bones: A Novel           1295\n",
       "2  0385504209                                The Da Vinci Code            883\n",
       "3  0060928336  Divine Secrets of the Ya-Ya Sisterhood: A Novel            732\n",
       "4  0312195516              The Red Tent (Bestselling Backlist)            723\n",
       "5  044023722X                                  A Painted House            647\n",
       "6  0679781587                                              NaN            639\n",
       "7  0142001740                          The Secret Life of Bees            615\n",
       "8  067976402X                           Snow Falling on Cedars            614\n",
       "9  0671027360                              Angels &amp; Demons            586"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Ambil 10 ISBN yang paling banyak dirating\n",
    "top_books = ratings['ISBN'].value_counts().head(10).reset_index()\n",
    "\n",
    "# Rename kolom supaya mudah di-merge\n",
    "top_books.columns = ['ISBN', 'Jumlah Rating']\n",
    "\n",
    "# Join ke dataset books untuk dapatkan judul buku\n",
    "top_books_titles = top_books.merge(books[['ISBN', 'Book-Title']], on='ISBN', how='left')\n",
    "\n",
    "# Tampilkan hasil\n",
    "top_books_titles[['ISBN', 'Book-Title', 'Jumlah Rating']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 564
    },
    "id": "yJ6UrX4g9TOi",
    "outputId": "90ce8219-24f8-4ae3-a430-627a0575ca4a"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA2wAAAIjCAYAAAB/FZhcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8ekN5oAAAACXBIWXMAAA9hAAAPYQGoP6dpAABXqUlEQVR4nO3de1iUdf7/8Rcgw0EFDylInijb1PKIaVNmHhAysiy3VTMjT60GJbJrSimeKtPWY2J0UGkrTevbUUsZddUt8YRSammHtWxT0ErBQwLC/P5omZ8jMwKKzMd8Pq7L62ru+z33582bcXde3vfc42W32+0CAAAAABjH29MNAAAAAABcI7ABAAAAgKEIbAAAAABgKAIbAAAAABiKwAYAAAAAhiKwAQAAAIChCGwAAAAAYCgCGwAAAAAYisAGAAAAAIYisAHAFWLSpEny8vKqkrW6du2qrl27Oh6vX79eXl5eeuedd6pk/RJpaWny8vLS999/X+nH9vLy0qRJkyplrYcfflhNmzattN4qS9OmTfXwww97uo1K46nXIQBcDAIbAFyGSsJByR9/f3+FhYUpOjpa8+bN0/HjxytlnYMHD2rSpEnKysqqlOOZqiTMhoSE6NSpU6X2N23aVHfddZcHOnPWtWtXp997QECAWrdurTlz5qi4uPiCjrlp0yZNmjRJx44dq9xmL9LZP6eXl5eqV6+uli1b6umnn3b5OwKAP6pqnm4AAHDhpkyZovDwcBUWFio7O1vr169XQkKCZs2apQ8//FCtW7d21I4fP17jxo2r0PEPHjyoyZMnq2nTpmrbtm25n5eenl6hdS6VQYMGqX///vLz8ytX/eHDh/Xiiy/qb3/7W5m1v/32m6pVq5z/G33llVfKHbgaNmyoadOmSZJ+/vlnLVmyRKNHj9aRI0f0zDPPVHjtTZs2afLkyXr44YdVq1Ytp3379u2Tt7fn/m23Z8+eeuihhyRJJ06c0L///W9NmDBBn3/+ud5++22P9QUAVYnABgCXsV69eqlDhw6Ox0lJSVq3bp3uuusu3X333frqq68UEBAgSapWrVqlBQx3Tp06pcDAQFkslku6Tnn5+PjIx8en3PVt27bV888/r0cffdQxN3f8/f0vtj0HX1/fctcGBwfrwQcfdDweMWKEmjdvrhdeeEFTpkyp0M9blvIG3UvlT3/6U6mftaCgQO+++65Onz5dqb8DADAVl0QCwB9M9+7dNWHCBP3www964403HNtdfYbNZrOpc+fOqlWrlmrUqKHrr79eTz75pKTfP+9z0003SZIGDx7suDQtLS1N0u+X5914443KzMxUly5dFBgY6HjuuZ9hK1FUVKQnn3xSoaGhql69uu6++279+OOPTjXuPjfl6pgvvPCCbrjhBgUGBqp27drq0KGDlixZ4thf0c+VJScnKycnRy+++GKZted+hs2VDz74QDExMQoLC5Ofn5+uvfZaTZ06VUVFRU51F/MZNn9/f9100006fvy4Dh8+7Nj+xRdf6OGHH9Y111wjf39/hYaGasiQIfrll18cNZMmTdKYMWMkSeHh4Y7fccm8zv1dlMzzs88+U2JiourVq6fq1avr3nvv1ZEjR5z6Ki4u1qRJkxQWFqbAwEB169ZNX3755UV/Li40NFReXl5O//hQkdfMufLz83XXXXcpODhYmzZt0vfff+/0Oj9beX7nAFDZOMMGAH9AgwYN0pNPPqn09HQNHz7cZc2ePXt01113qXXr1poyZYr8/Pz07bff6rPPPpMktWjRQlOmTFFycrIeeeQR3XbbbZKkW265xXGMX375Rb169VL//v314IMPKiQk5Lx9PfPMM/Ly8tLYsWN1+PBhzZkzR5GRkcrKyirzjNa5XnnlFT3++OP685//rFGjRun06dP64osvtGXLFj3wwAMVOlaJ2267Td27d9eMGTM0cuTICvd0rrS0NNWoUUOJiYmqUaOG1q1bp+TkZOXl5en555+/qGOfrSRknH1Jo81m03/+8x8NHjxYoaGh2rNnj15++WXt2bNHmzdvlpeXl+677z59/fXXWrp0qWbPnq2rrrpKklSvXr3zrvfYY4+pdu3amjhxor7//nvNmTNH8fHxWrZsmaMmKSlJM2bMUO/evRUdHa3PP/9c0dHROn36dLl/rtOnT+vnn3+WJJ08eVKfffaZXnvtNT3wwAOVcrb4t99+0z333KPt27drzZo1uummmy7JDWoA4GIQ2ADgD6hhw4YKDg7Wd99957bGZrOpoKBAn3zyieON+tlCQkLUq1cvJScny2q1Ol2aViI7O1upqan661//Wq6+fv31V3311VeqWbOmJKl9+/b6y1/+4ghfFbFy5UrdcMMNlf5ZpokTJ+r2229XamqqRo8efVHHWrJkiVPoGzFihEaMGKEFCxbo6aefvqBLDouKihwh5pdfftHChQu1fft2xcTEOK316KOPlvos3s0336wBAwbo008/1W233abWrVurffv2Wrp0qfr06VPus3x169ZVenq644xtcXGx5s2bp9zcXAUHBysnJ0ezZs1Snz599N577zmeN3ny5AqdoVq4cKEWLlzotK1Pnz565ZVXyn0Md06cOKG77rpLe/bs0bp16yr0GU0AqEpcEgkAf1A1atQ4790iS87GfPDBBxd8h0E/Pz8NHjy43PUPPfSQI6xJ0p///Gc1aNBAH3/8cYXXrlWrlv773/9q27ZtFX7u+XTp0kXdunXTjBkz9Ntvv13Usc4OUMePH9fPP/+s2267TadOndLevXsv6Jh79+5VvXr1VK9ePTVv3lzPP/+87r777lKX8J29dsmZqptvvlmStGPHjgtau8QjjzzidHntbbfdpqKiIv3www+SpLVr1+rMmTN69NFHnZ732GOPVWide+65RzabTTabTR988IGSkpK0atUqPfDAA7Lb7Rfcf25urqKiorR3716tX7+esAbAaAQ2APiDOnHihFM4Ole/fv106623atiwYQoJCVH//v21fPnyCoW3q6++ukI3GLnuuuucHnt5ealZs2YXdBna2LFjVaNGDXXs2FHXXXed4uLiHJdzXqxJkyY5zh5ejD179ujee+9VcHCwgoKCVK9ePceZytzc3As6ZtOmTWWz2bR69WotWLBAV199tY4cOVLqBhy//vqrRo0apZCQEAUEBKhevXoKDw+/qLVLNG7c2Olx7dq1JUlHjx6VJEdwa9asmVNdnTp1HLXl0bBhQ0VGRioyMlJ33323nn32WT399NN69913tWLFigvuPyEhQdu2bdOaNWt0ww03XPBxAKAqENgA4A/ov//9r3Jzc0u9YT5bQECANm7cqDVr1mjQoEH64osv1K9fP/Xs2bPUTTHOd4zK5u7Lvc/tqUWLFtq3b5/eeustde7cWf/3f/+nzp07a+LEiRfdQ5cuXdS1a9eLOst27Ngx3X777fr88881ZcoUffTRR7LZbJo+fbokXfBZzerVqysyMlJRUVEaOXKkPv74Y23dutVxw5cSJZeajhgxQu+++67S09O1atWqi1q7hLs7UV7MWa/y6tGjhyRp48aNjm3lfc2UuOeee2S32/Xcc8+VmkVFjwUAlxqBDQD+gF5//XVJUnR09HnrvL291aNHD82aNUtffvmlnnnmGa1bt07/+te/JLl/83qhvvnmG6fHdrtd3377rdNnp2rXru3yS5xLztqcrXr16urXr58WL16sAwcOKCYmRs8880yFbmzhTslZtpdeeumCnr9+/Xr98ssvSktL06hRo3TXXXcpMjKyQmeYyqN169Z68MEH9dJLL+nAgQOSfj/TtXbtWo0bN06TJ0/Wvffeq549e+qaa64p9fzK/h1LUpMmTSRJ3377rdP2X375xXEW7kKdOXNG0u9nkEtU5DUj/f45uEWLFmnJkiWKi4tz2lfy+zn3eO6OBQCXGoENAP5g1q1bp6lTpyo8PFwDBw50W/frr7+W2lbyWZ78/HxJvwciqfSb1wv1z3/+0+lzde+8844OHTqkXr16ObZde+212rx5swoKChzbVqxYUer2/2ffnl6SLBaLWrZsKbvdrsLCwovu9fbbb1fXrl01ffr0CwqAJWehzj7rVFBQoAULFlx0b+d64oknVFhYqFmzZrldW5LmzJlT6rmV/TuWfj8LVq1atVJfjzB//vyLPvZHH30kSWrTpo1jW3lfM2d76KGHNG/ePKWmpmrs2LGO7UFBQbrqqquczuBJuiS/NwAoD+4SCQCXsU8++UR79+7VmTNnlJOTo3Xr1slms6lJkyb68MMPz/vFwlOmTNHGjRsVExOjJk2a6PDhw1qwYIEaNmyozp07S/r9jXCtWrWUmpqqmjVrqnr16urUqZPjs1AVVadOHXXu3FmDBw9WTk6O5syZo2bNmjl99cCwYcP0zjvv6I477tBf/vIXfffdd3rjjTd07bXXOh0rKipKoaGhuvXWWxUSEqKvvvpK8+fPV0xMzHk/u1cREydOVLdu3S7oubfccotq166t2NhYPf744/Ly8tLrr79+SS4bbNmype688069+uqrmjBhgurWrasuXbpoxowZKiws1NVXX6309HTt37+/1HMjIiIkSU899ZT69+8vX19f9e7d2xHkLkRISIhGjRqlmTNn6u6779Ydd9yhzz//3HFH0vKe1fv6668d3yV46tQpbd68Wa+99pqaNWumQYMGOerK+5o5V3x8vPLy8vTUU08pODjYcVnpsGHD9Nxzz2nYsGHq0KGDNm7cqK+//voCpwEAF4fABgCXseTkZEm/n12qU6eOWrVqpTlz5mjw4MFlhpa7775b33//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OnXqeLgTcxQVFemtt97SyZMnZbVaPd2OEeLi4hQTE+P0vzWQvvnmG4WFhemaa67RwIEDdeDAAU+35HEffvihOnTooPvvv1/169dXu3bt9Morr3i6LeMUFBTojTfe0JAhQ+Tl5eXpdjzqlltu0dq1a/X1119Lkj7//HN9+umn6tWrl4c787wzZ86oqKhI/v7+TtsDAgI4o/8/+/fvV3Z2ttP/PwUHB6tTp07Gvheu5ukGULV+/vlnFRUVKSQkxGl7SEiI9u7d66GucLkoLi5WQkKCbr31Vt14442ebsfjdu3aJavVqtOnT6tGjRp677331LJlS0+35XFvvfWWduzYccV9ZqIsnTp1Ulpamq6//nodOnRIkydP1m233abdu3erZs2anm7PY/7zn//oxRdfVGJiop588klt27ZNjz/+uCwWi2JjYz3dnjHef/99HTt2TA8//LCnW/G4cePGKS8vT82bN5ePj4+Kior0zDPPaODAgZ5uzeNq1qwpq9WqqVOnqkWLFgoJCdHSpUuVkZGhZs2aebo9I2RnZ0uSy/fCJftMQ2ADUG5xcXHavXs3/0r3P9dff72ysrKUm5urd955R7GxsdqwYcMVHdp+/PFHjRo1SjabrdS/8F7pzv7X/9atW6tTp05q0qSJli9frqFDh3qwM88qLi5Whw4d9Oyzz0qS2rVrp927dys1NZXAdpaFCxeqV69eCgsL83QrHrd8+XK9+eabWrJkiW644QZlZWUpISFBYWFhvGYkvf766xoyZIiuvvpq+fj4qH379howYIAyMzM93RouEJdEXmGuuuoq+fj4KCcnx2l7Tk6OQkNDPdQVLgfx8fFasWKF/vWvf6lhw4aebscIFotFzZo1U0REhKZNm6Y2bdpo7ty5nm7LozIzM3X48GG1b99e1apVU7Vq1bRhwwbNmzdP1apVU1FRkadbNEatWrX0pz/9Sd9++62nW/GoBg0alPpHjhYtWnC56Fl++OEHrVmzRsOGDfN0K0YYM2aMxo0bp/79+6tVq1YaNGiQRo8erWnTpnm6NSNce+212rBhg06cOKEff/xRW7duVWFhoa655hpPt2aEkve7l9N7YQLbFcZisSgiIkJr1651bCsuLtbatWv57A1cstvtio+P13vvvad169YpPDzc0y0Zq7i4WPn5+Z5uw6N69OihXbt2KSsry/GnQ4cOGjhwoLKysuTj4+PpFo1x4sQJfffdd2rQoIGnW/GoW2+9tdRXhXz99ddq0qSJhzoyz+LFi1W/fn3FxMR4uhUjnDp1St7ezm9hfXx8VFxc7KGOzFS9enU1aNBAR48e1erVq3XPPfd4uiUjhIeHKzQ01Om9cF5enrZs2WLse2EuibwCJSYmKjY2Vh06dFDHjh01Z84cnTx5UoMHD/Z0ax514sQJp3/p3r9/v7KyslSnTh01btzYg515VlxcnJYsWaIPPvhANWvWdFzfHRwcrICAAA935zlJSUnq1auXGjdurOPHj2vJkiVav369Vq9e7enWPKpmzZqlPt9YvXp11a1b94r/3OPf//539e7dW02aNNHBgwc1ceJE+fj4aMCAAZ5uzaNGjx6tW265Rc8++6z+8pe/aOvWrXr55Zf18ssve7o1IxQXF2vx4sWKjY1VtWq8bZOk3r1765lnnlHjxo11ww03aOfOnZo1a5aGDBni6daMsHr1atntdl1//fX69ttvNWbMGDVv3vyKep9X1nu6hIQEPf3007ruuusUHh6uCRMmKCwsTH369PFc0+fj6dtUwjNeeOEFe+PGje0Wi8XesWNH++bNmz3dksf961//sksq9Sc2NtbTrXmUq5lIsi9evNjTrXnUkCFD7E2aNLFbLBZ7vXr17D169LCnp6d7ui0jcVv/3/Xr18/eoEEDu8VisV999dX2fv362b/99ltPt2WEjz76yH7jjTfa/fz87M2bN7e//PLLnm7JGKtXr7ZLsu/bt8/TrRgjLy/PPmrUKHvjxo3t/v7+9muuucb+1FNP2fPz8z3dmhGWLVtmv+aaa+wWi8UeGhpqj4uLsx87dszTbVWpst7TFRcX2ydMmGAPCQmx+/n52Xv06GH03zEvu52vhQcAAAAAE/EZNgAAAAAwFIENAAAAAAxFYAMAAAAAQxHYAAAAAMBQBDYAAAAAMBSBDQAAAAAMRWADAAAAAEMR2AAAAADAUAQ2AAA8YP369fLy8tKxY8c83QoAwGAENgAAzuPhhx+Wl5eXvLy85Ovrq/DwcD3xxBM6ffp0uY/RtWtXJSQkOG275ZZbdOjQIQUHB1dyxwCAP5Jqnm4AAADT3XHHHVq8eLEKCwuVmZmp2NhYeXl5afr06Rd8TIvFotDQ0ErsEgDwR8QZNgAAyuDn56fQ0FA1atRIffr0UWRkpGw2myTpl19+0YABA3T11VcrMDBQrVq10tKlSx3Pffjhh7VhwwbNnTvXcabu+++/L3VJZFpammrVqqXVq1erRYsWqlGjhu644w4dOnTIcawzZ87o8ccfV61atVS3bl2NHTtWsbGx6tOnT1WOAwBQhQhsAABUwO7du7Vp0yZZLBZJ0unTpxUREaGVK1dq9+7deuSRRzRo0CBt3bpVkjR37lxZrVYNHz5chw4d0qFDh9SoUSOXxz516pT+8Y9/6PXXX9fGjRt14MAB/f3vf3fsnz59ut58800tXrxYn332mfLy8vT+++9f8p8ZAOA5XBIJAEAZVqxYoRo1aujMmTPKz8+Xt7e35s+fL0m6+uqrnULVY489ptWrV2v58uXq2LGjgoODZbFYFBgYWOYlkIWFhUpNTdW1114rSYqPj9eUKVMc+1944QUlJSXp3nvvlSTNnz9fH3/8cWX/uAAAgxDYAAAoQ7du3fTiiy/q5MmTmj17tqpVq6a+fftKkoqKivTss89q+fLl+umnn1RQUKD8/HwFBgZWeJ3AwEBHWJOkBg0a6PDhw5Kk3Nxc5eTkqGPHjo79Pj4+ioiIUHFx8UX+hAAAU3FJJAAAZahevbqaNWumNm3aaNGiRdqyZYsWLlwoSXr++ec1d+5cjR07Vv/617+UlZWl6OhoFRQUVHgdX19fp8deXl6y2+2V8jMAAC5PBDYAACrA29tbTz75pMaPH6/ffvtNn332me655x49+OCDatOmja655hp9/fXXTs+xWCwqKiq6qHWDg4MVEhKibdu2ObYVFRVpx44dF3VcAIDZCGwAAFTQ/fffLx8fH6WkpOi6666TzWbTpk2b9NVXX+mvf/2rcnJynOqbNm2qLVu26Pvvv9fPP/98wZcwPvbYY5o2bZo++OAD7du3T6NGjdLRo0fl5eVVGT8WAMBABDYAACqoWrVqio+P14wZM/S3v/1N7du3V3R0tLp27arQ0NBSt9n/+9//Lh8fH7Vs2VL16tXTgQMHLmjdsWPHasCAAXrooYdktVpVo0YNRUdHy9/fvxJ+KgCAibzsXBwPAMBlqbi4WC1atNBf/vIXTZ061dPtAAAuAe4SCQDAZeKHH35Qenq6br/9duXn52v+/Pnav3+/HnjgAU+3BgC4RLgkEgCAy4S3t7fS0tJ000036dZbb9WuXbu0Zs0atWjRwtOtAQAuES6JBAAAAABDcYYNAAAAAAxFYAMAAAAAQxHYAAAAAMBQBDYAAAAAMBSBDQAAAAAMRWADAAAAAEMR2AAAAADAUAQ2AAAAADDU/wP+jTudJimPhgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualisasi distribusi rating\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.countplot(x='Book-Rating', data=ratings)  # Hapus palette\n",
    "plt.title('Distribusi Nilai Rating Buku')\n",
    "plt.xlabel('Rating')\n",
    "plt.ylabel('Jumlah')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FyQzl1Oy9Vqz"
   },
   "source": [
    "**Distribusi Nilai Rating**\n",
    "\n",
    "Distribusi data rating menunjukkan bahwa **sebagian besar nilai rating adalah 0**. Selain itu, rating tinggi di rentang **8 hingga 10** juga cukup sering diberikan, yang mengindikasikan adanya **kecenderungan positif** dari pengguna terhadap buku-buku yang mereka sukai.\n",
    "\n",
    "**Insight:**\n",
    "\n",
    "* Dominasi rating **0** mengisyaratkan adanya banyak data yang bersifat **kosong** atau merupakan **nilai default**, yang tidak merepresentasikan preferensi pengguna secara akurat.\n",
    "* Hal ini perlu mendapat perhatian khusus dalam tahap **pemodelan sistem rekomendasi**, karena bisa menyebabkan **bias atau distorsi** pada hasil prediksi rekomendasi.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 564
    },
    "id": "JFZb7ObW9WAl",
    "outputId": "be33861c-2067-4c0b-8a51-c7d7133128ed"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA2QAAAIjCAYAAABswtioAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8ekN5oAAAACXBIWXMAAA9hAAAPYQGoP6dpAAB46klEQVR4nO3de3yT5f3/8fedY9MzBdqCHOQkBzkpKjIdoCIH8TTZpps6xLODbYhTv+ynCKhzcxPUiTKnwuamczqdThzCQGAonjh4RAREqkBbCrSlx5zu3x9pQksLNDRp0uT1fDxQmly580l6UfLmOhmmaZoCAAAAALQ6S6wLAAAAAIBkRSADAAAAgBghkAEAAABAjBDIAAAAACBGCGQAAAAAECMEMgAAAACIEQIZAAAAAMQIgQwAAAAAYoRABgAAAAAxQiADgCQye/ZsGYbRKs81evRojR49OvT1qlWrZBiGXnrppVZ5/qDFixfLMAx9/fXXrfq8Uuu+3wCAtolABgBtVDBoBH+lpKSoc+fOGjdunB599FEdPHgwIs+ze/duzZ49W5s2bYrI9eLVNddco/T09CPen56ermuuuab1CmpCMOAFf6WmpmrAgAG66667VF5eHtPaAADHh0AGAG3c3Llz9eyzz+qJJ57Qz372M0nS9OnTNWjQIH388ccN2t51112qrq4O6/q7d+/WnDlzwg5ky5Yt07Jly8J6TDRcffXVqq6uVvfu3Vv9uY/n/W6OJ554Qs8++6zmzZunfv366f7779f48eNlmmbEnwsAEF22WBcAAGiZCRMm6LTTTgt9PXPmTK1cuVIXXnihLr74Ym3evFkul0uSZLPZZLNF90d/VVWVUlNT5XA4ovo8zWW1WmW1WmPy3NF6v7///e+rQ4cOkqSbb75ZkyZN0ssvv6x3331XI0aMiPjzAQCihxEyAEhA5557ru6++27t3LlTf/3rX0O3N7Wmafny5Tr77LOVnZ2t9PR09e3bV7/61a8kBdZ9nX766ZKkKVOmhKbKLV68WFJgndjAgQO1fv16jRw5UqmpqaHHHr6GLMjn8+lXv/qV8vPzlZaWposvvljffPNNgzYnnnhik9MDm7rmH/7wB5188slKTU1Vu3btdNppp+m5554L3R+tNWQej0dz5sxRnz59lJKSovbt2+vss8/W8uXLQ22aer8XLVqkc889V7m5uXI6nRowYICeeOKJFtVy7rnnSpJ27NghSfL7/Xr44Yd18sknKyUlRXl5ebrpppt04MCBBo878cQTdeGFF2rt2rU644wzlJKSop49e+ovf/lLo+f4+OOPNWrUKLlcLnXp0kX33XefFi1a1Oi99fv9mj17tjp37qzU1FSdc845+vzzzxt9T4+0vq6p71dz69y/f79++ctfatCgQUpPT1dmZqYmTJigjz76KNy3FABaDSNkAJCgrr76av3qV7/SsmXLdMMNNzTZ5rPPPtOFF16owYMHa+7cuXI6ndq2bZvefvttSVL//v01d+5czZo1SzfeeKO++93vSpK+853vhK6xb98+TZgwQVdccYWuuuoq5eXlHbWu+++/X4Zh6M4771RxcbEefvhhjRkzRps2bQqN5DXXn/70J/385z/X97//ff3iF79QTU2NPv74Y7333nv68Y9/HNa1wjV79mw98MADuv7663XGGWeovLxcH374oTZs2KDzzz//iI974okndPLJJ+viiy+WzWbTv//9b/30pz+V3+/X1KlTj6uW7du3S5Lat28vSbrpppu0ePFiTZkyRT//+c+1Y8cOPfbYY9q4caPefvtt2e320GO3bdum73//+7ruuus0efJkPfPMM7rmmms0bNgwnXzyyZKkXbt26ZxzzpFhGJo5c6bS0tL01FNPyel0Nqpl5syZevDBB3XRRRdp3Lhx+uijjzRu3DjV1NQc12sLp86vvvpK//rXv/SDH/xAPXr0UFFRkf74xz9q1KhR+vzzz9W5c+cW1QAA0UAgA4AE1aVLF2VlZYU+rDdl+fLlcrvd+s9//hOaAldfXl6eJkyYoFmzZmnEiBG66qqrGrUpLCzUwoULddNNNzWrrv3792vz5s3KyMiQJJ166qn64Q9/GApX4ViyZIlOPvlkvfjii2E9LhKWLFmiCy64QE8++WRYj1u9enWD4Dlt2jSNHz9e8+bNa3Yg279/vySpoqJCy5Yt0+OPP668vDx997vf1dq1a/XUU0/pb3/7W4NQes4552j8+PF68cUXG9y+ZcsWrVmzJhS2f/jDH6pr165atGiRfv/730uSfvvb3+rAgQPasGGDhg4dKikwYtqnT58GdRUVFWnevHm69NJL9corr4RunzNnjmbPnt38N6kJzalz0KBB+vLLL2WxHJoAdPXVV6tfv356+umndffdd7eoBgCIBqYsAkACS09PP+pui9nZ2ZKkV199VX6//7iew+l0asqUKc1u/5Of/CQUxqTAeqhOnTrpjTfeCPu5s7Oz9e233+qDDz4I+7EtlZ2drc8++0xbt24N63H1w1hZWZlKSko0atQoffXVVyorK2vWNfr27auOHTuqR48euummm9S7d28tWbJEqampevHFF5WVlaXzzz9fJSUloV/Dhg1Tenq63nrrrQbXGjBgQCjkSFLHjh3Vt29fffXVV6Hbli5dqhEjRoTCmCTl5OToyiuvbHCtFStWyOv16qc//WmD24ObzbREc+p0Op2hMObz+bRv377QNNwNGza0uAYAiAYCGQAksIqKigbh53CXX365zjrrLF1//fXKy8vTFVdcoX/84x9hhbMTTjghrA08Dh9VMQxDvXv3Pq41XnfeeafS09N1xhlnqE+fPpo6dWpoumU01F/zNHfuXJWWluqkk07SoEGDdPvttzfa1bIpb7/9tsaMGaO0tDRlZ2erY8eOoXV3zQ1k//znP7V8+XKtWrVK27Zt06effqphw4ZJkrZu3aqysjLl5uaqY8eODX5VVFSouLi4wbW6devW6Prt2rVrsN5s586d6t27d6N2h9+2c+fOJm/PyclRu3btmvXajqQ5dfr9fs2fP199+vSR0+lUhw4d1LFjR3388cfNfm8BoLUxZREAEtS3336rsrKyJj9IB7lcLq1Zs0ZvvfWWlixZoqVLl+qFF17Queeeq2XLljVrd8Jw1301x5EOU/b5fA1q6t+/v7Zs2aLXX39dS5cu1T//+U89/vjjmjVrlubMmRPWc6akpKi2tlamaTZ6ftM0VVNTo5SUlNBtI0eO1Pbt2/Xqq69q2bJleuqppzR//nwtXLhQ119/fZPPsX37dp133nnq16+f5s2bp65du8rhcOiNN97Q/Pnzmx2ER44c2eQUUykQSnJzc/W3v/2tyfs7duzY4OsjfY+jvYX+0b7HTWlOnb/+9a91991369prr9W9996rnJwcWSwWTZ8+/bhHgAEg2ghkAJCgnn32WUnSuHHjjtrOYrHovPPO03nnnad58+bp17/+tf7f//t/euuttzRmzJgjfnA+XodP8TNNU9u2bdPgwYNDt7Vr106lpaWNHrtz50717NmzwW1paWm6/PLLdfnll8vtduuyyy7T/fffr5kzZzYIUMfSvXt3eb1ebd++vVGI3bZtm3w+X6OzzHJycjRlyhRNmTJFFRUVGjlypGbPnn3EQPbvf/9btbW1eu211xqM+Bw+jbAlevXqpf/+978666yzIhaWu3fvrm3btjW6/fDbgu/Ptm3b1KNHj9Dt+/bta7TDY3DErLS0NDR1Vjo0ynY8XnrpJZ1zzjl6+umnG9xeWlp6xAALALHGlEUASEArV67Uvffeqx49ejRa51NfcHOI+oLrhGprayUFAo+kJgPS8fjLX/7SYF3bSy+9pD179mjChAmh23r16qV3331Xbrc7dNvrr7/eaHv8ffv2Nfja4XBowIABMk1THo8nrLqCz//YY481um/BggUN2jT13Onp6erdu3fofWtKcJSn/qhOWVmZFi1aFFatR/PDH/5QPp9P9957b6P7vF7vcX0fx40bp3Xr1jU4HHz//v2NRuHOO+882Wy2Rtv4N/We9urVS5K0Zs2a0G2VlZX685//HHZ9QVartdHI3osvvqhdu3Yd9zUBINoYIQOANu4///mPvvjiC3m9XhUVFWnlypVavny5unfvrtdee+2oo0Rz587VmjVrNHHiRHXv3l3FxcV6/PHH1aVLF5199tmSAh+cs7OztXDhQmVkZCgtLU3Dhw9vMAISjpycHJ199tmaMmWKioqK9PDDD6t3794Ntua//vrr9dJLL2n8+PH64Q9/qO3bt+uvf/1r6EN80NixY5Wfn6+zzjpLeXl52rx5sx577DFNnDjxqGvnmjJ06FBdf/31euSRR7R169bQ1vXLly/XG2+8oeuvv15DhgwJtR8wYIBGjx6tYcOGKScnRx9++KFeeuklTZs27YjPMXbsWDkcDl100UW66aabVFFRoT/96U/Kzc3Vnj17wqr3SEaNGqWbbrpJDzzwgDZt2qSxY8fKbrdr69atevHFF/XII4/o+9//fljXvOOOO/TXv/5V559/vn72s5+Ftr3v1q2b9u/fHxpFzcvL0y9+8Qs99NBDuvjiizV+/Hh99NFHoV0864+2jh07Vt26ddN1112n22+/XVarVc8884w6duyogoKC43rtF154oebOnaspU6boO9/5jj755BP97W9/azSqCgBxxQQAtEmLFi0yJYV+ORwOMz8/3zz//PPNRx55xCwvL2/0mHvuuces/6N/xYoV5iWXXGJ27tzZdDgcZufOnc0f/ehH5pdfftngca+++qo5YMAA02az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79cEAACA6OBTINqMUCBr4zssBgVH+oKvCwAAAMknpoHsiSee0ODBg5WZmanMzEyNGDFC//nPf0L319TUaOrUqWrfvr3S09M1adIkFRUVNbhGQUGBJk6cqNTUVOXm5ur222+X1+tt0GbVqlU69dRT5XQ61bt3by1evLhRLQsWLNCJJ56olJQUDR8+XO+//35UXjOOj9805W4jm3o0V0rdSF+Nj63vAQAAklVMA1mXLl30m9/8RuvXr9eHH36oc889V5dccok+++wzSdKtt96qf//733rxxRe1evVq7d69W5dddlno8T6fTxMnTpTb7dY777yjP//5z1q8eLFmzZoVarNjxw5NnDhR55xzjjZt2qTp06fr+uuv15tvvhlq88ILL2jGjBm65557tGHDBg0ZMkTjxo1TcXFx670ZOKpgGDMkJcrsvmCwrGGEDAAAIGnFdA3ZRRdd1ODr+++/X0888YTeffdddenSRU8//bSee+45nXvuuZKkRYsWqX///nr33Xd15plnatmyZfr888/13//+V3l5eRo6dKjuvfde3XnnnZo9e7YcDocWLlyoHj166KGHHpIk9e/fX2vXrtX8+fM1btw4SdK8efN0ww03aMqUKZKkhQsXasmSJXrmmWf0f//3f03WXltbq9ra2tDX5eXlkiSPxyOPxxPZNypMweePdR3Hw+/3y+VyyTD9Mn2HRjpr6w5Qdlok+X063ghjMQy5XC5ZpAbXj4Rwr51iBF5FtcffrPaGGXhv/H5/1L+3bbkPIfboP2gp+hBagv6DlohU/wnn8YZpmnHxz/M+n08vvviiJk+erI0bN6qwsFDnnXeeDhw4oOzs7FC77t27a/r06br11ls1a9Ysvfbaa9q0aVPo/h07dqhnz57asGGDTjnlFI0cOVKnnnqqHn744VCbRYsWafr06SorK5Pb7VZqaqpeeuklXXrppaE2kydPVmlpqV599dUm6509e7bmzJnT6PbnnntOqampLX07cJiy1I7a3eEkpdaUqnvxZ7EuJyKqHBnamT9YNm+t+uz+MNblAAAAIEKqqqr04x//WGVlZcrMzDxq25jvsvjJJ59oxIgRqqmpUXp6ul555RUNGDBAmzZtksPhaBDGJCkvL0+FhYWSpMLCQuXl5TW6P3jf0dqUl5erurpaBw4ckM/na7LNF198ccS6Z86cqRkzZoS+Li8vV9euXTV27NhjvunR5vF4tHz5cp1//vmy2+0xrSVcH330kUaOHKk/vPym+gwcHLq9stInVfmVltVO7bqOPu7rr3j1Zf3u9qma/afndeao479OJK6d4Te1c59XXptTGaeMku0Ym5Vs/fRj/eyycVqzZo2GDBkSoaqb1pb7EGKP/oOWog+hJeg/aIlI9Z/g7LnmiHkg69u3rzZt2qSysjK99NJLmjx5slavXh3rso7J6XTK6XQ2ut1ut8fNH/54qqW5LBaLqqurZRoWGdZD3dNdd06X02ZtcHu4/Kap6upq+aUWXScS17ZbJYfFK7dfqjGtyjjGdv6mEXhvLBZLq31f22IfQvyg/6Cl6ENoCfoPWqKl/Secx8Y8kDkcDvXu3VuSNGzYMH3wwQd65JFHdPnll8vtdqu0tLTBKFlRUZHy8/MlSfn5+Y12Qwzuwli/zeE7MxYVFSkzM1Mul0tWq1VWq7XJNsFrIPY8dZt6OBJky/sgl80it9uvKq9fGY4E2a0EAAAAzRZ3nwD9fr9qa2s1bNgw2e12rVixInTfli1bVFBQoBEjRkiSRowYoU8++aTBbojLly9XZmamBgwYEGpT/xrBNsFrOBwODRs2rEEbv9+vFStWhNog9ur29JA9Qba8D0q1BV5PlTculnICAACglYU9Qvbtt9/qtddeU0FBgdxud4P75s2bF9a1Zs6cqQkTJqhbt246ePCgnnvuOa1atUpvvvmmsrKydN1112nGjBnKyclRZmamfvazn2nEiBE688wzJUljx47VgAEDdPXVV+vBBx9UYWGh7rrrLk2dOjU0nfDmm2/WY489pjvuuEPXXnutVq5cqX/84x9asmRJqI4ZM2Zo8uTJOu2003TGGWfo4YcfVmVlZWjXRcRecIQsUba8D3LZLJJ8quYsMgAAgKQUViBbsWKFLr74YvXs2VNffPGFBg4cqK+//lqmaerUU08N+8mLi4v1k5/8RHv27FFWVpYGDx6sN998U+eff74kaf78+bJYLJo0aZJqa2s1btw4Pf7446HHW61Wvf7667rllls0YsQIpaWlafLkyZo7d26oTY8ePbRkyRLdeuuteuSRR9SlSxc99dRToS3vJenyyy/X3r17NWvWLBUWFmro0KFaunRpo40+EDvuBJ2yyAgZAABAcgsrkM2cOVO//OUvNWfOHGVkZOif//yncnNzdeWVV2r8+PFhP/nTTz991PtTUlK0YMECLViw4IhtunfvrjfeeOOo1xk9erQ2btx41DbTpk3TtGnTjtoGseHzm6rLY7InXCALDPnV+kz5/KasCfb6AAAAcHRhTQDbvHmzfvKTn0iSbDabqqurlZ6errlz5+q3v/1tVAoEgtMVLZISbAmZ7BZDdYNkqvYxSgYAAJBswgpkaWlpoXVjnTp10vbt20P3lZSURLYyoI47tH7MkGEkWCLToVGyKi/ryAAAAJJNWFMWzzzzTK1du1b9+/fXBRdcoNtuu02ffPKJXn755dBGG0CkeUI7LMa2jmhJtRkq90jVrCMDAABIOmEFsnnz5qmiokKSNGfOHFVUVOiFF15Qnz59wt5hEWiuRN3QIyi40yIjZAAAAMknrEDWs2fP0O/T0tK0cOHCiBcEHM5Tb8piImKnRQAAgOQV9jlkkuR2u1VcXCy/v+G/6Hfr1i0iRQH1eXyJHchcdWvIanym/KYpSwKukwMAAEDTwgpkX375pa677jq98847DW43TVOGYcjn80W0OEA6NEKWqFMWHZbA7pE+M7COLM2emK8TAAAAjYUVyKZMmSKbzabXX39dnTp1Ssgd7xB/3KFNPRKzvxmGoVSbRQc9flV7/Uqzh7X5KQAAANqwsALZpk2btH79evXr1y9a9QCNHBohi3EhUeSyGTroYR0ZAABAsgnrI+6AAQM4bwytzp3gm3pI9Tb24HBoAACApBJWIPvtb3+rO+64Q6tWrdK+fftUXl7e4BcQaT6/qbo8luCBLPBHsZqt7wEAAJJKWFMWx4wZI0k677zzGtzOph6IluB0RYsCG18kKlfdCFm11wz9eQIAAEDiCyuQvfXWW9GqA2hSaLqi1UjokOK0GLIYkt+Uqn1maAojAAAAEltYgWzUqFHRqgNokie4w2ICb+ghBXZaTKvbabHC4w9NYQQAAEBiCyuQrVmz5qj3jxw5skXFAIdzJ/gZZPVl2AM7LVZ4/Mp1xboaAAAAtIawAtno0aMb3VZ/GhlryBBpniTYYTEo3W6R5NNBDxt7AAAAJIuw5kUdOHCgwa/i4mItXbpUp59+upYtWxatGpHEPL7kCWQZdfMyKz2m/Cbb3wMAACSDsEbIsrKyGt12/vnny+FwaMaMGVq/fn3ECgOkelMWE3mLxTpOqyGbIXlNqdJrKsOe+K8ZAAAg2UVk54C8vDxt2bIlEpcCGji0qUfihxPDMEKjZBVupi0CAAAkg7BGyD7++OMGX5umqT179ug3v/mNhg4dGsm6AEmH1pA5kmTTwXS7RQfcfh30+NUp1sUAAAAg6sIKZEOHDpVhGDIPW99y5pln6plnnoloYYBU7xyyJBghk6R0h0WqDOy0CAAAgMQXViDbsWNHg68tFos6duyolJSUiBYFSJLPb6oujyVNIAtOWaz2mfL6TdmS5HUDAAAkq7ACWffu3aNVB9BIcLqiRVIS7OkhKRA8nVZDtT5TFR6/sp3WWJcEAACAKAorkD366KNN3m4YhlJSUtS7d2+NHDlSVisfItFyoemKVqPBeXeJLsNuUa3PRyADAABIAmEFsvnz52vv3r2qqqpSu3btJAXOJktNTVV6erqKi4vVs2dPvfXWW+ratWtUCkbyCC6jSpYNPYLS7YZKasQB0QAAAEkgrI+6v/71r3X66adr69at2rdvn/bt26cvv/xSw4cP1yOPPKKCggLl5+fr1ltvjVa9SCLJtqFHUGjrewIZAABAwgtrhOyuu+7SP//5T/Xq1St0W+/evfX73/9ekyZN0ldffaUHH3xQkyZNinihSD4eX3IGsjRbIJC5/VKtz5QzWRbQAQAAJKGwRsj27Nkjr9fb6Hav16vCwkJJUufOnXXw4MHIVIekdugMsuQKJFaLoTRb4DUzSgYAAJDYwgpk55xzjm666SZt3LgxdNvGjRt1yy236Nxzz5UkffLJJ+rRo0dkq0RSqr+pR7JJr5u2yDoyAACAxBZWIHv66aeVk5OjYcOGyel0yul06rTTTlNOTo6efvppSVJ6eroeeuihqBSL5HJoU48kDmRuAhkAAEAiC2sNWX5+vpYvX64vvvhCX375pSSpb9++6tu3b6jNOeecE9kKkbQ8oU09YlxIDGQ6Dm3s4TdNWZJo238AAIBkElYgC+rXr5/69esX6VqABpJ1l0VJclkN2QzJawZCWaaD88gAAAASUViBbMaMGU3eXv9g6EsuuUQ5OTkRKQ7JyzQM1eWxpAxkhmEo02HR/lq/yt0EMgAAgEQVViDbuHGjNmzYIJ/PF5qm+OWXX8pqtapfv356/PHHddttt2nt2rUaMGBAVApGcvAbga5pkZSEe3pI0qFAxsYeAAAACSus1TmXXHKJxowZo927d2v9+vVav369vv32W51//vn60Y9+pF27dmnkyJEcDI0W81sCI0J2iyEjSddPZdoD78FBt1+maca4GgAAAERDWIHsd7/7ne69915lZmaGbsvKytLs2bP14IMPKjU1VbNmzdL69esjXiiSSyiQJfFMvTS7IUvdOrIqL4EMAAAgEYUVyMrKylRcXNzo9r1796q8vFySlJ2dLbfbHZnqkLRMS2DKYjKuHwuyGIYy6raYZNoiAABAYgp7yuK1116rV155Rd9++62+/fZbvfLKK7ruuut06aWXSpLef/99nXTSSdGoFUmk/pTFZJYZDGScRwYAAJCQwtrU449//KNuvfVWXXHFFfJ6vYEL2GyaPHmy5s+fLymwJf5TTz0V+UqRVAhkAZkOi1QZCGQZsS4GAAAAERdWIEtPT9ef/vQnzZ8/X1999ZUkqWfPnkpPTw+1GTp0aEQLRHI6NGUxxoXEWIbdIkOBM9n8luM6NhAAAABx7Lg+4aWnp2vw4MGRrgUI8RuMkEmS1WIozW6owmPKY0+NdTkAAACIsLACWWVlpX7zm99oxYoVKi4ult/fcF1LcNQMaCmmLB6SabeqwuOV1+GKdSkAAACIsLAC2fXXX6/Vq1fr6quvVqdOnZL2fChEn0kgC8l0WLS7SvLYCWQAAACJJqxA9p///EdLlizRWWedFa16AEkKrZdK9jVkUt3GHpJ8NqdSs3NiXA0AAAAiKayPu+3atVNODh8IEV0p6ZlS3egrI2SB9yDVFngfeg7jH0MAAAASSViB7N5779WsWbNUVVUVrXoAped0kCRZjcDhyJCy60bJTjrrvBhXAgAAgEgKa8riQw89pO3btysvL08nnnii7HZ7g/s3bNgQ0eKQnNLaBQIZo2OHtHNatbvKp75nnSdT7liXAwAAgAgJK5BdeumlUSoDOCQtJxjIYlxIHMl0WCTTr8yO+Tpo7ol1OQAAAIiQsALZPffcE606gJC0du0lMUJWn8UwZHdXyeNMV4nYbREAACBRhD0GUVpaqqeeekozZ87U/v37JQWmKu7atSvixSE5pbfrKIlAdjiHu1KSVKKUGFcCAACASAlrhOzjjz/WmDFjlJWVpa+//lo33HCDcnJy9PLLL6ugoEB/+ctfolUnkkh6DmvImuKorVBlRp7K5FSN168UG3M6AQAA2rqwPtHNmDFD11xzjbZu3aqUlEP/Sn/BBRdozZo1ES8OyYkpi02z+r0q3vGlTMPQ1wc9sS4HAAAAERBWIPvggw900003Nbr9hBNOUGFhYcSKQnILburBAFBjX769QpK0vZydFgEAABJBWB95nU6nysvLG93+5ZdfqmPHjhErCsktvW7bewcjZI1sqQtkO8o9Mk0zxtUAAACgpcIKZBdffLHmzp0rjycwXcowDBUUFOjOO+/UpEmTolIgkg/nkB3Zjg3rZDH9qvD6VVzti3U5AAAAaKGwAtlDDz2kiooK5ebmqrq6WqNGjVLv3r2VkZGh+++/P1o1IomYOrSGzEYga8Tncau9aiRJXzFtEQAAoM0La5fFrKwsLV++XG+//bY++ugjVVRU6NRTT9WYMWOiVR+SjEcWWaxWSRwMfSQdVKO9StWXZW6NyE+NdTkAAABogWYHshdeeEGvvfaa3G63zjvvPP30pz+NZl1IUu66QVvD75PFYISsKbmq0hYjR3uqvCqs8io/Nax/VwEAAEAcadYYxBNPPKEf/ehH+vDDD7V161ZNnTpVt99+e7RrQxJyKzA6ZvGzPupInPKrf7ZTkvTh3uoYVwMAAICWaFYge+yxx3TPPfdoy5Yt2rRpk/785z/r8ccfj3ZtSEKHRsi8Ma4kvg3rGDgHcPOBWlV6/DGuBgAAAMerWYHsq6++0uTJk0Nf//jHP5bX69WePXuiVhiSEyNkzdM5za5OqTb5TGnTvppYlwMAAIDj1KxAVltbq7S0tEMPsljkcDhUXc10KURWKJCZBLJjOa1ulGzj3hr5OJMMAACgTWr2bgB33323UlMP7ejmdrt1//33KysrK3TbvHnzIlsdkg5TFpuvX7ZTK3dVqsLr15ZStwa0c8a6JAAAAISpWYFs5MiR2rJlS4PbvvOd7+irr74KfW2wIx4iwMOUxWazWgyd0sGltYVVWr+3mkAGAADQBjUrkK1atSrKZQABwREyAlnzDO2QoneKqrSr0qs9lR51SrPHuiQAAACEgaN3EVeCa8iYstg86XZLaAv8D/ayuQcAAEBbQyBDXGGELHyn57okBbbAL3fzvgEAALQlBDLEDb9pymOwhixc+ak2dUu3y5T0IaNkAAAAbQqBDHGj2hvYut3v98tg2/uwnFE3SvZRSY1qfRwUDQAA0FYQyBA3Kr2BIFFVtl/s2RmeXpl25TitqvWb+nhfbazLAQAAQDM1+xyyoNLSUr3//vsqLi6W39/wX+J/8pOfRKwwJJ+qukBWub9EyiCShcMwDJ2R69LSbyr04d5qDeuYIgtHUQAAAMS9sALZv//9b1155ZWqqKhQZmZmg7PHDMMgkKFFquqmLFYe2CdldIhxNW3PyTlOrd5TqTK3X1+WutWPc8kAAADiXlhTFm+77TZde+21qqioUGlpqQ4cOBD6tX///mjViCRR5QmMkFXsL4lxJW2T3WLo1A4pkqT3i6tjXA0AAACaI6xAtmvXLv385z9XampqtOpBEgtNWTxAIDtep3ZwyWpIu6sCB0UDAAAgvoUVyMaNG6cPP/wwWrUgyR2askggO15pdov61R0Uvb6ELfABAADi3THXkL322muh30+cOFG33367Pv/8cw0aNEh2u71B24svvjjyFSJpHBoh2xfjStq2YR1T9NmBWm0+UKtzO6cp1c5mqgAAAPHqmIHs0ksvbXTb3LlzG91mGIZ8Ps6OwvELBrIKAlmLdE6zq1OqTXuqvPpoX41G5DPFGAAAIF4d85/O/X5/s34RxtBSwYOhq0oJZC0V3NxjY0mN/KYZ42oAAABwJGGfQwZES5WvbspiKTt2Hs3mzZuP2cYnya4TVO6R3tz4hfJ07F0XO3TooG7dukWgQgAAADRX2IGssrJSq1evVkFBgdxud4P7fv7zn0esMCQXv2mqJjRCRiBryr7iQskwdNVVVzWr/dipv9I5192qlz/8Qk/ddNkx26empmrz5s2EMgAAgFYUViDbuHGjLrjgAlVVVamyslI5OTkqKSlRamqqcnNzCWQ4brU+U8GJdZVMWWxSRVmZZJqadu9DGnL68GO291lsOmCa6nX6d/XE8vdl87mP2Hbnti26b9p1KikpIZABAAC0orAC2a233qqLLrpICxcuVFZWlt59913Z7XZdddVV+sUvfhGtGpEEght62Ey//F5vjKuJbyf06KW+g4c2q+0XB2q1r9Yv6wm91TfbEd3CAAAAELaw9sPetGmTbrvtNlksFlmtVtXW1qpr16568MEH9atf/SpaNSIJBM8gs4vNYSKpa3rgaIp9NT4d9PhjXA0AAAAOF1Ygs9vtslgCD8nNzVVBQYEkKSsrS998803kq0PSqK4bIXOI0BBJaXaLOqZYJUk7D3pksuMiAABAXAlryuIpp5yiDz74QH369NGoUaM0a9YslZSU6Nlnn9XAgQOjVSOSQHVohIxAFmndMmwqqfGpzO1Xqduvdk5rrEsCAABAnbBGyH7961+rU6dOkqT7779f7dq10y233KK9e/fqySefjEqBSA5VoREypixGWorVok6pjJIBAADEo7AC2WmnnaZzzjlHUmDK4tKlS1VeXq7169dryJAhYT/5Aw88oNNPP10ZGRnKzc3VpZdeqi1btjRoU1NTo6lTp6p9+/ZKT0/XpEmTVFRU1KBNQUGBJk6cGNrt8fbbb5f3sI0hVq1apVNPPVVOp1O9e/fW4sWLG9WzYMECnXjiiUpJSdHw4cP1/vvvh/2acHyCgYwRsujokm6X1ZAqvaZKagi9AAAA8SKsQBZpq1ev1tSpU/Xuu+9q+fLl8ng8Gjt2rCorK0Ntbr31Vv373//Wiy++qNWrV2v37t267LJDZyr5fD5NnDhRbrdb77zzjv785z9r8eLFmjVrVqjNjh07NHHiRJ1zzjnatGmTpk+fruuvv15vvvlmqM0LL7ygGTNm6J577tGGDRs0ZMgQjRs3TsXFxa3zZiS5al9g1IYRsuiwWwydkBaYobzzoFc+P6NkAAAA8eCYa8hOOeUUGYbRrItt2LAhrCdfunRpg68XL16s3NxcrV+/XiNHjlRZWZmefvppPffcczr33HMlSYsWLVL//v317rvv6swzz9SyZcv0+eef67///a/y8vI0dOhQ3Xvvvbrzzjs1e/ZsORwOLVy4UD169NBDDz0kSerfv7/Wrl2r+fPna9y4cZKkefPm6YYbbtCUKVMkSQsXLtSSJUv0zDPP6P/+7//Cel0IXxWbekRd51Sbiqp8qvWb+vqgR72y2AYfAAAg1o4ZyC699NJWKCOgrKxMkpSTkyNJWr9+vTwej8aMGRNq069fP3Xr1k3r1q3TmWeeqXXr1mnQoEHKy8sLtRk3bpxuueUWffbZZzrllFO0bt26BtcItpk+fbokye12a/369Zo5c2bofovFojFjxmjdunVN1lpbW6va2trQ1+Xl5ZIkj8cjj8fTgneh5YLPH+s6wlFVtyW7ze+Vy+WSYfpl+iJ/HpnFMORyuWSRIn79aF47Ete3SOqVYdHnZT4VVvuU43Ar2xEYJDdMv1wul/x+f4M+3Jb6EOIH/QctRR9CS9B/0BKR6j/hPP6Ygeyee+5pUTHN5ff7NX36dJ111lmhHRsLCwvlcDiUnZ3doG1eXp4KCwtDbeqHseD9wfuO1qa8vFzV1dU6cOCAfD5fk22++OKLJut94IEHNGfOnEa3L1u2TKmpqc181dG1fPnyWJfQbPs6D5NsKSrdW6Tnn39eqt2vA+tXRfx5Tu2SE7i+FPHrR/Pakbx+u3Y9dCCjs77cV6WeezbKavrUQdLzzz+vXbt2adeuXaG2bakPIf7Qf9BS9CG0BP0HLdHS/lNVVdXstmFte19fRUWF/P6G08syMzOP93KaOnWqPv30U61du/a4r9GaZs6cqRkzZoS+Li8vV9euXTV27NgWvQ+R4PF4tHz5cp1//vmy2+0xraW5/vB5meSXOnfM0YSRF+oPL7+pPgMHR/x5Vrz6sn53+1TN/tPzOnPU6DZz7UheP9M09dEBr2rk1IHeI9Qn06atn36sn102TmvWrNGQIUPaZB9C/KD/oKXoQ2gJ+g9aIlL9Jzh7rjnCCmQ7duzQtGnTtGrVKtXU1IRuN01ThmHI5zu+DRmmTZum119/XWvWrFGXLl1Ct+fn58vtdqu0tLTBKFlRUZHy8/NDbQ7fDTG4C2P9NofvzFhUVKTMzEy5XC5ZrVZZrdYm2wSvcTin0ymn09nodrvdHjd/+OOplqPx+k2567J9isVUdXW1TMMiw3rc/15wRH4zcH2/FPHrR/Pakby+TVKfLIs+2e/W3lpT7T2GTMOi6upqWSyWBn2mrfQhxCf6D1qKPoSWoP+gJVraf8J5bFi7LF511VU6cOCAnnnmGa1YsUIrV67UypUr9dZbb2nlypVhF2qapqZNm6ZXXnlFK1euVI8ePRrcP2zYMNntdq1YsSJ025YtW1RQUKARI0ZIkkaMGKFPPvmkwW6Iy5cvV2ZmpgYMGBBqU/8awTbBazgcDg0bNqxBG7/frxUrVoTaIHqq6zb0MCTZxO5/rSHTYQ3tuvhlmVteKxt8AAAAxEJY/8z+0Ucfaf369erbt29Ennzq1Kl67rnn9OqrryojIyO05isrK0sul0tZWVm67rrrNGPGDOXk5CgzM1M/+9nPNGLECJ155pmSpLFjx2rAgAG6+uqr9eCDD6qwsFB33XWXpk6dGhrBuvnmm/XYY4/pjjvu0LXXXquVK1fqH//4h5YsWRKqZcaMGZo8ebJOO+00nXHGGXr44YdVWVkZ2nUR0VPlDYQwl82QwfrbVtMt3aYKj19lbr/Ks06QKzM71iUBAAAknbAC2emnn65vvvkmYoHsiSeekCSNHj26we2LFi3SNddcI0maP3++LBaLJk2apNraWo0bN06PP/54qK3VatXrr7+uW265RSNGjFBaWpomT56suXPnhtr06NFDS5Ys0a233qpHHnlEXbp00VNPPRXa8l6SLr/8cu3du1ezZs1SYWGhhg4dqqVLlzba6AORV+0LjJCl2iwSgazVWAxDfbMd+mhfrWrl0I9+8ycOHQAAAGhlYQWyp556SjfffLN27dqlgQMHNpobOXhweJswmOaxp6elpKRowYIFWrBgwRHbdO/eXW+88cZRrzN69Ght3LjxqG2mTZumadOmHbMmRFb9ETK0LrvFUP9shzbtrVafM0frS7Ncp8W6KAAAgCQSViDbu3evtm/f3mAan2EYLd7UA8ktuIYs1RbWkkZESJrdooyDe3Qw6wQVGJkqrPKqPWugAQAAWkVYn4Cvvfba0EHLX331lXbs2NHg/8DxqKoLZC4rgSxWnLUV2vD6C5Kk1bsrY1wNAABA8ghrhGznzp167bXX1Lt372jVgyRUXTdlMZUpizH134UPatjEH2rHQY++qfDGuhwAAICkENaQxLnnnquPPvooWrUgSYVGyJiyGFMHdheoiyokSf8rquEAAgAAgFYQ1gjZRRddpFtvvVWffPKJBg0a1GhTj4svvjiixSE51B8hq41xLcmup8pUaMnQnmqfrK6cWJcDAACQ8MIKZDfffLMkNdhSPohNPXC86o+QEchiyym/Tuvo0rqiau3N6iZ/M3ZCBQAAwPELa46Y3+8/4i/CGI5XtS84QsaUxXgwPNclp0WqdaTpizIOhgMAAIgmPgEjpkzTrDdCxqYe8SDFZtHpHZySpPUltc06LxAAAADHJ6wpi01NVaxv1qxZLSoGycftN+Wv+7zPCFn8GJzj0NtF1SqukfZUedU5jYPJAAAAoiGsQPbKK680+Nrj8WjHjh2y2Wzq1asXgQxhC27oYTMku4URsnjhslmUWblXZel52lBSQyADAACIkrAC2caNGxvdVl5ermuuuUbf+973IlYUkkdwuiKjY/GnXUWhytLztPlArc47IY1jCQAAAKKgxZ+wMjMzNWfOHN19992RqAdJpqpuhIz1Y/EnxV2h3BSLfKb08b6aWJcDAACQkCLyT95lZWUqKyuLxKWQZKoZIYtbhqShOYHNPTaW1LC5BwAAQBSENWXx0UcfbfC1aZras2ePnn32WU2YMCGihSE51D+DDPGnX7Zdq4tqVOr2a8dBj3pmOmJdEgAAQEIJK5DNnz+/wdcWi0UdO3bU5MmTNXPmzIgWhuRQzZTFuGa3GBqU49SHe2u0oaSGQAYAABBhYQWyHTt2RKsOJKkqH1MW490pHVL04d4abS9zq6TGqw4pYf3YAAAAwFE065PVZZddduwL2WzKz8/X+eefr4suuqjFhSE5BEfIUhkhi1vtU2zqk+XQ1jK33tpVqR/0yop1SQAAAAmjWcMSWVlZx/zlcrm0detWXX755ZxHhmYLburhsjJCFs9Gd06VRdL2co++PuiOdTkAAAAJo1kjZIsWLWr2BV9//XX99Kc/1dy5c4+7KCQPtr1vG9qn2HRKxxSt31ujlbsqdU1fuywG3zMAAICWiviwxNlnn63TTjst0pdFguJg6LbjrPxUOa2Giqt9+nR/bazLAQAASAgR/xScnZ2tl19+OdKXRQLym6ZqfME1ZASyeJdqs+g7eS5J0prdVXL7OJcMAACgpfgUjJip8R76QJ/ClMU2YVhHl7IcFlV4/Xq/uDrW5QAAALR5BDLETHDLe6fVkJX1SG2CzWJodOc0SdJ7xVU66PHFuCIAAIC2jQOFEDPBETKXlTAWLzZv3ixJ8vsDYfmjjz6SxdLw321MSVnKU5nfqX998o1O1v5mXbtDhw7q1q1bROsFAABo6whkiJnaujVITgJZzO0rLpQMQ1dddZUkyeVy6fnnn9fIkSNVXd14amLXQcP00z8v1Tdmqu788UXa8+Wnx3yO1NRUbd68mVAGAABQD4EMMVNTN2UxhTPIYq6irEwyTU279yENOX24DNMv1e7XH15+U6bR9PenvKZc7pRM3faX15VZ+q2OFqt3btui+6Zdp5KSEgIZAABAPQQyxAwjZPHnhB691HfwUJk+rw6sX6U+AwfLsDb9Y6LG69eGklp5HGnKPWmQclKsrVwtAABA28fQBGImuOV9CoGsTUqxWdQ5LRDWvj7okWmyDT4AAEC4CGSIGUbI2r4uaTbZDKnaZ+pArT/W5QAAALQ5BDLETGgNGYdCt1k2i6GOrsBUxZIatsAHAAAIF5+EETOMkCWGDnVrx/bX+uRj2iIAAEBYCGSIGdaQJYYMu0VOiyGfKZUybREAACAsBDLEDCNkicEwDLWvGyXby7RFAACAsBDIEDOcQ5Y4OtStIztQ45PPz7RFAACA5uKTMGKGEbLEkW4zlGI15FdgLRkAAACah0CGmDBNM7SGjEDW9hmGEdrcg90WAQAAmo9AhpjwmlJwZhubeiSGYCA7UOuXl2mLAAAAzUIgQ0wE148ZkhwWAlkiSLUZclkNmWLaIgAAQHMRyBATtd5D0xUNg0CWCAzDCG3uwbRFAACA5iGQISZYP5aY2jsDgazM7ZefQ6IBAACOiUCGmKjlUOiElGozZLcE1geWuzkkGgAA4FgIZIiJQ1ve0wUTiWEYauc8tLkHAAAAjo5Pw4iJQ4dCM0KWaLIdgR8rpW7WkQEAABwLgQwxwaHQiSu7boSsymuq1scoGQAAwNEQyBATNawhS1h2i6EMe+D7Wsq0RQAAgKMikCEmWEOW2LJD68iYtggAAHA0fBpGTLCGLLG1cwQCWanbL5Pt7wEAAI6IQIaYYA1ZYku3G7IZks+UDnqYtggAAHAkBDLERGgNmY1AlogMw6g3bZFABgAAcCQEMsQEa8gSH9vfAwAAHBufhhETrCFLfMEDois8pvyGNcbVAAAAxCcCGVqdaZqsIUsCDquh1LopqR6HK8bVAAAAxCcCGVqd1wxs9iAxQpboMuyBHzFeG4EMAACgKQQytLrg6JghyWEhkCWyYCDz2FNiXAkAAEB8IpCh1QXXjzmthgyDQJbIDo2Qpciw8OMGAADgcHxCQqtj/VjycNkMWQxJFotye5wU63IAAADiDoEMra7GW3cGGYEs4RmGofS6UbKuA4fFuBoAAID4QyBDq+MMsuQSnLbYZeApMa4EAAAg/vCJGK2u/hoyJL5gIOt68qkxrgQAACD+EMjQ6oIjZExZTA7BKYt5vfvLJ77nAAAA9dliXQCSTw2beiQVh0Wy+LySzaZy0xHrcgAAAOIKI2RodYdGyOh+ycAwDNm81ZKkMhHIAAAA6uMTMVpdLWvIko7NUyNJKieQAQAANEAgQ6urYQ1Z0gkGsjI5Y1wJAABAfCGQodVxMHTysXkDgazasKnK449xNQAAAPGDQIZWV8MasqRjMf0q3vGlJGl3lTfG1QAAAMQPPhGj1bGGLDl988kGSdKeKk+MKwEAAIgfBDK0utAImY1Alky+/SwQyHZXMkIGAAAQRCBDq/L6TdXlMUbIksy3n22UJBVWe2WaZoyrAQAAiA8EMrSq4OiYJDktBLJkUrhtswzTVLXX1EE29gAAAJBEIEMrq79+zDAIZMnE665VmgLrx4qqmbYIAAAgEcjQyjiDLLllyi1JKmSnRQAAAEkEMrQyziBLbhl1gayoyhfjSgAAAOIDgQytijPIklsmUxYBAAAa4FMxWhVnkCW34AjZQY9flWzsAQAAQCBD66rxsoYsmdlkKsdplcQoGQAAgEQgQytjDRnyXHWBjI09AAAACGRoXawhQ36qTVLggGgAAIBkx6ditCrWkCGvLpAxQgYAAEAgQyvjHDLkuQKBrNTtV42XjT0AAEByI5ChVbGGDC6bRZmOwI8eNvYAAADJjkCGVlVDIIOk/LpRsqJqDogGAADJjUCGVhVcQ8amHsmNdWQAAAABfCpGq2LKIqT6I2QEMgAAkNwIZGg1Xr+punOh2dQjyQVHyPbV+OSuC+kAAADJiECGVlNb74M3I2TJLd1uUZrNkClpbw2jZAAAIHkRyNBqauqdQWYYBLJkFzwgelclgQwAACSvmAayNWvW6KKLLlLnzp1lGIb+9a9/NbjfNE3NmjVLnTp1ksvl0pgxY7R169YGbfbv368rr7xSmZmZys7O1nXXXaeKiooGbT7++GN997vfVUpKirp27aoHH3ywUS0vvvii+vXrp5SUFA0aNEhvvPFGxF9vsmP9GOrrlm6XJO086I5xJQAAALET00BWWVmpIUOGaMGCBU3e/+CDD+rRRx/VwoUL9d577yktLU3jxo1TTU1NqM2VV16pzz77TMuXL9frr7+uNWvW6MYbbwzdX15errFjx6p79+5av369fve732n27Nl68sknQ23eeecd/ehHP9J1112njRs36tJLL9Wll16qTz/9NHovPglxKDTqOzHDIUn6psIrn8k6MgAAkJxssXzyCRMmaMKECU3eZ5qmHn74Yd1111265JJLJEl/+ctflJeXp3/961+64oortHnzZi1dulQffPCBTjvtNEnSH/7wB11wwQX6/e9/r86dO+tvf/ub3G63nnnmGTkcDp188snatGmT5s2bFwpujzzyiMaPH6/bb79dknTvvfdq+fLleuyxx7Rw4cJWeCeSAyNkqC/XZZXLZqjaa2pPpVdd6kbMAAAAkklMA9nR7NixQ4WFhRozZkzotqysLA0fPlzr1q3TFVdcoXXr1ik7OzsUxiRpzJgxslgseu+99/S9731P69at08iRI+VwOEJtxo0bp9/+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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualisasi distribusi usia pengguna\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.histplot(users['Age'].dropna(), bins=30, kde=True, color='skyblue')\n",
    "plt.title('Distribusi Usia Pengguna')\n",
    "plt.xlabel('Usia')\n",
    "plt.ylabel('Jumlah Pengguna')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "orau1js_9agu"
   },
   "source": [
    "**Distribusi Usia Pengguna:**\n",
    "Visualisasi data menunjukkan bahwa sebagian besar pengguna berada dalam rentang usia **20 hingga 50 tahun**. Namun, terdapat kejanggalan data pada usia **0 tahun dan di atas 100 tahun**, yang mengindikasikan keberadaan **outlier**.\n",
    "\n",
    "**Insight:**\n",
    "* Kelompok usia terbanyak berasal dari rentang **30 hingga 40 tahun**.\n",
    "* Nilai usia ekstrem seperti **0** dan **lebih dari 100 tahun** perlu dipertimbangkan untuk dibersihkan.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "o3CF8s6z9dcb"
   },
   "source": [
    "# Data Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cxJDvO9H9gMQ"
   },
   "source": [
    "## Penanganan Nilai Hilang pada Kolom Age\n",
    "\n",
    "Pada tahap ini, kami menangani data yang hilang (missing value) pada kolom **Age** dalam dataset pengguna (`Users`).\n",
    "\n",
    "Setelah pemeriksaan, ditemukan sekitar **110.762 data pengguna tidak memiliki nilai usia**. Karena proporsi nilai hilang ini cukup besar (sekitar **40%**), kami memutuskan untuk **menghapus baris data pengguna yang kolom Age-nya kosong** agar proses pemodelan tidak bias.\n",
    "\n",
    "**Metode yang digunakan:** Menghilangkan missing value pada kolom Age dengan menggunakan fungsi `dropna(subset=['Age'])`.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "XPdNVNdt9bAy",
    "outputId": "c0f672b0-2b03-411b-d4cb-952bcd20fd3b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Users setelah menghapus missing Age: (168096, 3)\n"
     ]
    }
   ],
   "source": [
    "users = users.dropna(subset=['Age'])\n",
    "\n",
    "print('Jumlah data Users setelah menghapus missing Age:', users.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "N8xCf7NA9m6R"
   },
   "source": [
    "**Hasil**: Awalnya, dataset **Users** memiliki **278.858 baris data**. Setelah proses penghapusan baris yang memiliki nilai kosong pada kolom **Age**, jumlah data berkurang menjadi **168.096 entri** dengan **3 kolom**.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QNt5grjD9q4o"
   },
   "source": [
    "## Penanganan Outlier pada Kolom Age\n",
    "\n",
    "Pada tahap ini, kami membersihkan nilai-nilai **outlier** di kolom **Age** pada dataset pengguna (`Users`).\n",
    "\n",
    "Dari eksplorasi data awal, ditemukan beberapa usia yang tidak masuk akal, seperti **0 tahun** dan **244 tahun**. Untuk meningkatkan kualitas data, kami melakukan **filtering** sehingga hanya menyertakan pengguna dengan usia di antara **5 hingga 100 tahun**.\n",
    "\n",
    "**Metode yang digunakan:** Memfilter data dengan kondisi usia **>= 5** dan **<= 100**.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "oEnGLfK09nPs",
    "outputId": "8a7324e7-8d23-46af-aec0-aa7ff90d1356"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Users setelah membersihkan outlier Age: (166848, 3)\n"
     ]
    }
   ],
   "source": [
    "users = users[(users['Age'] >= 5) & (users['Age'] <= 100)]\n",
    "\n",
    "print('Jumlah data Users setelah membersihkan outlier Age:', users.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0wBIrHIy9zbf"
   },
   "source": [
    "**Hasil**: Setelah menghapus nilai kosong di kolom Age, dataset Users memiliki **168.096 entri**. Setelah proses pembersihan outlier dengan menyaring usia antara **5 hingga 100 tahun**, jumlah data berkurang menjadi **166.848 entri** dengan **3 kolom**.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K1UIG9_S9244"
   },
   "source": [
    "## Penyaringan (Filtering) Data Ratings\n",
    "\n",
    "Pada tahap ini, kami memfilter dataset **Ratings** dengan menghapus semua data yang memiliki nilai **rating 0**.\n",
    "\n",
    "Rating 0 menunjukkan bahwa pengguna tidak secara aktif memberikan penilaian terhadap buku tersebut, sehingga data ini tidak memberikan informasi preferensi yang berarti untuk sistem rekomendasi.\n",
    "\n",
    "**Metode yang digunakan:** Menghapus baris data dengan kondisi `Book-Rating == 0` menggunakan teknik filtering.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ZPx7dazW9z5M",
    "outputId": "db7a8d8f-e435-4b28-a851-81e8c9455a51"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Ratings setelah menghapus rating 0: (433671, 3)\n"
     ]
    }
   ],
   "source": [
    "ratings = ratings[ratings['Book-Rating'] != 0]\n",
    "\n",
    "print('Jumlah data Ratings setelah menghapus rating 0:', ratings.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dbJLgrm099ar"
   },
   "source": [
    "**Hasil**: Dataset **Ratings** awalnya memiliki **1.149.780 entri**. Setelah menghapus data dengan nilai rating **0**, jumlah entri berkurang menjadi **433.671 entri** dengan **3 kolom**.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lH21WYNG-Awu"
   },
   "source": [
    "## Filtering Pengguna (users) dan Buku Berdasarkan Aktivitas Minimum\n",
    "\n",
    "Pada tahap ini, kami melakukan penyaringan data pengguna dan buku dengan kriteria aktivitas minimum sebagai berikut:\n",
    "\n",
    "* **Penyaringan Pengguna:** Mempertahankan hanya pengguna yang telah memberikan minimal **3 rating**.\n",
    "* **Penyaringan Buku:** Mempertahankan hanya buku yang menerima minimal **3 rating**.\n",
    "\n",
    "**Tujuan penyaringan ini adalah untuk:**\n",
    "\n",
    "* Memastikan hanya pengguna yang aktif memberikan rating yang diproses.\n",
    "* Memastikan buku yang dianalisis memiliki cukup banyak penilaian untuk memberikan informasi yang valid bagi sistem rekomendasi.\n",
    "\n",
    "**Metode yang digunakan:** Melakukan penyaringan berdasarkan jumlah interaksi menggunakan fungsi `value_counts()` dan `isin()`.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "0nK2bsC-99yk",
    "outputId": "034f8de9-2b8e-4185-f481-7e2c57298267"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data Ratings setelah filtering user dan book aktif: (203851, 3)\n"
     ]
    }
   ],
   "source": [
    "user_counts = ratings['User-ID'].value_counts()\n",
    "active_users = user_counts[user_counts >= 3].index\n",
    "ratings = ratings[ratings['User-ID'].isin(active_users)]\n",
    "\n",
    "# Filter buku yang menerima minimal 3 rating\n",
    "book_counts = ratings['ISBN'].value_counts()\n",
    "popular_books = book_counts[book_counts >= 3].index\n",
    "ratings = ratings[ratings['ISBN'].isin(popular_books)]\n",
    "\n",
    "print('Jumlah data Ratings setelah filtering user dan book aktif:', ratings.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zCymdwQE-N4g"
   },
   "source": [
    "**Hasil**: Setelah menghapus rating dengan nilai 0, dataset **Ratings** memiliki **433.671 entri**. Setelah dilakukan filtering untuk mempertahankan pengguna dan buku yang memiliki minimal 3 rating, jumlah entri berkurang menjadi **203.851** dengan **3 kolom**.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oWMjAF60-QKU"
   },
   "source": [
    "## Konversi Kolom Menjadi List\n",
    "\n",
    "Pada tahap ini, kami mengubah kolom-kolom utama dari dataset **Ratings** menjadi format **list** menggunakan fungsi `.tolist()`. Langkah ini bertujuan agar data lebih mudah dan fleksibel untuk diproses pada tahap berikutnya.\n",
    "\n",
    "**Kolom yang dikonversi meliputi:**\n",
    "\n",
    "* **User-ID** menjadi `list user_id`\n",
    "* **ISBN** menjadi `list isbn`\n",
    "* **Book-Rating** menjadi `list book_rating`\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Vkq3n4UZ-OSt",
    "outputId": "1180f9e8-7394-47d2-8fec-e2db80c80311"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data user_id: 203851\n",
      "Jumlah data isbn: 203851\n",
      "Jumlah data book_rating: 203851\n"
     ]
    }
   ],
   "source": [
    "user_id = ratings['User-ID'].tolist()\n",
    "isbn = ratings['ISBN'].tolist()\n",
    "book_rating = ratings['Book-Rating'].tolist()\n",
    "\n",
    "print('Jumlah data user_id:', len(user_id))\n",
    "print('Jumlah data isbn:', len(isbn))\n",
    "print('Jumlah data book_rating:', len(book_rating))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QNodPmHG-ZUk"
   },
   "source": [
    "## Membuat DataFrame Bersih untuk Modeling\n",
    "\n",
    "Pada tahap ini, kami membuat sebuah DataFrame baru bernama **ratings\\_clean** yang berisi tiga kolom utama hasil dari konversi list:\n",
    "\n",
    "* **user\\_id** : ID pengguna\n",
    "* **isbn** : ISBN buku\n",
    "* **book\\_rating** : Nilai rating yang diberikan pengguna pada buku\n",
    "\n",
    "Pembuatan DataFrame ini bertujuan untuk memudahkan proses pemodelan sistem rekomendasi pada tahap berikutnya, baik menggunakan metode Collaborative Filtering maupun Content-Based Filtering.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 423
    },
    "id": "FOJ_j1A--bNc",
    "outputId": "16a2aa64-4346-4524-bf51-e2d9d23a33bc"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>isbn</th>\n",
       "      <th>book_rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>276747</td>\n",
       "      <td>0060517794</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>276747</td>\n",
       "      <td>0671537458</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>276747</td>\n",
       "      <td>0679776818</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>276762</td>\n",
       "      <td>0380711524</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>276772</td>\n",
       "      <td>0553572369</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203846</th>\n",
       "      <td>276688</td>\n",
       "      <td>0892966548</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203847</th>\n",
       "      <td>276688</td>\n",
       "      <td>1551669315</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203848</th>\n",
       "      <td>276704</td>\n",
       "      <td>0345386108</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203849</th>\n",
       "      <td>276704</td>\n",
       "      <td>0743211383</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203850</th>\n",
       "      <td>276704</td>\n",
       "      <td>1563526298</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>203851 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        user_id        isbn  book_rating\n",
       "0        276747  0060517794            9\n",
       "1        276747  0671537458            9\n",
       "2        276747  0679776818            8\n",
       "3        276762  0380711524            5\n",
       "4        276772  0553572369            7\n",
       "...         ...         ...          ...\n",
       "203846   276688  0892966548           10\n",
       "203847   276688  1551669315            6\n",
       "203848   276704  0345386108            6\n",
       "203849   276704  0743211383            7\n",
       "203850   276704  1563526298            9\n",
       "\n",
       "[203851 rows x 3 columns]"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings_clean = pd.DataFrame({\n",
    "    'user_id': user_id,\n",
    "    'isbn': isbn,\n",
    "    'book_rating': book_rating\n",
    "})\n",
    "\n",
    "ratings_clean"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OJ6bhtpU-f6U"
   },
   "source": [
    "**Hasil**: DataFrame **ratings\\_clean** yang dibuat memiliki **203.851 baris** dan **3 kolom**.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nf3n_9FN-jM_"
   },
   "source": [
    "## Membuat Matriks User-Item untuk Collaborative Filtering\n",
    "\n",
    "Pada tahap ini, kami menyusun **User-Item Matrix** dari dataset **ratings\\_clean**.\n",
    "\n",
    "**Metode yang digunakan:**\n",
    "\n",
    "* Membuat pivot table dengan fungsi `pivot_table` dari pandas.\n",
    "\n",
    "**Struktur matriks:**\n",
    "\n",
    "* Baris merepresentasikan `user_id`.\n",
    "* Kolom merepresentasikan `isbn` buku.\n",
    "* Nilai pada sel adalah `book_rating` yang diberikan oleh pengguna terhadap buku tersebut.\n",
    "\n",
    "**Penjelasan proses:**\n",
    "\n",
    "* Setiap pengguna menjadi satu baris di tabel.\n",
    "* Setiap buku diwakili oleh satu kolom.\n",
    "* Jika pengguna memberi rating pada sebuah buku, nilai rating tersebut muncul pada sel terkait.\n",
    "* Jika tidak ada rating, maka sel berisi `NaN` (nilai kosong).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 304
    },
    "id": "HUrjF5km-gOA",
    "outputId": "0290987c-d341-45a8-9420-1e2239b37841"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User-Item Matrix Shape: (20908, 25790)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>isbn</th>\n",
       "      <th>0000000000</th>\n",
       "      <th>0000000000000</th>\n",
       "      <th>0002005018</th>\n",
       "      <th>0002116286</th>\n",
       "      <th>0002239183</th>\n",
       "      <th>0002240114</th>\n",
       "      <th>0002243962</th>\n",
       "      <th>0002244098</th>\n",
       "      <th>0002251760</th>\n",
       "      <th>0002255014</th>\n",
       "      <th>...</th>\n",
       "      <th>9783426650752</th>\n",
       "      <th>978972024570</th>\n",
       "      <th>9812327975</th>\n",
       "      <th>9871138148</th>\n",
       "      <th>987932504</th>\n",
       "      <th>9895550138</th>\n",
       "      <th>9994770934</th>\n",
       "      <th>B00009EF82</th>\n",
       "      <th>B0000AA9IZ</th>\n",
       "      <th>O67174142X</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>user_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>92</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 25790 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "isbn     0000000000  0000000000000  0002005018  0002116286  0002239183  \\\n",
       "user_id                                                                  \n",
       "8               NaN            NaN         5.0         NaN         NaN   \n",
       "17              NaN            NaN         NaN         NaN         NaN   \n",
       "53              NaN            NaN         NaN         NaN         NaN   \n",
       "69              NaN            NaN         NaN         NaN         NaN   \n",
       "92              NaN            NaN         NaN         NaN         NaN   \n",
       "\n",
       "isbn     0002240114  0002243962  0002244098  0002251760  0002255014  ...  \\\n",
       "user_id                                                              ...   \n",
       "8               NaN         NaN         NaN         NaN         NaN  ...   \n",
       "17              NaN         NaN         NaN         NaN         NaN  ...   \n",
       "53              NaN         NaN         NaN         NaN         NaN  ...   \n",
       "69              NaN         NaN         NaN         NaN         NaN  ...   \n",
       "92              NaN         NaN         NaN         NaN         NaN  ...   \n",
       "\n",
       "isbn     9783426650752  978972024570  9812327975  9871138148  987932504  \\\n",
       "user_id                                                                   \n",
       "8                  NaN           NaN         NaN         NaN        NaN   \n",
       "17                 NaN           NaN         NaN         NaN        NaN   \n",
       "53                 NaN           NaN         NaN         NaN        NaN   \n",
       "69                 NaN           NaN         NaN         NaN        NaN   \n",
       "92                 NaN           NaN         NaN         NaN        NaN   \n",
       "\n",
       "isbn     9895550138  9994770934  B00009EF82  B0000AA9IZ  O67174142X  \n",
       "user_id                                                              \n",
       "8               NaN         NaN         NaN         NaN         NaN  \n",
       "17              NaN         NaN         NaN         NaN         NaN  \n",
       "53              NaN         NaN         NaN         NaN         NaN  \n",
       "69              NaN         NaN         NaN         NaN         NaN  \n",
       "92              NaN         NaN         NaN         NaN         NaN  \n",
       "\n",
       "[5 rows x 25790 columns]"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "user_item_matrix = ratings_clean.pivot_table(index='user_id', columns='isbn', values='book_rating')\n",
    "\n",
    "print('User-Item Matrix Shape:', user_item_matrix.shape)\n",
    "\n",
    "user_item_matrix.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0jopObnq-vLn"
   },
   "source": [
    "**Hasil**:\n",
    "1. Matriks User-Item yang terbentuk memiliki ukuran **(20.908, 25.790)**.\n",
    "2. Terdapat banyak nilai **missing (NaN)** karena sebagian besar pengguna hanya memberikan rating pada sebagian kecil dari total buku yang tersedia.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7HlQEkqv-yMj"
   },
   "source": [
    "## Membuat Daftar User-ID dan ISBN Unik\n",
    "\n",
    "Pada tahap ini, kami menyusun daftar **user\\_id** dan **ISBN** unik yang berasal dari dataset **ratings\\_clean**. Daftar ini nantinya akan digunakan untuk proses encoding data.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "o-WF6Qfp-vc5",
    "outputId": "995979fc-18b6-4212-f328-28dabd7320cd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah user unik: 20908\n",
      "Jumlah buku unik: 25790\n"
     ]
    }
   ],
   "source": [
    "user_ids = ratings_clean['user_id'].unique().tolist()\n",
    "print('Jumlah user unik:', len(user_ids))\n",
    "\n",
    "# Membuat list isbn unik\n",
    "isbn_ids = ratings_clean['isbn'].unique().tolist()\n",
    "print('Jumlah buku unik:', len(isbn_ids))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5IOuzZHC-5gz"
   },
   "source": [
    "**Hasil** : Terdapat **20.908 User-ID unik** dan **25.790 ISBN unik** dalam dataset setelah pembersihan.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q1PkLxGF-9uw"
   },
   "source": [
    "## Encoding User-ID dan ISBN ke Bentuk Integer\n",
    "\n",
    "Pada tahap ini, kami mengubah data **user\\_id** dan **isbn** menjadi representasi angka (integer) agar bisa digunakan sebagai input pada embedding layer dalam model deep learning.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "id": "PVOJkBak-_kv"
   },
   "outputs": [],
   "source": [
    "user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}\n",
    "isbn_to_isbn_encoded = {x: i for i, x in enumerate(isbn_ids)}\n",
    "\n",
    "user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}\n",
    "isbn_encoded_to_isbn = {i: x for i, x in enumerate(isbn_ids)}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WHT37ZYN_DiK"
   },
   "source": [
    "## Memetakan (Mapping) User dan ISBN yang Telah Di-encode ke DataFrame\n",
    "\n",
    "Setelah proses encoding selesai, kami menambahkan dua kolom baru, yaitu **user** dan **book**, ke dalam dataset **ratings\\_clean**. Kolom-kolom ini berisi hasil konversi **User-ID** dan **ISBN** ke dalam bentuk bilangan bulat (integer).\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "PfLaFX1E_FXx",
    "outputId": "312d0386-bedd-4d3d-90e2-d033931192f9"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>isbn</th>\n",
       "      <th>book_rating</th>\n",
       "      <th>user</th>\n",
       "      <th>book</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>276747</td>\n",
       "      <td>0060517794</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>276747</td>\n",
       "      <td>0671537458</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>276747</td>\n",
       "      <td>0679776818</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>276762</td>\n",
       "      <td>0380711524</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>276772</td>\n",
       "      <td>0553572369</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   user_id        isbn  book_rating  user  book\n",
       "0   276747  0060517794            9     0     0\n",
       "1   276747  0671537458            9     0     1\n",
       "2   276747  0679776818            8     0     2\n",
       "3   276762  0380711524            5     1     3\n",
       "4   276772  0553572369            7     2     4"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings_clean['user'] = ratings_clean['user_id'].map(user_to_user_encoded)\n",
    "ratings_clean['book'] = ratings_clean['isbn'].map(isbn_to_isbn_encoded)\n",
    "ratings_clean.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "M6HPpx3-_Jyn"
   },
   "source": [
    "## Konversi Nilai Rating ke Tipe Data float32\n",
    "\n",
    "Untuk kebutuhan pemodelan menggunakan TensorFlow, nilai rating pada dataset diubah menjadi tipe data **float32** agar kompatibel dengan proses training model.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "id": "YEAMxxMr_KK3"
   },
   "outputs": [],
   "source": [
    "# Import library tambahan\n",
    "import numpy as np\n",
    "\n",
    "ratings_clean['book_rating'] = ratings_clean['book_rating'].values.astype(np.float32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sfdL7Ejo_Z7_"
   },
   "source": [
    "## Check Jumlah User dan Buku Setelah Encoding\n",
    "\n",
    "Setelah melakukan encoding, kami melakukan pengecekan ulang untuk memastikan jumlah total user dan buku yang tersedia dalam dataset.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QfUm5F2v_brM",
    "outputId": "4391f026-a0ac-4d00-ce65-cf6f8d670ebc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah user: 20908, Jumlah buku: 25790\n"
     ]
    }
   ],
   "source": [
    "num_users = len(user_to_user_encoded)\n",
    "num_books = len(isbn_encoded_to_isbn)\n",
    "\n",
    "print(f\"Jumlah user: {num_users}, Jumlah buku: {num_books}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zs5iM918_fye"
   },
   "source": [
    "**Hasil** :  Setelah proses encoding, ditemukan bahwa terdapat **20.908** user unik dan **25.790** buku unik dalam dataset.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QnBavD51_i9z"
   },
   "source": [
    "## Membagi (Split) Dataset untuk Pelatihan dan Validasi\n",
    "\n",
    "Dataset diacak dan dipisahkan menjadi dua bagian, yaitu data pelatihan (training) dan data validasi (validation), dengan perbandingan 80:20.\n",
    "\n",
    "* **Fitur (x)** terdiri dari pasangan `user` dan `book`.\n",
    "* **Target (y)** adalah nilai rating buku (`book_rating`) yang telah dinormalisasi ke rentang 0 hingga 1.\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vZDpe7p8_gCA",
    "outputId": "8296d005-1050-49b2-e535-9a18294d1019"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape x_train: (163080, 2), y_train: (163080,)\n",
      "Shape x_val: (40771, 2), y_val: (40771,)\n"
     ]
    }
   ],
   "source": [
    "# Shuffle dataset\n",
    "ratings_clean = ratings_clean.sample(frac=1, random_state=42)\n",
    "\n",
    "# Membuat variabel x (fitur) dan y (target)\n",
    "x = ratings_clean[['user', 'book']].values\n",
    "y = ratings_clean['book_rating'].apply(lambda x: (x - ratings_clean['book_rating'].min()) / (ratings_clean['book_rating'].max() - ratings_clean['book_rating'].min())).values\n",
    "\n",
    "# Membagi data 80% train, 20% validation\n",
    "train_indices = int(0.8 * ratings_clean.shape[0])\n",
    "x_train, x_val, y_train, y_val = (\n",
    "    x[:train_indices],\n",
    "    x[train_indices:],\n",
    "    y[:train_indices],\n",
    "    y[train_indices:]\n",
    ")\n",
    "\n",
    "print(f\"Shape x_train: {x_train.shape}, y_train: {y_train.shape}\")\n",
    "print(f\"Shape x_val: {x_val.shape}, y_val: {y_val.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "u8MCxxd__pzK"
   },
   "source": [
    "**Hasil pembagian data:**\n",
    "\n",
    "* Bentuk data pelatihan: x\\_train (163.080, 2), y\\_train (163.080,)\n",
    "* Bentuk data validasi: x\\_val (40.771, 2), y\\_val (40.771,)\n",
    "\n",
    "Tujuan pemisahan ini adalah untuk memastikan data pelatihan dan validasi tidak tumpang tindih, sehingga model dapat belajar dengan baik dan diuji secara akurat."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EfWJEqtz_y5e"
   },
   "source": [
    "# Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8GaefaIS_1ko"
   },
   "source": [
    "## Model Development dengan Content Based Filtering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "z8hYoifg_5HK"
   },
   "source": [
    "### Membuat TF-IDF Matrix dari Judul Buku\n",
    "\n",
    "**1. Filtering ISBN Buku:**\n",
    "Dataset `books` difilter agar hanya menyertakan ISBN yang terdapat dalam `ratings_clean`.\n",
    "* Tujuan: Menyelaraskan data buku dengan buku-buku yang telah dirating secara aktif.\n",
    "\n",
    "**2. Menangani Missing Value:**\n",
    "Kolom `Book-Title` yang kosong diisi dengan string kosong (`''`) untuk mencegah error saat proses vektorisasi teks.\n",
    "* Teknik: `fillna('')`\n",
    "\n",
    "**3. TF-IDF Vectorization:**\n",
    "Digunakan **`TfidfVectorizer`** dari `scikit-learn` dengan parameter `stop_words='english'`.\n",
    "* Tujuan: Mengubah teks judul buku menjadi representasi numerik berdasarkan pentingnya kata (frekuensi term invers dokumen).\n",
    "\n",
    "**4. Fit dan Transform:**\n",
    "TF-IDF Vectorizer di-*fit* dan diterapkan (`transform`) pada kolom `Book-Title`.\n",
    "* Hasil: TF-IDF Matrix yang merepresentasikan setiap judul buku dalam bentuk vektor.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "GfMrj2M8_qUE",
    "outputId": "6b9637b4-ad00-4d24-d7b8-dfd694f30fc4"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TF-IDF Matrix Shape: (24253, 16052)\n"
     ]
    }
   ],
   "source": [
    "# Import library\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Gabungkan books dan ratings_clean untuk memastikan ISBN ada di kedua dataset\n",
    "books_filtered = books[books['ISBN'].isin(ratings_clean['isbn'])]\n",
    "\n",
    "# Reset index untuk kemudahan\n",
    "books_filtered = books_filtered.reset_index(drop=True)\n",
    "\n",
    "# Membuat TF-IDF Vectorizer pada kolom Book-Title\n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "\n",
    "# Beberapa judul mungkin NaN, kita isi kosong dulu\n",
    "books_filtered['Book-Title'] = books_filtered['Book-Title'].fillna('')\n",
    "\n",
    "# Fit and transform\n",
    "tfidf_matrix = tfidf.fit_transform(books_filtered['Book-Title'])\n",
    "\n",
    "print('TF-IDF Matrix Shape:', tfidf_matrix.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nGHnZAGnAWgG"
   },
   "source": [
    "**Hasil:**\n",
    "\n",
    "* **Ukuran Matrix:** `(24.253, 16.052)`\n",
    "\n",
    "  * **24.253** baris = jumlah judul buku yang dianalisis.\n",
    "  * **16.052** kolom = jumlah kata unik (fitur) dari semua judul buku setelah stopwords dihapus.\n",
    "* **Makna Matrix:**\n",
    "\n",
    "  * Setiap baris merepresentasikan **sebuah buku** dalam bentuk vektor fitur.\n",
    "  * Setiap kolom merepresentasikan **kata unik** yang digunakan dalam judul-judul buku.\n",
    "  * Nilai pada matrix menunjukkan **seberapa penting** kata tersebut bagi suatu judul (berdasarkan frekuensi term dan seberapa umum kata tersebut dalam seluruh koleksi judul).\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q3N2-RahAaL4"
   },
   "source": [
    "### Cosine Similarity Antar Judul Buku\n",
    "\n",
    "**Teknik yang Digunakan**\n",
    "* **Cosine Similarity**:\n",
    "  * Mengukur **sudut** antar dua vektor TF-IDF.\n",
    "  * Semakin kecil sudutnya (semakin paralel vektornya), maka semakin **mirip** kedua judul buku.\n",
    "  * Nilai cosine similarity:\n",
    "\n",
    "    * **1.0** → sangat mirip\n",
    "    * **0.0** → tidak mirip sama sekali\n",
    "\n",
    "**Proses**\n",
    "* Menggunakan fungsi `cosine_similarity` dari `sklearn.metrics.pairwise`.\n",
    "* Input: TF-IDF matrix (24.253 x 16.052).\n",
    "* Output: Matrix similiarity (24.253 x 24.253), di mana:\n",
    "\n",
    "  * Baris ke-*i* dan kolom ke-*j* merepresentasikan tingkat kemiripan antara buku ke-*i* dan buku ke-*j* berdasarkan judul.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gxFfkcjNAW0b",
    "outputId": "023a81a7-85b4-46ef-80ec-ef10ceb25698"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cosine Similarity Matrix Shape: (24253, 24253)\n"
     ]
    }
   ],
   "source": [
    "# Menghitung Cosine Similarity dari TF-IDF Matrix\n",
    "cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)\n",
    "\n",
    "print('Cosine Similarity Matrix Shape:', cosine_sim.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0cDngXEbCcbX"
   },
   "source": [
    "**Hasil:**\n",
    "Matriks cosine similarity yang diperoleh memiliki dimensi (24.253, 24.253).\n",
    "Ini menunjukkan bahwa sebanyak 24.253 judul buku saling dibandingkan untuk menilai seberapa mirip masing-masing judul satu dengan yang lain.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "W1ZWC8shCg6R"
   },
   "source": [
    "**Menghitung Kemiripan Cosine Antar Pengguna**\n",
    "Pada tahap ini, kami melakukan perhitungan skor kemiripan antar seluruh pengguna berdasarkan pola rating mereka terhadap buku.\n",
    "\n",
    "**Metode yang digunakan:**\n",
    "\n",
    "* Cosine Similarity: Mengukur tingkat kesamaan antar vektor pengguna pada User-Item Matrix, dengan skor antara 0 (tidak mirip) hingga 1 (sangat mirip).\n",
    "\n",
    "* **Proses:**\n",
    "\n",
    "  * Menggunakan fungsi cosine\\_similarity dari scikit-learn.\n",
    "  * Nilai missing (NaN) pada User-Item Matrix diisi dengan angka 0 sebelum dilakukan perhitungan.\n",
    "  * Kemiripan cosine dihitung untuk setiap pasangan pengguna dalam dataset.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {
    "id": "J1j0O9UYCcsv"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User Similarity Matrix Shape: (20908, 20908)\n"
     ]
    }
   ],
   "source": [
    "# Menghitung Cosine Similarity antar pengguna\n",
    "user_similarity = cosine_similarity(user_item_matrix.fillna(0))  # Missing value diganti 0\n",
    "\n",
    "print('User Similarity Matrix Shape:', user_similarity.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1AHNIssUDI7M"
   },
   "source": [
    "**Hasil**: Matriks cosine similarity yang diperoleh memiliki dimensi (20.908, 20.908).\n",
    "Ini menunjukkan bahwa kemiripan dihitung antar 20.908 pengguna dengan membandingkan setiap pasangan pengguna satu sama lain.\n",
    "teks yang dimiringkan\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6lJB-aaqDMg2"
   },
   "source": [
    "**Membuat Fungsi Rekomendasi Berdasarkan Judul Buku**\n",
    "Pada tahap ini, kami mengembangkan fungsi rekomendasi yang bertujuan untuk menyajikan Top-N buku yang memiliki kemiripan tinggi berdasarkan judul buku.\n",
    "\n",
    "**Langkah-langkah yang diterapkan meliputi:**\n",
    "\n",
    "* Memanfaatkan matriks Cosine Similarity yang sudah tersedia.\n",
    "* Mencari indeks buku berdasarkan judul yang diinput.\n",
    "* Mengurutkan skor kemiripan dan memilih Top-N buku dengan nilai tertinggi.\n",
    "\n",
    "Fungsi ini bertujuan memberikan rekomendasi buku yang serupa sesuai dengan preferensi pengguna terhadap sebuah buku tertentu.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "id": "_rqqdT79DJNW"
   },
   "outputs": [],
   "source": [
    "def recommend_books(title, top_n=5):\n",
    "    # Cek apakah buku ada di dataset\n",
    "    if title not in books_filtered['Book-Title'].values:\n",
    "        return f\"Buku '{title}' tidak ditemukan di database.\"\n",
    "\n",
    "    # Cari index buku berdasarkan judul\n",
    "    idx = books_filtered[books_filtered['Book-Title'] == title].index[0]\n",
    "\n",
    "    # Ambil skor cosine similarity untuk buku tersebut\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "\n",
    "    # Urutkan skor dari tinggi ke rendah (kecuali dirinya sendiri [index 0])\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # Ambil top_n skor tertinggi (skip index 0 karena itu dirinya sendiri)\n",
    "    sim_scores = sim_scores[1:top_n+1]\n",
    "\n",
    "    # Ambil index buku-buku yang mirip\n",
    "    book_indices = [i[0] for i in sim_scores]\n",
    "\n",
    "    # Tampilkan judul buku rekomendasi\n",
    "    return books_filtered.iloc[book_indices][['Book-Title', 'Book-Author']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YDYxMT1qDV6E"
   },
   "source": [
    "**Contoh Output Rekomendasi Top-5 Buku (Menggunakan Content-Based Filtering)**\n",
    "Untuk menguji fungsi rekomendasi berbasis Content-Based Filtering yang sudah dibuat, kami menggunakan judul buku berikut sebagai input:\n",
    "\n",
    "**Buku Input:**\n",
    "‘Harry Potter and the Chamber of Secrets (Book 2)’\n",
    "\n",
    "Sistem kemudian mencari 5 buku dengan tingkat kemiripan tertinggi berdasarkan analisis TF-IDF dan Cosine Similarity antar judul buku.\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {
    "id": "t42g56wgDZqy"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Book-Title</th>\n",
       "      <th>Book-Author</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2813</th>\n",
       "      <td>Harry Potter and the Chamber of Secrets (Book 2)</td>\n",
       "      <td>J. K. Rowling</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5064</th>\n",
       "      <td>Harry Potter and the Chamber of Secrets (Book 2)</td>\n",
       "      <td>J. K. Rowling</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18653</th>\n",
       "      <td>Harry Potter and the Chamber of Secrets (Harry...</td>\n",
       "      <td>J. K. Rowling</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19563</th>\n",
       "      <td>Harry Potter and the Chamber of Secrets Postca...</td>\n",
       "      <td>J. K. Rowling</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11102</th>\n",
       "      <td>Harry Potter and the Chamber of Secrets (Book ...</td>\n",
       "      <td>J. K. Rowling</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Book-Title    Book-Author\n",
       "2813    Harry Potter and the Chamber of Secrets (Book 2)  J. K. Rowling\n",
       "5064    Harry Potter and the Chamber of Secrets (Book 2)  J. K. Rowling\n",
       "18653  Harry Potter and the Chamber of Secrets (Harry...  J. K. Rowling\n",
       "19563  Harry Potter and the Chamber of Secrets Postca...  J. K. Rowling\n",
       "11102  Harry Potter and the Chamber of Secrets (Book ...  J. K. Rowling"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Contoh menjalankan fungsi rekomendasi menggunakan Content-Based Filtering\n",
    "recommend_books('Harry Potter and the Chamber of Secrets (Book 2)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "g8oX8PkuDgTr"
   },
   "source": [
    "**Interpretasi Hasil:**\n",
    "\n",
    "Seluruh buku yang direkomendasikan merupakan variasi dari \"Harry Potter and the Chamber of Secrets,\" seperti edisi berbeda atau format yang beragam (buku cetak, audio CD, kartu pos).\n",
    "Hal ini menunjukkan bahwa metode Content-Based Filtering yang menggunakan kemiripan judul cukup efektif dalam menangkap kesamaan isi atau tema antar buku."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9JOalebtDlrw"
   },
   "source": [
    "## Model Development dengan Collaborative Filtering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "flpSntdQDqv9"
   },
   "source": [
    "Pada tahap ini, kami mengembangkan fungsi rekomendasi yang memberikan saran buku kepada pengguna berdasarkan kesamaan preferensi dengan pengguna lain.\n",
    "\n",
    "**Metode yang digunakan:**\n",
    "\n",
    "* **User-Based Collaborative Filtering:** Sistem mencari pengguna lain yang memiliki pola pemberian rating buku yang mirip dengan pengguna target menggunakan cosine similarity.\n",
    "* Buku-buku yang mendapat rating tinggi dari pengguna serupa, namun belum pernah dinilai oleh pengguna target, akan direkomendasikan.\n",
    "\n",
    "**Langkah-langkah fungsi:**\n",
    "\n",
    "1. Menghitung skor kemiripan antara pengguna target dengan seluruh pengguna lain.\n",
    "2. Memilih 5 pengguna paling mirip selain pengguna itu sendiri.\n",
    "3. Menghitung rata-rata rating untuk buku-buku yang diberikan oleh pengguna-pengguna mirip tersebut.\n",
    "4. Menyaring buku yang belum pernah dirating oleh pengguna target.\n",
    "5. Memilih Top-N buku berdasarkan rata-rata rating tertinggi.\n",
    "6. Menggabungkan data ISBN buku dengan judul (Book-Title) dan pengarang (Book-Author).\n",
    "7. Menambahkan informasi rata-rata rating (Average-Rating) ke dalam hasil rekomendasi.\n",
    "\n",
    "**Tujuan:**\n",
    "\n",
    "Menyediakan rekomendasi buku yang kemungkinan besar disukai oleh pengguna berdasarkan perilaku dan preferensi pengguna lain yang serupa.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {
    "id": "vj_L-3NZDgo2"
   },
   "outputs": [],
   "source": [
    "def recommend_books_userbased(user_id, user_item_matrix, user_similarity, books_filtered, top_n=5):\n",
    "    if user_id not in user_item_matrix.index:\n",
    "        return f\"User ID {user_id} tidak ditemukan dalam data.\"\n",
    "\n",
    "    # Ambil index user\n",
    "    idx = user_item_matrix.index.get_loc(user_id)\n",
    "\n",
    "    # Ambil skor similarity user target terhadap semua user lain\n",
    "    sim_scores = list(enumerate(user_similarity[idx]))\n",
    "\n",
    "    # Urutkan berdasarkan similarity terbesar (kecuali dirinya sendiri)\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # Ambil user-user mirip\n",
    "    similar_users_idx = [i[0] for i in sim_scores[1:6]]  # top 5 user mirip\n",
    "    similar_users = user_item_matrix.index[similar_users_idx]\n",
    "\n",
    "    # Ambil semua buku yang dirating tinggi oleh user-user mirip\n",
    "    books_recommend = user_item_matrix.loc[similar_users]\n",
    "    books_recommend = books_recommend.mean(axis=0)  # Rata-rata rating dari user-user mirip\n",
    "    books_recommend = books_recommend.sort_values(ascending=False)\n",
    "\n",
    "    # Ambil buku yang user belum pernah rating\n",
    "    user_books = user_item_matrix.loc[user_id]\n",
    "    unseen_books = books_recommend[user_books.isna()]\n",
    "\n",
    "    # Ambil Top-N ISBN dan ratingnya\n",
    "    top_books = unseen_books.head(top_n)\n",
    "\n",
    "    # Buat DataFrame hasil\n",
    "    recommendations = books_filtered[books_filtered['ISBN'].isin(top_books.index)][['ISBN', 'Book-Title', 'Book-Author']]\n",
    "\n",
    "    # Merge dengan average rating\n",
    "    recommendations = recommendations.merge(top_books.reset_index(), left_on='ISBN', right_on='isbn')\n",
    "    recommendations = recommendations.rename(columns={0: 'Average-Rating'})\n",
    "    recommendations = recommendations[['ISBN', 'Book-Title', 'Book-Author', 'Average-Rating']]\n",
    "\n",
    "    return recommendations.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4xucYftkD2Zj"
   },
   "source": [
    "**Contoh Hasil Rekomendasi User-Based Collaborative Filtering**\n",
    "\n",
    "Untuk menguji fungsi rekomendasi berbasis user, kami menggunakan salah satu User ID dari dataset sebagai input.\n",
    "\n",
    "**Input:**\n",
    "\n",
    "* User ID: 8\n",
    "\n",
    "**Proses:**\n",
    "\n",
    "* Sistem mencari 5 pengguna lain yang paling mirip dengan User ID 8 berdasarkan cosine similarity.\n",
    "* Sistem merekomendasikan buku-buku yang mendapat rating tinggi dari pengguna-pengguna mirip tersebut dan belum pernah diberikan rating oleh User ID 8.\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {
    "id": "pkuhdt1rD3sd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rekomendasi untuk User ID: 8\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ISBN</th>\n",
       "      <th>Book-Title</th>\n",
       "      <th>Book-Author</th>\n",
       "      <th>Average-Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0446310786</td>\n",
       "      <td>To Kill a Mockingbird</td>\n",
       "      <td>Harper Lee</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0684874350</td>\n",
       "      <td>ANGELA'S ASHES</td>\n",
       "      <td>Frank McCourt</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0440212561</td>\n",
       "      <td>Outlander</td>\n",
       "      <td>DIANA GABALDON</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         ISBN             Book-Title     Book-Author  Average-Rating\n",
       "0  0446310786  To Kill a Mockingbird      Harper Lee            10.0\n",
       "1  0684874350         ANGELA'S ASHES   Frank McCourt            10.0\n",
       "2  0440212561              Outlander  DIANA GABALDON            10.0"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Contoh Menjalankan Rekomendasi User-Based Collaborative Filtering\n",
    "# Pilih contoh user_id yang ada\n",
    "example_user = user_item_matrix.index[0]  # ambil user pertama misal\n",
    "\n",
    "print(f\"Rekomendasi untuk User ID: {example_user}\")\n",
    "recommend_books_userbased(example_user, user_item_matrix, user_similarity, books_filtered, top_n=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_VJOmM13EAkk"
   },
   "source": [
    "**Hasil:** Hanya 3 rekomendasi yang berhasil ditampilkan karena hanya 3 ISBN yang cocok dengan data buku yang tersedia (books_filtered)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WXsPhRgDEFK9"
   },
   "source": [
    "## Model Development dengan Collaborative Filtering menggunakan Keras"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-PFjPDkCEPuE"
   },
   "source": [
    "**Membangun Kelas Model RecommenderNet**\n",
    "\n",
    "Kami mengembangkan model deep learning kustom menggunakan TensorFlow/Keras dengan pendekatan subclassing API.\n",
    "\n",
    "**Detail Arsitektur:**\n",
    "\n",
    "* **User Embedding Layer:** Membuat representasi vektor laten yang mewakili setiap pengguna.\n",
    "* **Book Embedding Layer:** Membuat representasi vektor laten yang mewakili setiap buku.\n",
    "* **Bias Layer:** Menambahkan bias khusus untuk masing-masing pengguna dan buku.\n",
    "* **Dot Product:** Menghitung skor kecocokan antara vektor embedding pengguna dan buku.\n",
    "* **Fungsi Aktivasi:** Sigmoid, untuk memastikan nilai prediksi berada dalam rentang \\[0, 1].\n",
    "\n",
    "Model ini dirancang untuk secara efisien mempelajari pola interaksi antara pengguna dan item melalui embedding yang terlatih.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {
    "id": "ZlvDtZecECu1"
   },
   "outputs": [],
   "source": [
    "# Import library tambahan\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "\n",
    "# Build class RecommenderNet\n",
    "class RecommenderNet(tf.keras.Model):\n",
    "\n",
    "    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):\n",
    "        super(RecommenderNet, self).__init__(**kwargs)\n",
    "        self.num_users = num_users\n",
    "        self.num_books = num_books\n",
    "        self.embedding_size = embedding_size\n",
    "\n",
    "        # User embedding\n",
    "        self.user_embedding = layers.Embedding(\n",
    "            num_users,\n",
    "            embedding_size,\n",
    "            embeddings_initializer='he_normal',\n",
    "            embeddings_regularizer=keras.regularizers.l2(1e-6)\n",
    "        )\n",
    "        self.user_bias = layers.Embedding(num_users, 1)\n",
    "\n",
    "        # Book embedding\n",
    "        self.book_embedding = layers.Embedding(\n",
    "            num_books,\n",
    "            embedding_size,\n",
    "            embeddings_initializer='he_normal',\n",
    "            embeddings_regularizer=keras.regularizers.l2(1e-6)\n",
    "        )\n",
    "        self.book_bias = layers.Embedding(num_books, 1)\n",
    "\n",
    "    def call(self, inputs):\n",
    "        user_vector = self.user_embedding(inputs[:, 0])\n",
    "        user_bias = self.user_bias(inputs[:, 0])\n",
    "        book_vector = self.book_embedding(inputs[:, 1])\n",
    "        book_bias = self.book_bias(inputs[:, 1])\n",
    "\n",
    "        dot_user_book = tf.tensordot(user_vector, book_vector, 2)\n",
    "        x = dot_user_book + user_bias + book_bias\n",
    "\n",
    "        return tf.nn.sigmoid(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PRgSFaynEhP1"
   },
   "source": [
    "**Inisialisasi dan Kompilasi Model RecommenderNet**\n",
    "\n",
    "Setelah menyelesaikan desain arsitektur model, kami melakukan proses inisialisasi dan kompilasi dengan pengaturan sebagai berikut:\n",
    "\n",
    "* **Fungsi Loss:** Binary Crossentropy\n",
    "* **Optimizer:** Adam dengan learning rate 0.001\n",
    "* **Metrik Evaluasi:** Root Mean Squared Error (RMSE)\n",
    "\n",
    "Pemilihan fungsi loss binary crossentropy didasarkan pada karakter output model yang memprediksi probabilitas keterkaitan antara pengguna dan buku dalam rentang nilai \\[0,1].\n",
    "\n",
    "Setelah tahap kompilasi selesai, model siap untuk dilatih.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "id": "qt0OzhVXEkO8"
   },
   "outputs": [],
   "source": [
    "# Inisialisasi model\n",
    "model = RecommenderNet(num_users, num_books, embedding_size=50)\n",
    "\n",
    "# Compile model\n",
    "model.compile(\n",
    "    loss=tf.keras.losses.BinaryCrossentropy(),\n",
    "    optimizer=keras.optimizers.Adam(learning_rate=0.001),\n",
    "    metrics=[tf.keras.metrics.RootMeanSquaredError()]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kx5efNQTEou8"
   },
   "source": [
    "**Pelatihan Model RecommenderNet**\n",
    "\n",
    "Di tahap ini, model RecommenderNet dilatih menggunakan data yang telah dibagi menjadi set pelatihan dan validasi.\n",
    "\n",
    "Pengaturan pelatihan meliputi:\n",
    "\n",
    "* **Batch Size:** 8\n",
    "* **Jumlah Epoch:** 100\n",
    "* **Optimizer:** Adam\n",
    "* **Fungsi Loss:** Binary Crossentropy\n",
    "* **Metrik Evaluasi:** Root Mean Squared Error (RMSE)\n",
    "\n",
    "Konfigurasi ini bertujuan untuk mengoptimalkan performa model dalam mempelajari pola interaksi antara pengguna dan buku secara efektif.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {
    "id": "JsQnfScGEqDa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "20385/20385 [==============================] - 736s 36ms/step - loss: 0.6050 - root_mean_squared_error: 0.2395 - val_loss: 0.5683 - val_root_mean_squared_error: 0.2001\n",
      "Epoch 2/100\n",
      "20385/20385 [==============================] - 722s 35ms/step - loss: 0.5591 - root_mean_squared_error: 0.1902 - val_loss: 0.5611 - val_root_mean_squared_error: 0.1935\n",
      "Epoch 3/100\n",
      "20385/20385 [==============================] - 689s 34ms/step - loss: 0.5496 - root_mean_squared_error: 0.1805 - val_loss: 0.5568 - val_root_mean_squared_error: 0.1898\n",
      "Epoch 4/100\n",
      "20385/20385 [==============================] - 711s 35ms/step - loss: 0.5433 - root_mean_squared_error: 0.1740 - val_loss: 0.5542 - val_root_mean_squared_error: 0.1875\n",
      "Epoch 5/100\n",
      "20385/20385 [==============================] - 703s 34ms/step - loss: 0.5385 - root_mean_squared_error: 0.1691 - val_loss: 0.5520 - val_root_mean_squared_error: 0.1855\n",
      "Epoch 6/100\n",
      "20385/20385 [==============================] - 718s 35ms/step - loss: 0.5345 - root_mean_squared_error: 0.1650 - val_loss: 0.5510 - val_root_mean_squared_error: 0.1851\n",
      "Epoch 7/100\n",
      "20385/20385 [==============================] - 725s 36ms/step - loss: 0.5315 - root_mean_squared_error: 0.1619 - val_loss: 0.5499 - val_root_mean_squared_error: 0.1842\n",
      "Epoch 8/100\n",
      "20385/20385 [==============================] - 715s 35ms/step - loss: 0.5289 - root_mean_squared_error: 0.1594 - val_loss: 0.5491 - val_root_mean_squared_error: 0.1836\n",
      "Epoch 9/100\n",
      "20385/20385 [==============================] - 794s 39ms/step - loss: 0.5269 - root_mean_squared_error: 0.1573 - val_loss: 0.5487 - val_root_mean_squared_error: 0.1834\n",
      "Epoch 10/100\n",
      "20385/20385 [==============================] - 856s 42ms/step - loss: 0.5253 - root_mean_squared_error: 0.1556 - val_loss: 0.5489 - val_root_mean_squared_error: 0.1835\n",
      "Epoch 11/100\n",
      "20385/20385 [==============================] - 993s 49ms/step - loss: 0.5240 - root_mean_squared_error: 0.1544 - val_loss: 0.5489 - val_root_mean_squared_error: 0.1835\n",
      "Epoch 12/100\n",
      "20385/20385 [==============================] - 875s 43ms/step - loss: 0.5230 - root_mean_squared_error: 0.1533 - val_loss: 0.5490 - val_root_mean_squared_error: 0.1835\n",
      "Epoch 13/100\n",
      "20385/20385 [==============================] - 833s 41ms/step - loss: 0.5221 - root_mean_squared_error: 0.1524 - val_loss: 0.5495 - val_root_mean_squared_error: 0.1839\n",
      "Epoch 14/100\n",
      "20385/20385 [==============================] - 855s 42ms/step - loss: 0.5214 - root_mean_squared_error: 0.1516 - val_loss: 0.5498 - val_root_mean_squared_error: 0.1840\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "\n",
    "# Setup EarlyStopping\n",
    "early_stop = EarlyStopping(\n",
    "    monitor='val_root_mean_squared_error', # Monitor validasi RMSE\n",
    "    patience=5, # Tunggu 5 epoch berturut-turut, kalau nggak improve, stop\n",
    "    restore_best_weights=True # Balikin ke model dengan bobot terbaik\n",
    ")\n",
    "\n",
    "# Retraining dengan EarlyStopping\n",
    "history = model.fit(\n",
    "    x=x_train,\n",
    "    y=y_train,\n",
    "    batch_size=8,\n",
    "    epochs=100,\n",
    "    validation_data=(x_val, y_val),\n",
    "    callbacks=[early_stop]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XtZMgXM8E13l"
   },
   "source": [
    "**Hasil Pelatihan:**\n",
    "\n",
    "* Nilai RMSE terbaik pada data training mencapai sekitar 0.1516.\n",
    "* Nilai RMSE terbaik pada data validasi mencapai sekitar 0.1835.\n",
    "* Kurva pembelajaran menunjukkan tren yang stabil tanpa tanda-tanda overfitting yang signifikan.\n",
    "* Model mencapai performa optimal pada epoch ke-14 dan kemudian pelatihan dihentikan.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3bIoFkJAE5hP"
   },
   "source": [
    "**Simulasi Rekomendasi Model-Based Collaborative Filtering (Keras)**\n",
    "\n",
    "Pada tahap ini, kami menguji performa model Collaborative Filtering yang telah dilatih dengan memberikan rekomendasi buku untuk pengguna tertentu, yaitu **User-ID 278188**.\n",
    "\n",
    "**Proses:**\n",
    "\n",
    "1. Model menerima input berupa ID user dan seluruh daftar buku (ISBN).\n",
    "2. Model memprediksi skor kecocokan atau preferensi user terhadap setiap buku dalam skala \\[0,1].\n",
    "3. Berdasarkan skor prediksi, sistem memilih buku dengan skor tertinggi yang belum pernah dirating oleh user tersebut.\n",
    "4. Hasil rekomendasi berupa daftar buku yang diprediksi paling sesuai dengan minat dan preferensi user tersebut.\n",
    "\n",
    "**Tujuan:**\n",
    "\n",
    "Memberikan rekomendasi buku yang personal dan relevan untuk User-ID 263663 berdasarkan pola interaksi dan preferensi yang dipelajari oleh model dari data pengguna lain.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {
    "id": "7i4gBtLME2Ta"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Testing rekomendasi untuk User ID: 278188\n",
      "806/806 [==============================] - 2s 3ms/step\n",
      "Top-N Rekomendasi Buku untuk User:\n",
      "             ISBN                                         Book-Title  \\\n",
      "4206   0345339738  The Return of the King (The Lord of the Rings,...   \n",
      "13300  0060256656                                    The Giving Tree   \n",
      "30743  0743454529       My Sister's Keeper : A Novel (Picoult, Jodi)   \n",
      "64267  0836213319                       Dilbert: A Book of Postcards   \n",
      "79370  0439425220  Harry Potter and the Chamber of Secrets Postca...   \n",
      "\n",
      "            Book-Author  \n",
      "4206     J.R.R. TOLKIEN  \n",
      "13300  Shel Silverstein  \n",
      "30743      Jodi Picoult  \n",
      "64267       Scott Adams  \n",
      "79370     J. K. Rowling  \n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Ambil 1 contoh user random\n",
    "user_id = ratings_clean['user_id'].sample(1).iloc[0]\n",
    "print(f\"Testing rekomendasi untuk User ID: {user_id}\")\n",
    "\n",
    "# Cari buku yang pernah dirating user ini\n",
    "books_rated_by_user = ratings_clean[ratings_clean['user_id'] == user_id]\n",
    "\n",
    "# Cari buku yang belum dirating user ini\n",
    "all_isbns = set(ratings_clean['isbn'].unique())\n",
    "rated_isbns = set(books_rated_by_user['isbn'])\n",
    "unrated_isbns = list(all_isbns - rated_isbns)\n",
    "\n",
    "# Encode ISBN yang belum dirated\n",
    "unrated_books_encoded = [isbn_to_isbn_encoded.get(x) for x in unrated_isbns if isbn_to_isbn_encoded.get(x) is not None]\n",
    "user_encoder = user_to_user_encoded.get(user_id)\n",
    "\n",
    "# Bentuk array prediksi\n",
    "user_book_array = np.hstack(\n",
    "    (np.array([[user_encoder]] * len(unrated_books_encoded)), np.array(unrated_books_encoded).reshape(-1, 1))\n",
    ")\n",
    "\n",
    "# Prediksi rating untuk semua buku yang belum dirated\n",
    "predicted_ratings = model.predict(user_book_array).flatten()\n",
    "\n",
    "# Ambil Top-N rekomendasi\n",
    "top_n = 5\n",
    "top_ratings_indices = predicted_ratings.argsort()[-top_n:][::-1]\n",
    "recommended_isbns = [isbn_encoded_to_isbn.get(unrated_books_encoded[i]) for i in top_ratings_indices]\n",
    "\n",
    "# Tampilkan rekomendasi\n",
    "print('Top-N Rekomendasi Buku untuk User:')\n",
    "recommended_books = books[books['ISBN'].isin(recommended_isbns)][['ISBN', 'Book-Title', 'Book-Author']]\n",
    "print(recommended_books)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eJV6LBeJFdW_"
   },
   "source": [
    "**Interpretasi Hasil Testing Rekomendasi untuk User ID: 278188**\n",
    "\n",
    "Model berhasil memberikan rekomendasi buku yang populer dan relevan dengan preferensi pengguna tersebut. Buku-buku yang direkomendasikan, seperti *The Return of the King* dan *Harry Potter and the Chamber of Secrets*, menunjukkan kecenderungan minat pengguna terhadap genre fantasy dan literature klasik. Selain itu, rekomendasi buku seperti *The Giving Tree* dan *Dilbert: A Book of Postcards* juga mencerminkan variasi genre yang luas, menandakan bahwa model mampu menangkap preferensi pengguna yang beragam."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WXD7DcwkFlGU"
   },
   "source": [
    "## Analisis Model Sistem Rekomendasi\n",
    "\n",
    "Pada bagian ini, kami mengembangkan dua pendekatan sistem rekomendasi untuk memprediksi buku yang relevan bagi pengguna.\n",
    "\n",
    "1. **Content-Based Filtering**\n",
    "\n",
    "**Proses:**\n",
    "\n",
    "* Menyaring dataset buku agar hanya mencakup ISBN yang ada di dataset rating bersih.\n",
    "* Mengisi nilai kosong pada kolom judul buku dengan string kosong.\n",
    "* Menggunakan `TfidfVectorizer(stop_words='english')` untuk mengubah judul buku menjadi representasi numerik TF-IDF.\n",
    "* Membentuk matriks TF-IDF berukuran (24.253, 16.052).\n",
    "* Menghitung cosine similarity antar judul buku berdasarkan matriks TF-IDF.\n",
    "* Membuat fungsi `recommend_books(book_title)` yang menghasilkan rekomendasi Top-N buku dengan skor kemiripan tertinggi.\n",
    "\n",
    "**Output:**\n",
    "\n",
    "* Matriks TF-IDF berukuran (24.253, 16.052).\n",
    "* Matriks cosine similarity berukuran (24.253, 24.253), berhasil digunakan untuk mengukur kemiripan antar judul buku.\n",
    "\n",
    "**Contoh Rekomendasi Top-5 Buku:**\n",
    "Input: *'Harry Potter and the Chamber of Secrets (Book 2)'*\n",
    "Rekomendasi terdiri dari variasi buku yang sama dalam format atau edisi berbeda, menunjukkan sistem efektif menangkap kemiripan berdasarkan judul.\n",
    "\n",
    "---\n",
    "\n",
    "2. **Collaborative Filtering (Memory-Based, User-Based)**\n",
    "\n",
    "**Proses:**\n",
    "\n",
    "* Membentuk User-Item Matrix dari data rating.\n",
    "* Menghitung cosine similarity antar pengguna berdasarkan pola rating.\n",
    "* Membuat fungsi `recommend_books_userbased(user_id)` untuk menghasilkan rekomendasi Top-N buku berdasarkan kemiripan pengguna.\n",
    "\n",
    "**Output:**\n",
    "\n",
    "* User-Item Matrix berukuran (20.908, 25.790).\n",
    "* Matriks kemiripan pengguna berukuran (20.908, 20.908).\n",
    "\n",
    "**Contoh Rekomendasi untuk User ID 8:**\n",
    "Sistem mencari 5 pengguna paling mirip, kemudian merekomendasikan buku yang mereka nilai tinggi dan belum pernah dibaca user target. Dihasilkan 3 rekomendasi buku.\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IQhH_o_RFxrr"
   },
   "source": [
    "## Ringkasan Output Top-N Recommendation\n",
    "\n",
    "* Content-Based Filtering: rekomendasi berdasarkan kemiripan judul buku.\n",
    "* Collaborative Filtering: rekomendasi berdasarkan pola interaksi pengguna."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "53Gf8qP0F7Ix"
   },
   "source": [
    "\n",
    "## Perbandingan Tiga Pendekatan Rekomendasi\n",
    "\n",
    "| **Pendekatan**                                          | **Kelebihan**                                                                                                                               | **Kekurangan**                                                                                                                            |\n",
    "| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |\n",
    "| **Content-Based Filtering**                             | - Tidak memerlukan data interaksi (rating).<br>- Mampu merekomendasikan item baru yang belum pernah dinilai.                                | - Rekomendasi terbatas pada item yang mirip saja.<br>- Tidak mampu menangkap pola preferensi antar pengguna.                              |\n",
    "| **Collaborative Filtering (User-Based)**                | - Dapat mengenali pola kesamaan dalam komunitas pengguna.<br>- Rekomendasi lebih bervariasi karena mempertimbangkan selera banyak pengguna. | - Membutuhkan data interaksi user-item dalam jumlah cukup.<br>- Mengalami kesulitan jika menghadapi pengguna atau buku baru (cold-start). |\n",
    "| **Model-Based Collaborative Filtering (Deep Learning)** | - Mampu mempelajari pola preferensi yang kompleks dan tersembunyi.<br>- Tidak bergantung pada kemiripan eksplisit antar item.               | - Membutuhkan waktu pelatihan (training) yang lebih lama.<br>- Risiko overfitting jika data yang tersedia terbatas.                       |\n",
    "\n",
    "Dengan mengombinasikan ketiga pendekatan ini, sistem rekomendasi dapat menjadi lebih tangguh dan adaptif. Pendekatan gabungan memungkinkan sistem memberikan rekomendasi yang relevan, baik dari sisi kesamaan konten maupun dari pola interaksi dan preferensi pengguna.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Q-QUHCQXGF_i"
   },
   "source": [
    "# Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {
    "id": "hJi2nNMBFfZK"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot RMSE training dan validation\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')\n",
    "plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')\n",
    "plt.title('Learning Curve - RMSE')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Root Mean Squared Error')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "g2edQaLrGYou"
   },
   "source": [
    "## Curva - RMSE\n",
    "Grafik tersebut memperlihatkan tren perubahan nilai *Root Mean Squared Error* (RMSE) pada data pelatihan dan validasi sepanjang proses pelatihan model.\n",
    "\n",
    "RMSE pada data pelatihan secara bertahap menurun hingga mencapai nilai mendekati **0.1517**, sementara RMSE pada data validasi relatif stabil di kisaran **0.1835**.\n",
    "Tidak ditemukan tanda-tanda *overfitting* yang signifikan.\n",
    "Model menunjukkan konvergensi yang baik dalam rentang sekitar **10 hingga 15 epoch**.\n",
    "Temuan ini mengindikasikan bahwa model cukup efektif dalam memahami pola interaksi antara pengguna dan buku, serta mampu menghasilkan prediksi dengan tingkat kesalahan yang relatif rendah.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {
    "id": "6HXvjtyBGbq0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average Precision@5 for CBF: 0.0633\n"
     ]
    }
   ],
   "source": [
    "# Fungsi untuk menghitung Precision@K\n",
    "def precision_at_k(recommended_books, relevant_books, k=5):\n",
    "    \"\"\"\n",
    "    Menghitung precision pada Top-K rekomendasi.\n",
    "    recommended_books: list ISBN yang direkomendasikan\n",
    "    relevant_books: list ISBN yang relevan (diberi rating tinggi oleh user)\n",
    "    \"\"\"\n",
    "    if not relevant_books:\n",
    "        return 0.0\n",
    "    recommended_top_k = recommended_books[:k]\n",
    "    hits = len(set(recommended_top_k) & set(relevant_books))  # Buku relevan yang muncul di Top-K\n",
    "    return hits / k\n",
    "\n",
    "# Ambil user aktif yang sudah memberikan minimal 5 rating\n",
    "active_users = ratings_clean['user_id'].value_counts()\n",
    "sample_users = active_users[active_users >= 5].sample(100, random_state=42).index  # Sampling 100 user\n",
    "\n",
    "precision_scores = []\n",
    "\n",
    "# Iterasi untuk setiap user yang diambil\n",
    "for user_id in sample_users:\n",
    "    # Ambil semua rating tinggi (≥7) dari user\n",
    "    user_rated_books = ratings_clean[(ratings_clean['user_id'] == user_id) & (ratings_clean['book_rating'] >= 7)]\n",
    "    relevant_isbns = user_rated_books['isbn'].tolist()  # ISBN yang relevan untuk user\n",
    "\n",
    "    if not relevant_isbns:\n",
    "        continue  # Skip user jika tidak ada buku dengan rating tinggi\n",
    "\n",
    "    # Pilih salah satu buku sebagai input untuk rekomendasi\n",
    "    input_book_isbn = relevant_isbns[0]\n",
    "    input_title = books_filtered[books_filtered['ISBN'] == input_book_isbn]['Book-Title'].values\n",
    "    if len(input_title) == 0:\n",
    "        continue  # Skip jika judul tidak ditemukan\n",
    "\n",
    "    # Cari index dari judul di TF-IDF matrix\n",
    "    idx = books_filtered[books_filtered['Book-Title'] == input_title[0]].index\n",
    "    if len(idx) == 0:\n",
    "        continue  # Skip jika tidak ada index-nya\n",
    "\n",
    "    # Hitung skor kemiripan antara buku input dan seluruh buku\n",
    "    sim_scores = list(enumerate(cosine_sim[idx[0]]))  # cosine_sim dari TF-IDF\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Urutkan berdasarkan skor similarity\n",
    "    top_indices = [i[0] for i in sim_scores[1:6]]  # Ambil Top-5 (skip index 0 karena itu buku input sendiri)\n",
    "\n",
    "    # Ambil ISBN dari buku rekomendasi\n",
    "    recommended_isbns = books_filtered.iloc[top_indices]['ISBN'].tolist()\n",
    "\n",
    "    # Hitung precision untuk user ini\n",
    "    prec = precision_at_k(recommended_isbns, relevant_isbns)\n",
    "    precision_scores.append(prec)\n",
    "\n",
    "# Hitung rata-rata Precision_5 dari seluruh user\n",
    "avg_precision_at_5 = sum(precision_scores) / len(precision_scores)\n",
    "print(f\"Average Precision@5 for CBF: {avg_precision_at_5:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yW__3_HkGnKN"
   },
   "source": [
    "## Insight Evaluasi Content-Based Filtering (Precision_5)\n",
    "### Interpretasi Hasil:\n",
    "\n",
    "* Rata-rata *precision* sebesar **6,33%** mengindikasikan bahwa hanya sebagian kecil dari rekomendasi yang benar-benar relevan berdasarkan histori rating pengguna.\n",
    "* Sistem cenderung merekomendasikan buku dengan judul yang sangat mirip, namun tidak selalu sesuai dengan preferensi pengguna.\n",
    "* Keterbatasan pendekatan ini terletak pada ketergantungannya terhadap fitur tunggal, yaitu *Book Title*, sehingga kurang mampu menangkap preferensi pengguna secara menyeluruh.\n",
    "\n",
    "### Kemungkinan Penyebab:\n",
    "\n",
    "* Kemiripan judul tidak selalu mencerminkan kesamaan dalam isi, genre, atau kualitas buku.\n",
    "* Model hanya memanfaatkan informasi dari judul buku, tanpa mempertimbangkan *metadata* penting lainnya seperti sinopsis, genre, atau nama penulis.\n",
    "* Banyaknya buku dengan judul serupa (seperti versi audio, edisi khusus) dapat menyebabkan sistem memberikan rekomendasi yang bersifat duplikat atau tidak relevan bagi pengguna.\n",
    "\n",
    "### Poin Tambahan:\n",
    "\n",
    "* Meskipun nilai *precision* relatif rendah, pendekatan ini tetap berguna dalam mengatasi masalah *cold-start*, terutama:\n",
    "\n",
    "  * Bagi pengguna baru yang belum memiliki histori rating.\n",
    "  * Untuk buku baru yang belum banyak menerima ulasan atau rating dari pengguna lain.\n",
    "\n",
    "### Rekomendasi Pengembangan:\n",
    "\n",
    "* Menambahkan fitur tambahan seperti sinopsis, genre, atau atribut lain dalam model dapat meningkatkan kualitas rekomendasi dan relevansi hasil yang diberikan oleh sistem *Content-Based Filtering*.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0PMJXcgAG6R8"
   },
   "source": [
    "# Analisis Evaluasi Kinerja Sistem Rekomendasi\n",
    "\n",
    "Pada bagian ini, dilakukan evaluasi terhadap performa sistem rekomendasi yang telah dikembangkan, dengan membandingkan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering** (baik pendekatan *Memory-Based* maupun *Model-Based*).\n",
    "\n",
    "---\n",
    "\n",
    "**Metrik Evaluasi: Root Mean Squared Error (RMSE)**\n",
    "\n",
    "Untuk menilai kinerja model *Collaborative Filtering* yang dibangun menggunakan **TensorFlow Keras**, digunakan metrik **Root Mean Squared Error (RMSE)**. RMSE merupakan metrik populer dalam permasalahan regresi, termasuk prediksi nilai rating, karena dapat menunjukkan sejauh mana hasil prediksi menyimpang dari nilai aktual.\n",
    "\n",
    "**Alasan Penggunaan RMSE:**\n",
    "\n",
    "* Data yang digunakan merupakan rating berskala kontinu, bukan klasifikasi.\n",
    "* RMSE memberikan informasi yang jelas mengenai besarnya kesalahan prediksi dalam satuan rating.\n",
    "* Nilai RMSE yang lebih kecil mengindikasikan model yang lebih akurat.\n",
    "\n",
    "---\n",
    "\n",
    "**Hasil Evaluasi Model:**\n",
    "\n",
    "* **RMSE pada data pelatihan:** ± **0.1517**\n",
    "* **RMSE pada data validasi:** ± **0.1835**\n",
    "\n",
    "**Interpretasi:**\n",
    "\n",
    "* Nilai RMSE pada data training dan validation relatif kecil, menunjukkan bahwa model memiliki kemampuan prediksi yang cukup akurat.\n",
    "* Selisih antara RMSE training dan validation yang tidak terlalu besar menandakan tidak adanya gejala overfitting yang signifikan.\n",
    "* Model menunjukkan proses pembelajaran yang stabil dan mampu menggeneralisasi dengan baik terhadap data baru.\n",
    "\n",
    "---\n",
    "\n",
    "**Evaluasi Content-Based Filtering (Precision\\@5):**\n",
    "\n",
    "* **Rata-rata Precision\\@5:** **0.0633** atau **6,33%**\n",
    "\n",
    "**Interpretasi:**\n",
    "\n",
    "* Dari lima rekomendasi teratas, rata-rata hanya sekitar 6,3% yang benar-benar sesuai dengan preferensi pengguna berdasarkan histori rating.\n",
    "* Sistem cenderung merekomendasikan buku dengan kemiripan judul, namun belum tentu relevan secara personal.\n",
    "\n",
    "**Insight Tambahan:**\n",
    "\n",
    "* Rendahnya tingkat presisi disebabkan oleh keterbatasan fitur konten yang digunakan, yakni hanya mengandalkan judul buku.\n",
    "* Meskipun performanya kurang optimal dari sisi presisi, pendekatan **Content-Based Filtering** tetap memiliki nilai guna, terutama dalam mengatasi permasalahan *cold-start*, baik untuk pengguna baru maupun item (buku) yang belum banyak diberi rating.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2EgV9_XGHt5x"
   },
   "source": [
    "# Hubungan Evaluasi dengan Business Understanding\n",
    "\n",
    "Evaluasi terhadap model rekomendasi ini mengacu kembali pada tujuan bisnis dan rumusan masalah yang mendasari pembangunan sistem, guna memastikan solusi yang dikembangkan relevan dan aplikatif.\n",
    "\n",
    "---\n",
    "\n",
    "#### **Masalah 1:**\n",
    "\n",
    "**Pengguna mengalami kesulitan dalam menentukan pilihan buku di tengah banyaknya opsi yang tersedia.**\n",
    "\n",
    "**Solusi & Refleksi Evaluasi:**\n",
    "Sistem rekomendasi mampu mempermudah proses pencarian dengan menyajikan daftar buku teratas yang sesuai dengan preferensi masing-masing pengguna. Rekomendasi yang ditampilkan terbukti sejalan dengan minat dan genre favorit user, baik melalui pendekatan berbasis konten maupun perilaku pengguna lain yang serupa.\n",
    "\n",
    "---\n",
    "\n",
    "#### **Masalah 2:**\n",
    "\n",
    "**Minimnya data interaksi dari pengguna baru atau pasif menyulitkan dalam memahami preferensinya.**\n",
    "\n",
    "**Solusi & Refleksi Evaluasi:**\n",
    "Dengan mengandalkan pendekatan berbasis konten, sistem tetap dapat memberikan saran buku meskipun histori interaksi sangat terbatas. Hal ini menjadi solusi awal yang efektif dalam menghadapi tantangan *cold-start*, terutama saat pengguna belum aktif memberikan rating.\n",
    "\n",
    "---\n",
    "\n",
    "#### **Masalah 3:**\n",
    "\n",
    "**Platform membutuhkan pendekatan personalisasi untuk meningkatkan pengalaman pengguna dan mendorong keterlibatan lebih dalam.**\n",
    "\n",
    "**Solusi & Refleksi Evaluasi:**\n",
    "Melalui pendekatan Collaborative Filtering, baik berbasis pengguna maupun model, sistem dapat mengidentifikasi preferensi tersembunyi dan menyusun rekomendasi secara personal. Model deep learning juga berkontribusi besar dalam menghasilkan prediksi akurat berkat representasi laten (embedding) yang dibentuk dari interaksi user-buku, seperti tercermin dari nilai RMSE yang rendah.\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EBzcQrf2IK6A"
   },
   "source": [
    "### Tujuan Bisnis yang Tercapai:\n",
    "\n",
    "* Sistem membantu pengguna dalam menemukan buku yang relevan dengan minat dan kebiasaan bacanya.\n",
    "* Rekomendasi yang diberikan mampu mengurangi kebingungan pengguna saat dihadapkan dengan banyak pilihan.\n",
    "* Pengalaman pengguna meningkat karena sistem memberikan saran yang terasa lebih personal dan kontekstual.\n",
    "* Pengguna terdorong untuk menjelajahi genre-genre baru berkat kemampuan model dalam mengenali pola minat yang tidak langsung terlihat.\n",
    "\n",
    "---\n",
    "\n",
    "### Kesimpulan Evaluasi Sistem:\n",
    "\n",
    "Pendekatan **Content-Based Filtering** terbukti cukup efektif sebagai langkah awal, namun keterbatasannya terletak pada fokus yang hanya mengandalkan judul buku. Nilai Precision\\@5 yang hanya mencapai 6,33% menandakan bahwa sistem ini belum sepenuhnya mampu menangkap preferensi personal secara mendalam.\n",
    "\n",
    "Sementara itu, **User-Based Collaborative Filtering** menawarkan rekomendasi yang lebih sesuai dengan referensi pengguna lain yang serupa, meskipun performanya menurun ketika diterapkan pada pengguna baru atau yang belum aktif.\n",
    "\n",
    "Adapun pendekatan **Model-Based Collaborative Filtering** menunjukkan performa paling optimal. Nilai RMSE yang rendah dan kurva pelatihan yang stabil mengindikasikan bahwa model ini dapat secara efektif mengenali dan memetakan pola interaksi pengguna terhadap buku.\n",
    "\n",
    "Secara keseluruhan, kombinasi dari ketiga pendekatan ini telah berhasil memberikan solusi yang relevan terhadap permasalahan yang ada dan mendukung pencapaian target bisnis secara menyeluruh.\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
