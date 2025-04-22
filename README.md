# Zero-Shot Tweet Classification 📊🧠

Proyek ini berfokus pada klasifikasi tweet ke dalam beberapa kategori seperti Politik, Ekonomi, dan lain-lain, menggunakan pendekatan **Zero-Shot Classification** berbasis cosine similarity dengan model transformer dari Hugging Face.

## 📁 Struktur Proyek

```
├── dataset_labeled.csv
├── dataset_unlabeled.csv
├── dataset_unlabeled_predicted.csv
├── 2_ML_LP_7-6.ipynb
└── README.md
```

## 🔧 Langkah-langkah dan Penjelasan Kode

### 1. 🚿 Preprocessing Data

- Hitung jumlah data awal:
  ```python
  total_awal = len(data)
  print("Total tweet awal:", total_awal)
  ```

- Deteksi dan hitung duplikat:
  ```python
  jumlah_duplikat = data.duplicated(subset='text').sum()
  print("Jumlah duplikat:", jumlah_duplikat)
  ```

- Hapus duplikat dan NaN:
  ```python
  data = data.drop_duplicates(subset='text')
  data = data.dropna()
  ```

- Validasi kebersihan data:
  ```python
  print(data.isnull().sum())
  ```

### 2. 📊 Visualisasi Data

- Grafik distribusi label:
  ```python
  import matplotlib.pyplot as plt

  label_counts = data['label'].value_counts()
  ax = label_counts.plot(kind='bar')
  for i, count in enumerate(label_counts):
      ax.text(i, count + 5, str(count), ha='center')
  plt.title("Distribusi Label")
  plt.xlabel("Kategori")
  plt.ylabel("Jumlah Tweet")
  plt.show()
  ```
Bisa dilihat disini distribusi label tidak merata, yang artinya dari 4583 tweet yang ada lebih banyak yang ngomong tentang aspek politik (2952 tweet), dan secara kesulurahan tentang topik-topik yang sering muncul saat masa pemilu. 

-  Word Cloud:
  ```python
  from wordcloud import WordCloud

  text = " ".join(data['text'])
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.title("Word Cloud Data Mentah")
  plt.show()
  ```

### 3. ✍️ Text Cleaning

- Fungsi cleaning:
  ```python
  import re

  def preprocess_text(text):
      text = re.sub(r"http\S+|www.\S+", "", text)  # hapus URL
      text = re.sub(r"[^a-zA-Z\s]", "", text)      # hapus simbol
      text = text.lower()                          # huruf kecil
      text = re.sub(r"\s+", " ", text).strip()     # hapus spasi berlebih
      return text

  data['text'] = data['text'].apply(preprocess_text)
  ```

### 4. 🔎 Analisis Bigram

- Temukan bigram:
  ```python
  from collections import Counter
  from nltk.util import ngrams
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize

  def get_top_ngrams(texts, n=2, top_k=10):
      all_ngrams = []
      for text in texts:
          tokens = word_tokenize(text)
          n_grams = ngrams(tokens, n)
          all_ngrams.extend(n_grams)
      return Counter(all_ngrams).most_common(top_k)

  bigrams = get_top_ngrams(data['text'], n=2)
  for bigram, freq in bigrams:
      print(f"{' '.join(bigram)}: {freq}")
  ```
Bigram ini bantu memahami topik/tokoh yang dominan:

| Bigram              | Frekuensi |
|---------------------|-----------|
| ganjar pranowo      | 1017      |
| mahfud md           | 483       |
| pak anies           | 476       |
| anies baswedan      | 358       |
| pranowo dan         | 260       |
| ganjar mahfud       | 254       |
| dan mahfud          | 210       |
| pranowo mahfud      | 176       |
| prabowo subianto    | 175       |
| calon presiden      | 170       |

Bigram ini menunjukkan bahwa nama tokoh seperti *Ganjar Pranowo*, *Mahfud MD*, dan *Anies Baswedan* sering disebut, yang konsisten dengan konteks pemilu, dan juga *calon presiden*.

### 5. 🏷️ Label Mapping

- Ambil label dan mapping:
  ```python
  target_names = sorted(data['label'].unique())
  label_mapping = {label: idx for idx, label in enumerate(target_names)}
  ```

### 6. 🤖 Zero-Shot Classification

- Load model dan buat embeddings:
  ```python
  from sentence_transformers import SentenceTransformer
  from sklearn.metrics.pairwise import cosine_similarity
  import numpy as np

  model = SentenceTransformer("Rendika/tweets-election-classification")

  label_embeddings = model.encode(target_names)
  text_embeddings = model.encode(data['text'].tolist(), show_progress_bar=True)
  ```

- Hitung cosine similarity dan prediksi:
  ```python
  similarity_scores = cosine_similarity(text_embeddings, label_embeddings)
  predictions = np.argmax(similarity_scores, axis=1)

  y_true_indices = data['label'].map(label_mapping).values
  ```

### 7. 🧪 Evaluasi Model

- Laporan klasifikasi:
  ```python
  from sklearn.metrics import classification_report

  print(classification_report(y_true_indices, predictions, target_names=target_names))
  ```
Hasil dari evaluasi adalah sebagai berikut:
| Kategori                   | Precision | Recall | F1-score | Support |
|---------------------------|-----------|--------|----------|---------|
| Demografi                 | 0.50      | 0.02   | 0.03     | 60      |
| Ekonomi                   | 0.83      | 0.81   | 0.82     | 309     |
| Geografi                  | 0.11      | 0.05   | 0.07     | 20      |
| Ideologi                  | 0.76      | 0.76   | 0.76     | 339     |
| Pertahanan & Keamanan     | 0.76      | 0.86   | 0.81     | 330     |
| Politik                   | 0.94      | 0.87   | 0.90     | 2952    |
| Sosial Budaya             | 0.54      | 0.92   | 0.68     | 419     |
| Sumber Daya Alam          | 0.76      | 0.64   | 0.70     | 154     |
| **Accuracy**              |           |        | **0.84** | 4583    |
| **Macro Avg**             | 0.65      | 0.62   | 0.60     |         |
| **Weighted Avg**          | 0.86      | 0.84   | 0.84     |         |

Model menunjukkan akurasi umum sebesar **0.84** dan bekerja sangat baik pada label mayoritas seperti Politik, namun kurang optimal pada label minor seperti Geografi dan Demografi karena distribusi data yang tidak seimbang.

### 8. 🗃️ Klasifikasi Dataset Tanpa Label

- Load dan preprocessing:
  ```python
  unlabeled_data = pd.read_csv('dataset_unlabeled.csv', delimiter=';')
  unlabeled_data['Text'] = unlabeled_data['Text'].apply(preprocess_text)
  ```

- Prediksi label:
  ```python
  text_embeddings_unlabeled = model.encode(unlabeled_data['Text'].tolist(), show_progress_bar=True)
  similarity_unlabeled = cosine_similarity(text_embeddings_unlabeled, label_embeddings)
  predicted_indices = np.argmax(similarity_unlabeled, axis=1)
  unlabeled_data['predicted_label'] = [target_names[i] for i in predicted_indices]
  ```

- Simpan hasil:
  ```python
  unlabeled_data.to_csv("dataset_unlabeled_predicted.csv", index=False)
  ```

## 📝 Catatan
- Disini tidak dilakukan keseimbangan data, karena akan mengaruhi label dengan jumlah tweet yang besar.
- Distribusi data tidak seimbang, yang bisa mempengaruhi akurasi model.
- Tidak menggunakan preprocess stopwords, karena memengaruhi pre-trained model yang digunakan.

## 🧠 Referensi
- (https://github.com/kk7nc/Text_Classification)
- (https://huggingface.co/Rendika/tweets-election-classification)
