# LPDP Sentiment Analysis Pipeline

## Deskripsi

Analisis Sentimen Artikel Berita LPDP menggunakan NLP end-to-end

Pipeline ini mengklasifikasikan sentimen artikel berita LPDP (Lembaga Pengelola Dana Pendidikan) ke dalam 3 kelas: **Positif**, **Negatif**, dan **Netral**. Menggunakan dual-track approach dengan TF-IDF + model klasik serta IndoBERT fine-tuning.

---

## 📊 Dataset Overview

| Item                          | Detail                                                           |
| :---------------------------- | :--------------------------------------------------------------- |
| **Total Artikel Scraped**      | 1.937 artikel                                                    |
| **Artikel Valid**             | 1.370 artikel (setelah validasi URL manual)                     |
| **Artikel dengan Konten**     | 1.038 artikel ✅                                                |
| **Label Classes**             | Positive (37.1%) · Neutral (33.0%) · Negative (30.0%)          |
| **Labeling Method**           | Manual di Google Sheets                                          |
| **Bahasa**                    | Indonesia                                                        |

---

## 📁 Project Structure

```text
LPDP-SentimentAnalysis/
├── notebooks/                          # Jupyter Notebooks (Phase 1-10)
│   ├── 1. ScrappingArtikelLPDP.ipynb              # Phase 1: Scraping artikel
│   ├── 2. ScrappingKontenLPDP.ipynb               # Phase 2: Scraping konten artikel
│   ├── 3. TopicModellingLPDP.ipynb                # Phase 3: BERTopic 4 topik
│   ├── 4. PreprocessingLPDP.ipynb                 # Phase 4: Preprocessing (Track A/B)
│   ├── 5. FeatureExtractionLPDP.ipynb             # Phase 5: TF-IDF & BoW
│   ├── 6. NER_Visualisasi_AnalisisLPDP.ipynb      # Phase 6: Named Entity Recognition
│   ├── 7. POSTaggingLPDP.ipynb                    # Phase 7: Part-of-Speech Tagging
│   ├── 8. TextBlob&LexiconSentimentLPDP.ipynb     # Phase 8: Analisis sentimen leksikon
│   ├── 9. TrainTestSplitLPDP.ipynb                # Phase 9: Train/Test split stratified
│   └── 10. Modeling_Tier1_Tier2_Tier3_LPDP.ipynb  # Phase 10: Model training (Classical ML & IndoBERT)
│
├── data/                                # Dataset dan artefak data
│   ├── raw/                             # Data mentah dari scraping
│   │   ├── dataset_lpdp_konten_raw.csv           # 1.038 artikel dengan konten lengkap
│   │   └── dataset_lpdp_sorted.csv               # Artikel yang sudah disort dan divalidasi
│   │
│   ├── processed/                       # Data hasil preprocessing
│   │   ├── dataset_lpdp_preprocessed.csv         # Track A: Heavy preprocessing (stemming + stopword)
│   │   ├── dataset_lpdp_preprocessed_bert.csv    # Track B: Minimal preprocessing untuk IndoBERT
│   │   ├── dataset_lpdp_manual.csv               # Labeling manual hasil
│   │   └── dataset_lpdp_manual_stats.csv         # Statistik labeling
│   │
│   └── lexicon/                         # File leksikon sentimen
│       ├── positive.tsv                 # Leksikon kata positif dengan weight
│       ├── negative.tsv                 # Leksikon kata negatif dengan weight
│       └── slang_id.csv                 # Mapping slang Indonesia ke formal
│
├── outputs/                             # Hasil analisis dan modeling
│   ├── phase8_hasil_analisis_sentimen.csv       # Hasil analisis sentimen berbasis leksikon
│   ├── modeling_results_tier1_tier2_tier3.csv   # Performa model Classical ML
│   ├── indobert_tuning_results.csv              # Hasil fine-tuning IndoBERT
│   │
│   ├── output_split/                    # Artefak train/test split
│   │   ├── X_train_tfidf.pkl                   # TF-IDF matrix train
│   │   ├── X_test_tfidf.pkl                    # TF-IDF matrix test
│   │   ├── label_encoder.pkl                   # Label encoder
│   │   ├── stratified_kfold.pkl                # K-Fold splitter
│   │   ├── split_metadata.json                 # Metadata split
│   │   ├── track_a_train.csv                   # Track A training data
│   │   ├── track_a_test.csv                    # Track A test data
│   │   ├── track_b_train.csv                   # Track B training data
│   │   ├── track_b_test.csv                    # Track B test data
│   │   ├── label_distribution_full.png         # Label distribution plot (full dataset)
│   │   └── label_distribution_split.png        # Label distribution plot (train/test)
│   │
│   └── output_pos_tagging/              # Hasil POS tagging analysis
│       ├── pos_results.csv              # POS tagging results
│       ├── pos_summary.xlsx             # Summary POS analysis
│       ├── pos_distribusi_overall.png   # Overall POS distribution
│       ├── pos_heatmap_pos_sentimen.png # Heatmap: POS vs Sentimen
│       ├── pos_heatmap_pos_topik.png    # Heatmap: POS vs Topik
│       ├── pos_rasio_per_sentimen.png   # POS ratio per sentimen
│       ├── pos_rasio_per_topik.png      # POS ratio per topik
│       ├── pos_top_adj_per_sentimen.png # Top adjectives per sentiment
│       ├── pos_top_kata_noun_verb_adj.png      # Top words (noun/verb/adj)
│       └── pos_wordcloud_noun_verb_adj.png     # Wordcloud (noun/verb/adj)
│
├── docs/                                # Dokumentasi proyek
│   └── Pipeline_NLP_LPDP.md             # Detail lengkap pipeline dan daftar isi
│
├── README.md                            # File ini
├── LICENSE                              # MIT License
└── .markdownlint.json                   # Konfigurasi markdown linting
```

---

## 🔄 Pipeline Workflow

### Phase Overview

| Phase | Notebook File | Deskripsi |
| --- | --- | --- |
| 1 | `notebooks/1. ScrappingArtikelLPDP.ipynb` | Scraping artikel metadata via GNews |
| 2 | `notebooks/2. ScrappingKontenLPDP.ipynb` | Scraping konten + labeling manual |
| 3 | `notebooks/3. TopicModellingLPDP.ipynb` | BERTopic: discover 4 topik |
| 4 | `notebooks/4. PreprocessingLPDP.ipynb` | Preprocessing Track A/B |
| 5 | `notebooks/5. FeatureExtractionLPDP.ipynb` | TF-IDF feature extraction |
| 6 | `notebooks/6. NER_Visualisasi_AnalisisLPDP.ipynb` | Named Entity Recognition |
| 7 | `notebooks/7. POSTaggingLPDP.ipynb` | POS tagging (Stanza) |
| 8 | `notebooks/8. TextBlob&LexiconSentimentLPDP.ipynb` | Lexicon-based sentiment |
| 9 | `notebooks/9. TrainTestSplitLPDP.ipynb` | Train/test split (80:20) |
| 10 | `notebooks/10. Modeling_Tier1_Tier2_Tier3_LPDP.ipynb` | Model training |

**Output Locations:**

- **Phase 1**: `data/raw/dataset_lpdp_sorted.csv`
- **Phase 2**: `data/raw/dataset_lpdp_konten_raw.csv`
- **Phase 4**: `data/processed/dataset_lpdp_preprocessed*.csv`
- **Phase 5**: `outputs/output_split/X_train_test_tfidf.pkl`
- **Phase 7**: `outputs/output_pos_tagging/`
- **Phase 8**: `outputs/phase8_hasil_analisis_sentimen.csv`
- **Phase 9**: `outputs/output_split/track_*.csv`
- **Phase 10**: `outputs/modeling_results_tier1_tier2_tier3.csv`

### Dual-Track Approach

```text
DATA → PREPROCESSING
         ├─ TRACK A: Heavy Preprocessing
         │  ├─ Tokenization + Lowercase
         │  ├─ Stopword removal
         │  ├─ Stemming (Sastrawi)
         │  └─ TF-IDF → Classical ML
         │
         └─ TRACK B: Minimal Preprocessing
            ├─ Tokenization + Lowercase
            ├─ HTML normalization
            └─ IndoBERT tokenizer → Fine-tuning
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install jupyter pandas scikit-learn nltk sastrawi \
            bertopic transformers torch gensim \
            textblob stanza python-gnews
```

### 2. Run Pipeline Sequentially

Jalankan notebook dalam urutan Phase 1-10:

```bash
# Phase 1: Scraping artikel metadata
jupyter notebook notebooks/1.\ ScrappingArtikelLPDP.ipynb

# Phase 2: Scraping konten + labeling
jupyter notebook notebooks/2.\ ScrappingKontenLPDP.ipynb

# ... lanjutkan hingga Phase 10
jupyter notebook notebooks/10.\ Modeling_Tier1_Tier2_Tier3_LPDP.ipynb
```

### 3. Access Results

- **Hasil Modeling**: `outputs/modeling_results_tier1_tier2_tier3.csv`
- **Hasil IndoBERT**: `outputs/indobert_tuning_results.csv`
- **Visualisasi**: `outputs/output_pos_tagging/` dan `outputs/output_split/`

---

## 📊 Key Features

- **End-to-End NLP Pipeline**: Dari scraping → preprocessing → feature extraction →
  modeling → evaluation
- **Dual-Track Experiment Design**: Track A (TF-IDF + ML) & Track B (IndoBERT)
- **Indonesian NLP Techniques**: Sastrawi stemming, InSet lexicon, Stanza POS, IndoBERT
- **Comprehensive Linguistic Analysis**: NER, POS tagging, topic discovery, sentiment
- **Rich Visualization & Analysis**: Wordclouds, heatmaps, distribution charts

---

## 📚 Technology Stack

| Layer | Technology |
| --- | --- |
| **Scraping** | `python-gnews`, `requests`, `BeautifulSoup` |
| **Data Processing** | `pandas`, `numpy` |
| **NLP Preprocessing** | `NLTK`, `Sastrawi`, `regex` |
| **Sentiment Analysis** | `TextBlob`, `InSet` (Indonesian Sentiment Lexicon) |
| **POS/NER** | `Stanza`, `transformers` (IndoBERT) |
| **Topic Modeling** | `BERTopic` |
| **Feature Extraction** | `scikit-learn` (TF-IDF) |
| **Classical ML** | `scikit-learn` (Naive Bayes, Logistic Regression, LinearSVC) |
| **Deep Learning** | `transformers` (IndoBERT fine-tuning) |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Notebooks** | `jupyter`, `ipykernel` |

---

## 📖 Documentation

Dokumentasi lengkap pipeline dapat dilihat di [`docs/Pipeline_NLP_LPDP.md`](docs/Pipeline_NLP_LPDP.md).

Includes:

- Detailed phase descriptions
- Checklist dan pembagian tugas
- Tech stack specifications
- Referensi dan sumber data

---

## 📁 Data Dictionary

### Raw Data (data/raw/)

- **dataset_lpdp_konten_raw.csv**: 1.038 artikel dengan fields: `url`, `title`, `content`, `published_date`, `sentiment_label`, `topic`
- **dataset_lpdp_sorted.csv**: 1.370 artikel metadata: `url`, `title`, `source`, `published_date`, `sentiment_label`

### Processed Data (data/processed/)

- **dataset_lpdp_preprocessed.csv**: Track A dengan `text_clean` (heavy preprocessing)
- **dataset_lpdp_preprocessed_bert.csv**: Track B dengan `text_bert` (minimal preprocessing)
- **dataset_lpdp_manual_stats.csv**: Statistik labeling distribution

### Lexicon Data (data/lexicon/)

- **positive.tsv**: Positive words dengan weight score
- **negative.tsv**: Negative words dengan weight score
- **slang_id.csv**: Mapping slang → formal Indonesian

### Output Data (outputs/)

- **phase8_hasil_analisis_sentimen.csv**: Hasil lexicon-based sentiment (confidence scores per class)
- **modeling_results_tier1_tier2_tier3.csv**: Classical ML performa (accuracy, precision, recall, F1)
- **indobert_tuning_results.csv**: IndoBERT fine-tuning results per epoch

---

## 👥 Team & Credits

**Mata Kuliah**: Pengolahan Bahasa Alami (PBA)

**Kelompok**: 5

- Celine Auriel (5026221004)
- Nida Aulia Amartika (5026221095)
- Muhammad Iqbal Baiduri Yamani (5026221103)
- Salwa Iqlima A. (5026221098)
- Ratna Amalia Azzahra (5026221209)

**Departemen**: Sistem Informasi

**Institusi**: Institut Teknologi Sepuluh Nopember (ITS) Surabaya

---

## 📄 License

MIT License - Lihat file [LICENSE](LICENSE) untuk detail lengkap.

---

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Buat Pull Request

---

## 📞 Contact & Support

Untuk pertanyaan atau issue, silakan buka GitHub Issue di repository ini.

---

**Last Updated**: May 2026  
**Status**: ✅ Pipeline Complete (Phase 1-10 selesai)
