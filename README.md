# 🎵 From Rhythm to Recognition
## Leveraging Deep Learning and Machine Learning for Music Genre Classification

> **MSc Business Analytics Dissertation** — Dublin Business School, May 2025  
> **Grade:** First Class Honours (72%) | **Supervisor:** Dr. David Willams

---

## 📌 Project Overview

Digital streaming platforms manage tens of millions of music tracks, where inaccurate genre classification directly degrades recommendation quality, content discovery, and user engagement. This research investigates whether **deep learning and traditional machine learning models** can reliably automate music genre classification at scale — with meaningful implications for platforms like Spotify, Apple Music, and YouTube Music.

The study conducts a rigorous comparative evaluation of **30 CNN architectures, 5 SVM configurations, and 5 Random Forest models** across two large-scale, real-world datasets containing over 100,000 audio tracks.

---

## 🎯 Research Questions

1. Can subgenre identification improve music recommender systems?
2. Which deep learning model architecture performs best for music genre classification?
3. How do feature representation choices and dataset diversity impact model performance?

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | Macro F-1 |
|---|---|---|---|---|
| **Final CNN Model 2** | 59.14% | 0.61 | 0.57 | 0.58 |
| **CNN Full Training** | 60.01% | 0.63 | 0.59 | 0.59 |
| **SVM (RBF Kernel)** ⭐ | **66.22%** | **0.66** | **0.66** | **0.66** |
| **Random Forest** | 62.86% | 0.60 | 0.58 | 0.58 |

> **Key insight:** While SVM achieved the highest overall accuracy, CNN models demonstrated superior class-level balance and generalisation across imbalanced genre distributions — making them more suitable for real-world streaming applications with hundreds of genres.

## Graphs
<img width="979" height="699" alt="Screenshot 2026-06-23 at 10 11 27" src="https://github.com/user-attachments/assets/435b4432-1e9a-44c9-8d93-5e166424fa02" />

<img width="753" height="523" alt="image" src="https://github.com/user-attachments/assets/3a48f578-6d4b-4be6-9c41-c79f473ff201" />

<img width="973" height="407" alt="Screenshot 2026-06-23 at 10 10 44" src="https://github.com/user-attachments/assets/89679ee0-04ab-4706-b46b-1a098c72f17b" />

---

## 🗂️ Datasets

| Dataset | Tracks | Genres | Format |
|---|---|---|---|
| **FMA Large** | 106,574 | 161 (top 25 used) | WAV, 30s clips |
| **MuMu** | ~30,000 | 30 top-level | MP3 |

- **FMA Large** — open-source audio dataset from the Free Music Archive, selected for size, genre diversity, and structured metadata (`tracks.csv`, `features.csv`)
- **MuMu** — combines Million Song Dataset metadata with Amazon product reviews, adding commercial and cross-cultural context
- After cleaning and feature alignment: **~108,000 valid tracks** retained from ~136,000 raw

---

## ⚙️ Methodology

### Feature Extraction Pipeline

All audio processed using **Librosa** at 22,050 Hz sampling rate, truncated/padded to 30 seconds.

| Feature | Parameters | Purpose |
|---|---|---|
| **MFCCs** | 13 coefficients, n_fft=2048, hop=512 | Timbral characteristics, human auditory perception |
| **Mel-Spectrogram** | 128 Mel bands, log-scaled amplitude | 2D time-frequency input for CNN |
| **Chroma STFT** | 12 pitch classes (C–B) | Harmonic and tonal structure |
| **Spectral Contrast** | 7 frequency bands | Musical texture and dynamics |
| **Zero-Crossing Rate** | — | Percussive sound detection |

All features normalised using **z-score standardisation** (mean=0, variance=1).

### Data Augmentation
To address class imbalance and improve generalisation:
- Pitch shifting
- Time stretching
- Gaussian noise injection
- Class weight balancing (`class_weight='balanced'`)

### Model Architecture (Final CNN)

```
Input (MFCCs / Mel-Spectrogram)
    ↓
Conv1D (64 filters) → BatchNorm → MaxPooling1D
    ↓
Conv1D (128 filters) → BatchNorm → MaxPooling1D
    ↓
Conv1D (256 filters) → BatchNorm → MaxPooling1D
    ↓
Global Average Pooling
    ↓
Dense (512, ReLU) → Dropout (0.5)
    ↓
Softmax Output (25 genre classes)
```

**Optimiser:** Adam (lr=0.0001) | **Loss:** Sparse Categorical Crossentropy | **Epochs:** up to 50 with early stopping

---

## 🔑 Key Findings

**MFCCs + Mel-spectrograms** consistently outperformed raw metadata features as CNN inputs
**Class balancing** (weighted loss + augmentation) significantly improved recall for minority genres — *Jazz* and *Folk* recall improved most noticeably
**SVM with RBF kernel** excelled on structured, low-dimensional feature spaces but struggled with hybrid genres (e.g. Electronic/Ambient overlap)
**Random Forest** provided strong interpretability and feature importance rankings but failed to capture temporal/spectral dependencies in audio signals
Most persistent misclassification: **Electronic vs Ambient** (spectral similarity) and **Rock vs Metal** (shared instrumentation)
Combining **FMA + MuMu datasets** improved macro-F1 scores versus single-dataset training

---

## 💼 Business Applications


| **Streaming Recommendation Engines** | Improved genre tagging → better playlist generation and content discovery |
| **Music Library & Metadata Management** | Automated classification for archival and cataloguing at scale 
| **Customer Segmentation** | Genre preference modelling for behavioural analytics and targeted marketing |
| **Audio Search Engines** | Genre-aware retrieval for licensing, sync, and commercial use |
| **Product & Content Analytics** | Genre trend analysis to inform A&R and platform content strategy |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.x-yellow)
![Librosa](https://img.shields.io/badge/Librosa-Audio-green)

## Tools

| **Deep Learning** | TensorFlow, Keras |
| **Machine Learning** | Scikit-Learn (SVM, Random Forest, GridSearchCV) |
| **Audio Processing** | Librosa, NumPy |
| **Data Handling** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Environment** | Google Colab, Python 3.10 |


## 📁 Repository Structure

```
music-genre-classification/
│
├── README.md                          ← You are here
├── dissertation.pdf                   ← Full MSc research paper
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← Dataset overview and genre distribution
│   ├── 02_feature_extraction.ipynb    ← MFCC, Mel-spectrogram, Chroma pipeline
│   ├── 03_cnn_model.ipynb             ← CNN architecture, training, evaluation
│   ├── 04_svm_model.ipynb             ← SVM kernel comparison
│   └── 05_random_forest.ipynb         ← RF model and feature importance
│
├── results/
│   ├── cnn_accuracy_loss_curves.png
│   ├── svm_confusion_matrix.png
│   └── rf_confusion_matrix.png
│
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/kartikkodilkar/music-genre-classification.git
cd music-genre-classification

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01 → 05)
jupyter notebook
```

**Dataset:** Notebooks use the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (freely available on Kaggle) as a reproducible substitute for the original FMA-large dataset used in the dissertation.

---

## 📄 Citation & Research Context

This project was submitted as an Applied Research dissertation for the **MSc in Business Analytics** at Dublin Business School (May 2025). The research methodology, experimental design, and findings are documented in full in `dissertation.pdf`.

Key references:
- Tzanetakis & Cook (2002) — foundational genre classification framework
- Defferrard et al. (2017) — FMA dataset
- Choi et al. (2017) — CNN for music audio tagging
- Panigrahi et al. (2023) — CNN applied to GTZAN

---

## 👤 Author

**Kartik Kodilkar**  
MSc Business Analytics — Dublin Business School  
Senior ITSM & Data Analytics Professional | Dublin, Ireland  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/kartikkodilkar)
[![Email](https://img.shields.io/badge/Email-kartik.kodilkar%40gmail.com-red)](mailto:kartik.kodilkar@gmail.com)

---

*This research contributes to the field of Music Information Retrieval (MIR) by providing a rigorous comparative analysis of deep learning and traditional ML approaches for genre classification at scale.*
