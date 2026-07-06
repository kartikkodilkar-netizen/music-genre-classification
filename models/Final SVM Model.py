"""
Final SVM Model
=====================================================================
Changes from your original dissertation script:
1. FEATURES_CSV/TRACKS_CSV paths updated to the Kaggle download location
2. features.csv now loads with the correct 3-row header (same fix as RF)
3. Saves the trained model, scaler, PCA transformer, and label encoder -
   the ORIGINAL script never saved the PCA step, which would have silently
   broken predictions later even if you had saved the model alone, since
   new data needs to go through the same PCA transform before prediction.
4. Kept only the final (C=100) training run, since your original script
   trained twice (C=10, then C=100) and only the second result matters.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# UPDATED PATHS
# ---------------------------------------------------------------
BASE_PATH = "/root/.cache/kagglehub/datasets/imsparsh/fma-free-music-archive-small-medium/versions/1/fma_metadata"
FEATURES_CSV = os.path.join(BASE_PATH, "features.csv")
TRACKS_CSV = os.path.join(BASE_PATH, "tracks.csv")

# ---------------------------------------------------------------
# 1. Load cleaned metadata (FIXED: 3-row header for features.csv)
# ---------------------------------------------------------------
print("\nLoading features.csv...")
features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2], low_memory=False)
features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features[pd.to_numeric(features.index, errors='coerce').notna()]
features.index = features.index.astype(str)
print("Features shape:", features.shape)

print("\nLoading tracks.csv...")
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1], low_memory=False)
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
tracks.index = tracks.index.astype(str).map(str.strip)
print("Tracks shape:", tracks.shape)

tracks = tracks[pd.to_numeric(tracks.index, errors='coerce').notna()]
tracks.index = tracks.index.astype(int)

# ---------------------------------------------------------------
# 2. Detect genre column
# ---------------------------------------------------------------
genre_column = None
for col in tracks.columns:
    if 'genre_top' in col.lower():
        genre_column = col
        print(f"Detected genre column: {genre_column}")
        break
if genre_column is None:
    raise KeyError("Genre column not found in tracks.csv")

# ---------------------------------------------------------------
# 3. Match and filter top genres
# ---------------------------------------------------------------
features.index = features.index.astype(int)
tracks.index = tracks.index.astype(int)
common_ids = features.index.intersection(tracks.index)
features = features.loc[common_ids]
tracks = tracks.loc[common_ids]

tracks = tracks[tracks[genre_column].notna()]
top_genres = tracks[genre_column].value_counts().nlargest(25).index
mask = tracks[genre_column].isin(top_genres)
tracks = tracks[mask]
features = features.loc[tracks.index]

print("\nFinal shape after merge:")
print("Features:", features.shape)
print("Tracks:", tracks.shape)
print(f"Genres represented: {tracks[genre_column].nunique()}")

# ---------------------------------------------------------------
# 4. Encode labels
# ---------------------------------------------------------------
labels = tracks[genre_column]
le = LabelEncoder()
y = le.fit_transform(labels)

# ---------------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA to reduce dimensionality
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------------------
# 6. Train/Validation Split
# ---------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------------
# 7. Train SVM Model (final config: C=100)
# ---------------------------------------------------------------
print("\nTraining SVM model (C=100, rbf kernel)...")
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)  # probability=True needed for confidence scores later
svm.fit(X_train, y_train)

# ---------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------
y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ---------------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------------
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, fmt='g')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# NEW: SAVE EVERYTHING NEEDED FOR DEPLOYMENT
# ---------------------------------------------------------------
joblib.dump(svm, "fma_svm_model.pkl", compress=3)
joblib.dump(scaler, "fma_svm_scaler.pkl")
joblib.dump(pca, "fma_svm_pca.pkl")  # critical - original script never saved this
joblib.dump(le, "fma_svm_label_encoder.pkl")

import json
with open("fma_svm_feature_columns.json", "w") as f:
    json.dump(features.columns.tolist(), f)

print("\nSaved 5 files:")
print("  - fma_svm_model.pkl")
print("  - fma_svm_scaler.pkl")
print("  - fma_svm_pca.pkl")
print("  - fma_svm_label_encoder.pkl")
print("  - fma_svm_feature_columns.json")
print("\nDownload all 5 from Colab's file browser and send them back.")
