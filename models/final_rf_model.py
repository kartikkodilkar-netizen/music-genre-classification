"""
Final RF Model
==============
Changes from your original dissertation script:
1. FEATURES_CSV/TRACKS_CSV paths updated to match the Kaggle download
   location (your original used Google Drive paths, which no longer apply)
2. Added code at the end to actually SAVE the trained model, scaler,
   and label encoder as files - this was missing before, which is why
   nothing survived after your Colab session ended last year.
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# UPDATED PATHS - pointing to the Kaggle download instead of Google Drive
# ---------------------------------------------------------------
BASE_PATH = "/root/.cache/kagglehub/datasets/imsparsh/fma-free-music-archive-small-medium/versions/1/fma_metadata"
FEATURES_CSV = os.path.join(BASE_PATH, "features.csv")
TRACKS_CSV = os.path.join(BASE_PATH, "tracks.csv")

# Load features
# NOTE: FMA's features.csv uses a 3-row header (feature name / statistic / number)
# stacked on top of each other - a normal single-header read misreads these as data rows.
print("Loading features.csv...")
features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2], low_memory=False)
features.columns = ['_'.join(col).strip() for col in features.columns.values]
# Drop any leftover non-numeric index rows (safety net, shouldn't be needed with correct header now)
features = features[pd.to_numeric(features.index, errors='coerce').notna()]
features.index = features.index.astype(str)

# Load tracks
print("Loading tracks.csv...")
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1], low_memory=False)
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
tracks = tracks[tracks.index.to_series().apply(lambda x: str(x).isdigit())]
tracks.index = tracks.index.astype(int)

# Detect genre column
genre_column = None
for col in tracks.columns:
    if 'genre_top' in col.lower() or 'genre' in col.lower():
        genre_column = col
        break
print(f"Detected genre column: {genre_column}")

# Drop rows with missing genre
tracks = tracks[tracks[genre_column].notna()]

# Convert indices to match
features.index = features.index.astype(int)
tracks.index = tracks.index.astype(int)

# Match only common tracks
common_ids = features.index.intersection(tracks.index)
features = features.loc[common_ids]
tracks = tracks.loc[common_ids]

# Filter to top 25 genres
top_genres = tracks[genre_column].value_counts().nlargest(25).index
tracks = tracks[tracks[genre_column].isin(top_genres)]
features = features.loc[tracks.index]

print(f"\nFinal dataset: {features.shape[0]} tracks, {len(top_genres)} genres")

# Prepare data
X = features.values
y = tracks[genre_column].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train Random Forest
print("\nTraining Random Forest (this may take a few minutes)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")
report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
conf_matrix = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, xticklabels=le.classes_, yticklabels=le.classes_, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# NEW: SAVE THE MODEL - this was missing from the original script
# ---------------------------------------------------------------
joblib.dump(rf, "fma_rf_model.pkl")
joblib.dump(scaler, "fma_rf_scaler.pkl")
joblib.dump(le, "fma_rf_label_encoder.pkl")

import json
with open("fma_rf_feature_columns.json", "w") as f:
    json.dump(features.columns.tolist(), f)

print("\nSaved 4 files:")
print("  - fma_rf_model.pkl")
print("  - fma_rf_scaler.pkl")
print("  - fma_rf_label_encoder.pkl")
print("  - fma_rf_feature_columns.json")
print("\nDownload all 4 from Colab's file browser and send them back.")
