"""
Final CNN Model 2
=====================================================================
Changes from your original dissertation script:
1. FEATURES_CSV/TRACKS_CSV paths updated to the Kaggle download location
2. features.csv now loads with the correct 3-row header (same fix as RF/SVM)
3. Your original DID save a model checkpoint (.h5), but never saved the
   scaler or label encoder - meaning even if that .h5 had survived, you
   couldn't have actually used it, since new audio needs to go through
   the same scaling before prediction, and the genre names need decoding
   back from numbers. This version saves all three together.
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---------------------------------------------------------------
# UPDATED PATHS
# ---------------------------------------------------------------
BASE_PATH = "/root/.cache/kagglehub/datasets/imsparsh/fma-free-music-archive-small-medium/versions/1/fma_metadata"
FEATURES_CSV = os.path.join(BASE_PATH, "features.csv")
TRACKS_CSV = os.path.join(BASE_PATH, "tracks.csv")

# -----------------------------------
# 1. Load and Clean Features + Tracks (FIXED: 3-row header)
# -----------------------------------
print("\nLoading features.csv...")
features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2], low_memory=False)
features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features[pd.to_numeric(features.index, errors='coerce').notna()]
features.index = features.index.astype(str)
print(f"Features shape: {features.shape}")

print("\nLoading tracks.csv...")
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1], low_memory=False)
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
print(f"Tracks shape BEFORE cleaning: {tracks.shape}")

tracks = tracks[pd.to_numeric(tracks.index, errors='coerce').notna()]
tracks.index = tracks.index.astype(int)
tracks.dropna(axis=1, how='all', inplace=True)
print(f"Tracks shape AFTER cleaning: {tracks.shape}")

# -------------------------
# 2. Match Features and Tracks
# -------------------------
features.index = features.index.astype(int)
tracks.index = tracks.index.astype(int)
common_ids = features.index.intersection(tracks.index)
features = features.loc[common_ids]
tracks = tracks.loc[common_ids]
print(f"\nAfter aligning common IDs: features shape = {features.shape}, tracks shape = {tracks.shape}")

# -----------------------------------
# 3. Detect genre column safely
# -----------------------------------
genre_column = None
for col in tracks.columns:
    if 'genre_top' in col.lower():
        genre_column = col
        break
if genre_column is None:
    raise KeyError("No genre column ('genre_top') found after cleaning!")
print(f"Genre column detected: {genre_column}")

tracks = tracks[tracks[genre_column].notna()]
features = features.loc[tracks.index]
print(f"After removing missing genres: features shape = {features.shape}, tracks shape = {tracks.shape}")

# -----------------------------------
# 4. Filter Top 25 Genres
# -----------------------------------
top_genres = tracks[genre_column].value_counts().nlargest(25).index
tracks = tracks[tracks[genre_column].isin(top_genres)]
features = features.loc[tracks.index]
print(f"\nTracks after filtering top 25 genres: {tracks.shape}")
print(f"Features after filtering top 25 genres: {features.shape}")
print(f"Actual genres represented: {tracks[genre_column].nunique()}")

plt.figure(figsize=(15, 6))
tracks[genre_column].value_counts().plot(kind='bar')
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Number of Tracks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# 5. Prepare Data
# -----------------------------------
X = features.values
y = tracks[genre_column].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
print(f"\nData split: {X_train.shape[0]} training / {X_val.shape[0]} validation samples")

# -----------------------------------
# 6. Build CNN1D Model
# -----------------------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(256, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    GlobalAveragePooling1D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------------
# 7. Train the Model
# -----------------------------------
checkpoint_cb = ModelCheckpoint('fma_cnn_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
earlystop_cb = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=64,
                    callbacks=[checkpoint_cb, earlystop_cb])

# -----------------------------------
# 8. Plot Accuracy and Loss Curves
# -----------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# NEW: SAVE THE SCALER, LABEL ENCODER, AND FEATURE COLUMNS
# (the .h5 model already saves via checkpoint above, but was previously
# useless on its own without these companion files)
# ---------------------------------------------------------------
joblib.dump(scaler, "fma_cnn_scaler.pkl")
joblib.dump(le, "fma_cnn_label_encoder.pkl")

with open("fma_cnn_feature_columns.json", "w") as f:
    json.dump(features.columns.tolist(), f)

print("\n✅ FULL MASTER PIPELINE COMPLETED SUCCESSFULLY!")
print("\nSaved 4 files:")
print("  - fma_cnn_model.h5 (from checkpoint callback)")
print("  - fma_cnn_scaler.pkl")
print("  - fma_cnn_label_encoder.pkl")
print("  - fma_cnn_feature_columns.json")
print("\nDownload all 4 from Colab's file browser and send them back.")
