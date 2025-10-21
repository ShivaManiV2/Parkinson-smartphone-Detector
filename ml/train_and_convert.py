# ml/train_and_convert.py
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
from features import tapping_features, tremor_features, voice_features, featurize_dict
import os

# -- Helper to load dataset (replace with real dataset)
def generate_dummy_data(n=500):
    X = []
    y = []
    for i in range(n):
        fdict = {}
        
        # simulate label
        lab = 1 if np.random.rand() < 0.4 else 0
        
        # simulate tapping timestamps
        tcount = np.random.randint(10,40)
        ts = np.cumsum(np.random.normal(0.25, 0.05, size=tcount)) # seconds
        fdict.update(tapping_features(ts))
        
        # tremor simulate
        fs_tremor = 50  # Use fs=50 for tremor/accel data
        t_tremor = np.arange(0,3,1/fs_tremor)
        if lab==1:
            sig = 0.1*np.sin(2*np.pi*5*t_tremor) + 0.02*np.random.randn(len(t_tremor))
        else:
            sig = 0.01*np.random.randn(len(t_tremor))
        fdict.update(tremor_features(sig, fs_tremor))
        
        # voice - generate silence-ish or vowel-like
        fs_voice = 16000 # Use a proper sampling rate for voice
        t_voice = np.arange(0, 1.5, 1/fs_voice) # Use 1.5s voice sample
        if lab==1:
            wav = 0.05*np.sin(2*np.pi*200*t_voice) + 0.01*np.random.randn(len(t_voice))
        else:
            wav = 0.01*np.random.randn(len(t_voice))
        fdict.update(voice_features(wav, fs_voice))
        
        vec = featurize_dict(fdict)
        X.append(vec)
        y.append(lab)
        
    X = np.vstack(X)
    y = np.array(y)
    return X,y

# Train model
X,y = generate_dummy_data(800)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Simple dense model
input_dim = x_train.shape[1]
print(f"Input dimension: {input_dim}") # Should be 38

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=30, batch_size=32)

# Evaluate
print("Test eval:", model.evaluate(x_test,y_test))

# --- Save and Convert Model ---
os.makedirs('ml/models', exist_ok=True)
SAVED_MODEL_DIR = 'ml/models/saved_model_pd' 

# Use model.export() to create the SavedModel directory for TFLite
model.export(SAVED_MODEL_DIR) 

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# --- THIS IS YOUR MODEL FILE ---
OUTPUT_TFLITE_FILE = 'ml/models/pd_model.tflite'
open(OUTPUT_TFLITE_FILE, 'wb').write(tflite_model)

print(f"TFLite model written to {OUTPUT_TFLITE_FILE}")
print("COPY THIS FILE to your android-app/app/src/main/assets/ folder")