from xgboost import XGBClassifier
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import librosa

general_path = 'Data'

w, sr = librosa.load(f'{general_path}/test.wav')

# 파일 전 후의 무음구간 제거
audio_file, _ = librosa.effects.trim(w)

# 저장된 모델 및 스케일러 로드
xgb_loaded = joblib.load('xgb_model.joblib')
print("Model loaded from 'xgb_model.joblib'")

min_max_scaler_loaded = joblib.load('min_max_scaler.joblib')
print("MinMaxScaler loaded from 'min_max_scaler.joblib'")

label_encoder_loaded = joblib.load('label_encoder.joblib')
print("Model loaded from 'label_encoder.joblib'")

hop_length = 5000
chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
y_harm, y_perc = librosa.effects.hpss(audio_file)
tempo, _ = librosa.beat.beat_track(y=w, sr = sr)

audio_data = [chromagram.mean(), chromagram.var(),
            spectral_centroids.mean(), spectral_centroids.var(),
            spectral_rolloff.mean(), spectral_rolloff.var(),
            y_harm.mean(), y_harm.var(),
            y_perc.mean(), y_perc.var(),
            tempo]

mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)
for mfcc in mfccs:
    audio_data.append(mfcc.mean())
    audio_data.append(mfcc.var())

# feature name
feature_names = ['chroma_stft_mean', 'chroma_stft_var',
                 'spectral_centroid_mean', 'spectral_centroid_var',
                 'rolloff_mean', 'rolloff_var',
                 'harmony_mean', 'harmony_var',
                 'perceptr_mean', 'perceptr_var',
                 'tempo']

for i in range(1, 21):
    feature_names.append(f'mfcc{i}_mean')
    feature_names.append(f'mfcc{i}_var')

audio_data_df = pd.DataFrame([audio_data], columns=feature_names)

# NORMALIZE X
audio_data_scaled = min_max_scaler_loaded.transform(audio_data_df)

new_audio_pred = xgb_loaded.predict(audio_data_scaled)
predicted_label = label_encoder_loaded.inverse_transform(new_audio_pred)
print('Predicted label for new audio data:', predicted_label[0])