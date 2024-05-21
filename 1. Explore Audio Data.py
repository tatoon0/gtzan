# Usual Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display

import os
general_path = 'Data'

# y: sound
# sr: sample rate
y, sr = librosa.load(f'{general_path}/genres_original/reggae/reggae.00036.wav')

print('y:', y, '\n')
print('y shape:', np.shape(y), '\n')
print('Sample Rate (KHz):', sr, '\n')

# y / sr = len of audio
print('Check Len of Audio:', 661794/22050, '\n')

# 파일 전 후의 무음구간 제거
audio_file, _ = librosa.effects.trim(y)

# waveform 시각화
plt.figure(figsize = (16, 6))
librosa.display.waveshow(y = audio_file, sr = sr, color = "#A300F9");
plt.title("Sound Waves in Reggae 36", fontsize = 23);
plt.show()

# 푸리에 변환
n_fft = 2048
hop_length = 512
D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))

# Spectrogram (DB 스케일로 변환)
DB = librosa.amplitude_to_db(D, ref = np.max)
plt.figure(figsize = (16, 6))
librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool')
plt.colorbar()
plt.show()

# Mel Spectrogram (인간의 청각특성에 맞춰 저주파에서는 고해상도, 고주파에서는 저해상도)
y, sr = librosa.load(f'{general_path}/genres_original/metal/metal.00036.wav')
y, _ = librosa.effects.trim(y)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize = (16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool');
plt.colorbar()
plt.title("Metal Mel Spectrogram", fontsize = 23);
plt.show()

y, sr = librosa.load(f'{general_path}/genres_original/classical/classical.00036.wav')
y, _ = librosa.effects.trim(y)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize = (16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool');
plt.colorbar()
plt.title("Classical Mel Spectrogram", fontsize = 23);
plt.show()

# Harmonics: 음색
# Perceptrual: 리듬이나 감정을 전달하는 파형 (드럼 같은 강한 충격파를 뜻하는 듯)
y_harm, y_perc = librosa.effects.hpss(audio_file)
plt.figure(figsize = (16, 6))
plt.plot(y_harm, color = '#A300F9'); # 보라색
plt.plot(y_perc, color = '#FFB100'); # 노란색
plt.show()

# BPM (beats per minute)
tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
print('tempo', tempo)

# Spectral Centroid는 소리의 중심이 어디에 위치하는지를 나타내며, 이는 소리에 존재하는 주파수들의 가중 평균으로 계산된다.
spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
plt.figure(figsize = (16, 6))
librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color = '#A300F9'); # 보라색
plt.plot(t, normalize(spectral_centroids), color='#FFB100'); # 노란색
plt.show()

# Spectral RollOff (85% 스펙트럴 롤오프는 스펙트럼 에너지의 85%가 해당 주파수 이하에 존재하는 주파수를 나타낸다.)
spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
plt.figure(figsize = (16, 6))
librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color = '#A300F9'); # 보라색
plt.plot(t, normalize(spectral_rolloff), color='#FFB100'); # 노란색
plt.show()

# signal ==FFT,DB==> spectrum ==MEL filter==> mel spectrum ==Log,IFFT==> MFCC
mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool')
plt.show()

# scaled
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
plt.show()

# chroma (분석한 주파수를 12음계 단위로 합산)
hop_length = 5000
chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm');
plt.show()