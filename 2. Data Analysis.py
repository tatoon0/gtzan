# Usual Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

import os
general_path = 'Data'

# EDA (탐색적 데이터 분석, Exploratory Data Analysis)
# 1. 평균, 중앙값, 분산, 표준편차 등 데이터를 요약
# 2. 데이터 시각화
# 3. 상관관계 분석
# 4. 분포분석
data = pd.read_csv(f'{general_path}/features_30_sec.csv')
spike_cols = [col for col in data.columns if 'mean' in col]
corr = data[spike_cols].corr()

# 상관관계 분석
mask = np.triu(np.ones_like(corr, dtype=np.bool_))
f, ax = plt.subplots(figsize=(16, 11));
cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 25)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);
plt.savefig("Corr Heatmap.jpg")
plt.show()

# 분포 분석
x = data[["label", "tempo"]]
f, ax = plt.subplots(figsize=(16, 9))
sns.boxplot(x="label", y="tempo", data=x, palette='husl', hue="label", dodge=False)
plt.title('BPM Boxplot for Genres', fontsize=25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=10)
plt.xlabel("Genre", fontsize=15)
plt.ylabel("BPM", fontsize=15)
plt.legend([], [], frameon=False)
plt.savefig("BPM Boxplot.jpg")
plt.show()

# PCA
# 고차원의 데이터를 저차원으로 변환하여 데이터의 주요 패턴을 시각화하고 이해하는 데 유용
# 특히 장르와 같은 범주형 변수를 시각화하여 가능한 그룹을 식별하는 데 유용
data = data.iloc[0:, 1:]
y = data['label']
X = data.loc[:, data.columns != 'label']

# NORMALIZE X 
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

# PCA 2 COMPONENTS
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis = 1)

# VISUALIZE
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7, s = 100);
plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert.jpg")
plt.show()