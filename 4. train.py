from xgboost import XGBClassifier
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa

general_path = 'Data'

data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:] 

y = data['label']
X = data.loc[:, data.columns.str.contains('chroma|centroid|rolloff|harmony|perceptr|tempo|mfcc')]

# NORMALIZE X
cols = X.columns
print(cols)
# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(X)
# X = pd.DataFrame(np_scaled, columns = cols)

# # 레이블 인코딩
# label_encoder = preprocessing.LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# # Final model
# xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# xgb.fit(X_train, y_train)
# preds = xgb.predict(X_test)
# print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')

# # 모델 저장
# joblib.dump(xgb, 'xgb_model.joblib')
# print("Model saved as 'xgb_model.joblib'")

# joblib.dump(min_max_scaler, 'min_max_scaler.joblib')
# print("MinMaxScaler saved as 'min_max_scaler.joblib'")

# joblib.dump(label_encoder, 'label_encoder.joblib')
# print("MinMaxScaler saved as 'label_encoder.joblib'")