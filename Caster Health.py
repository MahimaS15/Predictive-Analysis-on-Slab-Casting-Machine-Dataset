# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 01:41:31 2025

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: LOAD AND CLEAN DATA

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Documents\25th_oct_23_na25084001_2_BO.csv")

#checking first few rows and info
print(df.head())
print(df.info())
print(df.dtypes)

#removing duplicates
df = df.drop_duplicates()

#dropping irrelevant columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Drop non-numeric columns and NaN columns
df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1)

#Filter Dataframe
selected_columns=['F3-1_B9-1_opp_diff_cur_tm','F4-1_B8-1_opp_diff_cur_tm','F5-1_B7-1_opp_diff_cur_tm','F6-1_B6-1_opp_diff_cur_tm','F7-1_B5-1_opp_diff_cur_tm','F8-1_B4-1_opp_diff_cur_tm','F9-1_B3-1_opp_diff_cur_tm','N1-1_S2-1_opp_diff_cur_tm']
selected_columns=['N2-1_S1-1_opp_diff_cur_tm','F3-2_B9-2_opp_diff_cur_tm','F4-2_B8-2_opp_diff_cur_tm','F5-2_B7-2_opp_diff_cur_tm','F6-2_B6-2_opp_diff_cur_tm','F7-2_B5-2_opp_diff_cur_tm','F8-2_B4-2_opp_diff_cur_tm','F9-2_B3-2_opp_diff_cur_tm']
#Compute correlation
corr_matrix=df[selected_columns].corr()
#Plot Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',linewidths=0.5)
plt.title("Heatmap of selected columns")
plt.tight_layout()
plt.show()

#Box Plot
df[['Casting Speed','Mold Level','Carbon Percent']].plot.box()
plt.show()

#Distplot for a specific column
plt.subplot(1,2,1)
sns.histplot(df['Casting Speed'], kde=True, color='skyblue')
plt.title('Histogram for Casting speed')
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR)))
df_filtered = df_numeric[outlier_mask.sum(axis=1) > int(0.8 * df_numeric.shape[1])]

print("[INFO] Data shape after outlier removal:", df_filtered.shape)

# STEP 2: UNSUPERVISED MODELLING (PCA + KMeans)

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_filtered)
print("Data shape of Standard Scaler:", data_scaled.shape)

# PCA to 2 components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)

# PCA Plot
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=kmeans_labels, cmap='viridis')
plt.title("PCA + KMeans Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# STEP 3: LABEL CREATION

# Reset index
df_model = df_filtered.reset_index(drop=True)
df_model['label'] = 0
df_model.loc[df_model.index[-300:], 'label'] = 1
print("[INFO] Class distribution:\n", df_model['label'].value_counts())

# STEP 4: BALANCING WITH SMOTE

X = df_model.drop(columns=['label'])
y = df_model['label']

# SMOTE to reach 1:0.25 ratio
smote = SMOTE(sampling_strategy=0.25, random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)
print("[INFO] After SMOTE class counts:", np.bincount(y_sm))

# STEP 5: SPLITTING

X_temp, X_val, y_temp, y_val = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# STEP 6: TRAIN MODELS

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# STEP 7: EVALUATION FUNCTION

def evaluate_model(model, X, y, label):
    print(f"\n[INFO] Evaluation for {label}")
    y_pred = model.predict(X)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

evaluate_model(log_model, X_train, y_train, "Logistic - Train")
evaluate_model(log_model, X_test, y_test, "Logistic - Test")
evaluate_model(log_model, X_val, y_val, "Logistic - Validation")

evaluate_model(xgb_model, X_train, y_train, "XGB - Train")
evaluate_model(xgb_model, X_test, y_test, "XGB - Test")
evaluate_model(xgb_model, X_val, y_val, "XGB - Validation")

# STEP 8: FEATURE IMPORTANCE FROM XGBOOST

# Ensure matching length of features
assert len(X.columns) == len(xgb_model.feature_importances_)

# Top 20 features
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)
print(top_features)
top_features = top_features[top_features > 0]

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 20 Important Features (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()