import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SIZE_MAPPING = {'XXS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6, 'XXXL': 7}
SIZE_REVERSE_MAPPING = {v: k for k, v in SIZE_MAPPING.items()}

# --- Load data ---
df = pd.read_csv('dataset_size.csv')
print(df.shape)
print(df.head())
print(df.describe())

# --- Cek missing value ---
print(df.isnull().sum())

for col in ['age', 'height', 'weight']:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# --- Distribusi kelas ---
print(df['size'].value_counts())

plt.figure(figsize=(6, 4))
df['size'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Distribusi Ukuran Pakaian')
plt.xlabel('Size')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=200)
plt.close()

# --- Encode target ---
df['size'] = df['size'].map(SIZE_MAPPING)

X = df[['age', 'height', 'weight']]
y = df['size']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# --- Training & tuning tiap model ---
models = {}
cv_results = {}

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), rf_params, cv=5)
rf.fit(X_train, y_train)
models['Random Forest'] = rf.best_estimator_
print('RF best params:', rf.best_params_)

# KNN
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn.fit(X_train, y_train)
models['KNN'] = knn.best_estimator_
print('KNN best params:', knn.best_params_)

# Logistic Regression
lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
lr = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), lr_params, cv=5)
lr.fit(X_train, y_train)
models['Logistic Regression'] = lr.best_estimator_
print('LR best params:', lr.best_params_)

# SVM (Linear)
svm_params = {'C': [0.1, 1, 10], 'max_iter': [2000]}
svm = GridSearchCV(LinearSVC(random_state=RANDOM_STATE), svm_params, cv=5)
svm.fit(X_train, y_train)
models['SVM'] = svm.best_estimator_
print('SVM best params:', svm.best_params_)

# Cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: CV mean = {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Evaluasi di test set ---
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=list(SIZE_MAPPING.keys())))

# Confusion matrix semua model
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=SIZE_MAPPING.keys(), yticklabels=SIZE_MAPPING.keys())
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=200)
plt.close()

# --- Tabel perbandingan model ---
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy': [results[n]['accuracy'] for n in models],
    'Precision': [results[n]['precision'] for n in models],
    'Recall': [results[n]['recall'] for n in models],
    'F1 Score': [results[n]['f1'] for n in models],
    'CV Mean': [cv_results[n]['mean'] for n in models],
    'CV Std': [cv_results[n]['std'] for n in models],
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

print(comparison_df.to_string(index=False))
comparison_df.to_csv('model_comparison.csv', index=False)

plt.figure(figsize=(8, 5))
plt.barh(comparison_df['Model'], comparison_df['Accuracy'], color='steelblue')
plt.xlabel('Accuracy')
plt.title('Perbandingan Akurasi Model')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=200)
plt.close()

rf_model = models['Random Forest']
importance = pd.Series(rf_model.feature_importances_, index=['age', 'height', 'weight'])
importance = importance.sort_values(ascending=False)
print(importance)

plt.figure(figsize=(6, 4))
importance.plot(kind='barh', color='#4ECDC4')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=200)
plt.close()

# --- Fungsi prediksi dan melihat model terbaik ---
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"Model terbaik: {best_model_name} (accuracy {comparison_df.iloc[0]['Accuracy']:.4f})")


def predict_clothing_size(age, height, weight, model=best_model, scaler=scaler):
    """Prediksi ukuran pakaian dari umur, tinggi (cm), dan berat (kg)."""
    input_data = pd.DataFrame([[age, height, weight]], columns=['age', 'height', 'weight'])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    return SIZE_REVERSE_MAPPING[pred]


if __name__ == '__main__':
    ukuran_baju = [(26, 155, 59)]
    for age, height, weight in ukuran_baju:
        size = predict_clothing_size(age, height, weight)
        print(f"Age={age}, Height={height}cm, Weight={weight}kg -> Size {size}")
