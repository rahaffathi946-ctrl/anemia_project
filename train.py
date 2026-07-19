"""
Anemia Types Classification - Training Pipeline
Adapted from Untitled15.ipynb (Colab) to run headless.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==========================================================
# Load Data
# ==========================================================
df = pd.read_csv("dd.csv", sep=";")
print("Shape:", df.shape)
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()

# ==========================================================
# Clinical range validation (same ranges as notebook)
# ==========================================================
clinical_ranges = {
    "WBC":   (0.1, 200),
    "LYMp":  (0, 100),
    "NEUTp": (0, 100),
    "LYMn":  (0, 50),
    "NEUTn": (0, 100),
    "RBC":   (0.5, 8),
    "HGB":   (1, 25),
    "HCT":   (5, 70),
    "MCV":   (40, 150),
    "MCH":   (10, 50),
    "MCHC":  (20, 45),
    "PLT":   (1, 1500),
    "PDW":   (5, 30),
    "PCT":   (0.01, 2)
}

for col, (mn, mx) in clinical_ranges.items():
    if col in df.columns:
        mask = (df[col] < mn) | (df[col] > mx)
        cnt = mask.sum()
        if cnt > 0:
            print(f"{col}: {cnt} out-of-range values -> NaN")
            df.loc[mask, col] = np.nan

numeric_columns = df.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy="median")
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# ==========================================================
# Encode target
# ==========================================================
target_column = "Diagnosis"
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

print("\nClasses:")
for i, c in enumerate(label_encoder.classes_):
    print(f"  {i} -> {c}")

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nClass distribution:\n", y.value_counts().sort_index())

# ==========================================================
# Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# SMOTE on training only
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# Models
# ==========================================================
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
xgboost_model = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", eval_metric="mlogloss", random_state=42
)

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train_scaled, y_train)

print("Training Random Forest...")
random_forest_model.fit(X_train_smote, y_train_smote)

print("Training XGBoost...")
xgboost_model.fit(X_train_smote, y_train_smote)

# ==========================================================
# Predictions
# ==========================================================
lr_pred = logistic_model.predict(X_test_scaled)
lr_prob = logistic_model.predict_proba(X_test_scaled)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)

xgb_pred = xgboost_model.predict(X_test)
xgb_prob = xgboost_model.predict_proba(X_test)

classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)


def evaluate_model(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
    return {
        "Model": name, "Accuracy": accuracy, "Precision (Macro)": precision,
        "Recall (Macro)": recall, "F1-score (Macro)": f1, "ROC-AUC (OvR)": roc_auc
    }


lr_results = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
rf_results = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
xgb_results = evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)

results_df = pd.DataFrame([lr_results, rf_results, xgb_results])
print("\nFinal Comparison:")
print(results_df.to_string(index=False))

# ==========================================================
# Feature importance
# ==========================================================
rf_importance = pd.DataFrame({
    "Feature": X.columns, "Importance": random_forest_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nTop RF Features:\n", rf_importance.head(10).to_string(index=False))

# ==========================================================
# Select + save best model
# ==========================================================
best_model_name = results_df.loc[results_df["F1-score (Macro)"].idxmax(), "Model"]
print("\nBest model (by macro F1):", best_model_name)

if best_model_name == "Logistic Regression":
    best_model = logistic_model
elif best_model_name == "Random Forest":
    best_model = random_forest_model
else:
    best_model = xgboost_model

joblib.dump(best_model, "best_anemia_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")
# Save median values per feature so the web app can fill in any CBC
# fields the user leaves blank (same UX as the original site).
joblib.dump(dict(zip(X.columns, imputer.statistics_)), "feature_medians.pkl")

results_df.to_csv("model_comparison.csv", index=False)
rf_importance.to_csv("feature_importance.csv", index=False)

print("\nSaved: best_anemia_model.pkl, scaler.pkl, label_encoder.pkl, feature_names.pkl")
print("Done.")
