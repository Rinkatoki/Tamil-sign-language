# ==========================================
# TRAIN XGBOOST MODEL
# ==========================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


MODEL_OUTPUT = "tamil_sign_xgb_247.pkl"
ENCODER_OUTPUT = "label_encoder_247.pkl"

print("\nTraining XGBoost...\n")


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("landmarks_twohand_247_tamil.csv")

X = df.drop(columns=["label"])
y = df["label"]


# ==========================================
# ENCODE LABELS (required for XGBoost)
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# DEFINE MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ XGBOOST RESULTS")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# SAVE MODEL + LABEL ENCODER
# ==========================================

joblib.dump(model, MODEL_OUTPUT)
joblib.dump(label_encoder, ENCODER_OUTPUT)

print("\nModel saved:", MODEL_OUTPUT)
print("Label encoder saved:", ENCODER_OUTPUT)
print("Training complete 🚀")