# ==========================================
# TRAIN XGBOOST MODEL
# ==========================================
from sklearn.preprocessing import StandardScaler



import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

from sklearn.decomposition import PCA



pca = PCA(0.98)


iso = IsolationForest(contamination=0.02, random_state=42)



MODEL_OUTPUT = "tamil_sign_xgb_247.pkl"
ENCODER_OUTPUT = "label_encoder_247.pkl"

print("\nTraining XGBoost...\n")


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("landmarks_twohand_247_tamil.csv")

X = df.drop(columns=["label"])
y = df["label"]

X = pca.fit_transform(X)
# ==========================================
# ENCODE LABELS (required for XGBoost)
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
mask = iso.fit_predict(X) == 1


X = X[mask]
y_encoded = y_encoded[mask]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

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
    n_estimators=1200,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.85,
    gamma=0.2,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=True
)


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
joblib.dump(scaler, "scaler_247.pkl")
joblib.dump(pca, "pca_247.pkl")
print("\nModel saved:", MODEL_OUTPUT)
print("Label encoder saved:", ENCODER_OUTPUT)
print("Training complete 🚀")