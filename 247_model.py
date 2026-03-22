# ==========================================
# TRAIN RANDOM FOREST
# ==========================================
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


MODEL_OUTPUT = "tamil_sign_rf_247.pkl"
print("\nTraining Random Forest...\n")

df = pd.read_csv("landmarks_twohand_247.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=45,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ RANDOM FOREST RESULTS")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_OUTPUT)

print("\nModel saved:", MODEL_OUTPUT)
print("Training complete 🚀")