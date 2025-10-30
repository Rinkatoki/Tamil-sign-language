import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# 1️⃣ Load the new CSV
df = pd.read_csv("landmarks_twohand.csv")

# 2️⃣ Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# 3️⃣ Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 4️⃣ Train a Random Forest model
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=45,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5️⃣ Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# 6️⃣ Save model and classes
joblib.dump(model, "tamil_sign_twohand_model.pkl")
np.save("label_classes.npy", y.unique())
print("🎉 Model saved as tamil_sign_twohand_model.pkl")
