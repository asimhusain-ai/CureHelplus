import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

# --- Load Dataset ---
try:
    df = pd.read_csv("diabetes.csv")
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please make sure the dataset file is in the correct directory.")
    exit()


# --- Data Preprocessing ---

cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)


for col in cols_to_replace:
    df[col].fillna(df[col].median(), inplace=True)

print("\nDataset after preprocessing (handling zero values):")
print(df.describe())


X = df.drop(columns=['Outcome'])
y = df['Outcome']


# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures have been scaled using StandardScaler.")


# --- Train the XGBoost Model ---
model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    use_label_encoder=False,  
    eval_metric='logloss',    
    random_state=42
)

print("\nTraining the XGBoost model...")
model.fit(X_train_scaled, y_train)
print("Model training complete.")


# --- Evaluate the Model ---
print("\nEvaluating the model on the test set...")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))


# --- Save the Model and Scaler ---
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "diabetes_scaler.pkl")

print("\n--- Model and Scaler Saved ---")
print("Trained model saved to: diabetes_model.pkl")
print("Fitted scaler saved to: diabetes_scaler.pkl")