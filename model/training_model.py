import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
# Crucially, use the Pipeline from imblearn for handling resampling steps correctly within CV
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- Config ---
model_choice = "rf"  # "rf", "lr", or "svm"
feature_file = "data/newtop20_features.csv"  
label_file = "data/y_trained.csv" # *** CORRECTED FILENAME ***

# --- Data Loading ---
print("Loading data...")
try:
    X = pd.read_csv(feature_file)
    y = pd.read_csv(label_file).squeeze()
    print(f"   Loaded X with shape {X.shape} and y with shape {y.shape}")

    # Ensure y is numeric if it's not already (e.g., 'No'/'Yes')
    if y.dtype == object or y.dtype.name == 'category':
        print("   Converting target labels to numeric (No=0, Yes=1)...")
        y = y.map({'No': 0, 'Yes': 1})
        if y.isnull().any():
             raise ValueError("Mapping 'No'/'Yes' to 0/1 resulted in NaNs. Check label values.")

    # Verify row count match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")

    print("  Data loaded and validated successfully.")

except FileNotFoundError as e:
    print(f" Error: Could not find required file: {e}")
    print("   Please ensure the feature engineering script has run and saved the correct files.")
    raise
except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise

# --- Train/Test Split ---
print(" Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# --- Model Selection & Pipeline Construction ---
print(f"🔍 Setting up model '{model_choice}' with ImbPipeline (including SMOTE within CV)...")

# Decide if scaler is needed
use_scaler = model_choice in ["lr", "svm"]

# Build the ImbPipeline steps
steps = []
steps.append(('smote', SMOTE(random_state=42)))

# Add scaler if needed
if use_scaler:
    steps.append(('scaler', StandardScaler()))

# Add the classifier
if model_choice == "rf":
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced') # Consider class_weight with SMOTE
    param_grid = {
        "clf__n_estimators": [10, 300, 500],
        "clf__max_depth": [None, 10, 20, 50],
        "clf__min_samples_split": [2, 3, 5],
        "clf__min_samples_leaf": [1, 2, 4]
    }
elif model_choice == "lr":
    base_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Consider class_weight with SMOTE
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__max_iter": [1000]
    }
elif model_choice == "svm":
    base_model = SVC(probability=True, random_state=42, class_weight='balanced') # probability=True for calibration, consider class_weight with SMOTE
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"]
    }
else:
    raise ValueError("model_choice must be 'rf', 'lr', or 'svm'")

# Add the classifier to the steps
steps.append(('clf', base_model))

# Create the ImbPipeline
# Note: We don't need 'clf__' prefix for parameters in param_grid because
# the final step is named 'clf' in the pipeline.
pipe = ImbPipeline(steps)
print(f"   Pipeline steps: {pipe.named_steps.keys()}")

print("Starting hyperparameter tuning with GridSearchCV...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)

try:

    grid_search.fit(X_train, y_train)
    print("   Hyperparameter tuning complete.")
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Best Cross-Validation AUC-ROC: {grid_search.best_score_:.4f}")

except Exception as e:
    print(f"Error during hyperparameter tuning: {e}")
    raise

# --- Probability Calibration ---
print("📏 Calibrating model probabilities...")
try:
    calibrated_clf = CalibratedClassifierCV(grid_search.best_estimator_, cv="prefit")
    calibrated_clf.fit(X_train, y_train) 
    print("   ✅ Model calibration complete.")
except Exception as e:
    print(f"Error during model calibration: {e}")
    raise

# --- Evaluation on Test Set ---
print("Evaluating final calibrated model on the test set...")
try:
    y_pred = calibrated_clf.predict(X_test)
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1] # Probability for the positive class

    # Calculate confidence metrics
    # Confidence is defined as how far the probability is from 0.5 (the decision boundary)
    confidence_scores = np.abs(y_proba - 0.5) * 2  # Scale to [0, 1] where 1 is highest confidence
    mean_confidence = np.mean(confidence_scores)
    std_confidence = np.std(confidence_scores)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Mean Confidence: {mean_confidence:.4f}")
    print(f"Std Confidence: {std_confidence:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

except Exception as e:
    print(f"Error during test set evaluation: {e}")
    raise

# --- Save Model and Columns ---
print("Saving the final model and feature columns...")
try:
    os.makedirs("model", exist_ok=True)
    joblib.dump(list(X.columns), 'model/model_columns.pkl')
    joblib.dump(calibrated_clf, 'model/final_model.pkl') # Save the calibrated model
    joblib.dump(calibrated_clf, 'model/final_rf_model.pkl')  # Also save with the specific name for RF model
    print(" Final calibrated model and columns saved to 'model/' directory.")
except Exception as e:
    print(f"Error saving model or columns: {e}")
    raise

print("\n Training, tuning, calibration, and saving completed successfully.")
