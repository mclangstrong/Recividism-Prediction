import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Load Data
print("Loading data...")
X = pd.read_csv("data/newtop20_features.csv")  # Changed from top_20_feature_percentages.csv to newtop20_features.csv
try:
    y_file_path = "data/y_trained.csv"
    y = pd.read_csv(y_file_path).squeeze()
    print(f"   Loaded X with shape {X.shape} and y (resampled) with shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch after loading: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")
    print(f"Data loaded successfully from '{y_file_path}'.")
except FileNotFoundError:
    error_msg = (f"Error: '{y_file_path}' not found.\n"
                 f"Please ensure the feature engineering script saves the resampled target labels "
                 f"to this file after applying SMOTE.")
    print(error_msg)
    if os.path.exists("data"):
        files = os.listdir("data")
        print(f"Files available in 'data' directory: {files}")
    raise 
except Exception as e:
    print(f"Error loading data: {e}")
    raise


# Clean Up Features (Redundant if already done in feature engineering, but safe)
print("Checking for redundant columns...")

barangay_cols = [col for col in X.columns if 'barangay' in col.lower()]
if barangay_cols:
    X.drop(columns=barangay_cols, inplace=True, errors='ignore') # Add errors='ignore'
    print(f"Removed possible leakage features: {barangay_cols}")
else:
     print("No Barangay-related leakage features found.")


rnr_scores = [col for col in X.columns if 'rnr_' in col.lower()]
if rnr_scores:
    print(f"RNR scores retained: {rnr_scores}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

# Models & Hyperparameters (Enhanced to work better with multi-label features)
print("Setting up models and hyperparameters (Random Forest, Logistic Regression, SVM)...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
}

param_grids = {
    "Logistic Regression": {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    },
    "Random Forest": {
        'clf__n_estimators': [10, 300, 500],
        'clf__max_depth': [5, 10, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', None]
    },
    "SVM": {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.001, 0.01]
    }
}

#  Output Directory
print("Creating output directories...")
os.makedirs("model_eval/eval_reports", exist_ok=True)
os.makedirs("model_eval/roc_curves", exist_ok=True)


# 10 split folds for cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    print(f"\n Training: {name}")

    # Use StandardScaler for linear models, None for tree-based models
    use_scaler = name in ["Logistic Regression", "SVM"]

    steps = []
    steps.append(('smote', SMOTE(random_state=42)))
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', model))

    pipe = ImbPipeline(steps)

    param_grid = param_grids[name]

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

    try:
        print(f"Starting hyperparameter tuning for {name}...")
        grid.fit(X_train, y_train)
        print(f"Hyperparameter tuning complete for {name}.")

        best_model = grid.best_estimator_
        print(f"Best Params for {name}: {grid.best_params_}")
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        # Calculate confidence metrics
        # Confidence is defined as how far the probability is from 0.5 (the decision boundary)
        confidence_scores = np.abs(y_proba - 0.5) * 2  # Scale to [0, 1] where 1 is highest confidence
        mean_confidence = np.mean(confidence_scores)
        std_confidence = np.std(confidence_scores)

        # Calculate metrics using the test set
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        print(f"Test Metrics for {name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(matrix)

        # --- Save evaluation report ---
        report_path = f"model_eval/eval_reports/{name.replace(' ', '_')}_report.txt"
        try:
            with open(report_path, "w") as f:
                f.write(f"Model Type: {name}\n")
                f.write(f"Best Params: {grid.best_params_}\n\n")
                f.write("--- Test Set Metrics ---\n")
                f.write(f"Accuracy  : {acc:.4f}\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n\n")
                f.write(f"Mean Confidence: {mean_confidence:.4f}\n")
                f.write(f"Std Confidence: {std_confidence:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(matrix))
                f.write("\n\nClassification Report:\n")
                f.write(report)
                f.write("\n\n--- Best Pipeline Steps ---\n")
                # Print steps for reference
                for step_name, step_obj in best_model.named_steps.items():
                    f.write(f"Step: {step_name} -> {type(step_obj).__name__}\n")
            print(f"Evaluation report saved to {report_path}")
        except Exception as e:
            print(f"Failed to save report for {name}: {e}")

        # --- Save feature importances/coefficients ---
        try:
            feature_names = X.columns

            clf_step = best_model.named_steps['clf']
            if hasattr(clf_step, 'feature_importances_'):
                importances = clf_step.feature_importances_
                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                feat_imp.to_csv(f"model_eval/eval_reports/{name.replace(' ', '_')}_feature_importance.csv")
                print(f"   💾 Feature importances saved to model_eval/eval_reports/{name.replace(' ', '_')}_feature_importance.csv")
            elif hasattr(clf_step, 'coef_'):
                coef = clf_step.coef_[0] 
                feat_imp = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
                feat_imp.to_csv(f"model_eval/eval_reports/{name.replace(' ', '_')}_feature_importance.csv")
                print(f"Feature coefficients saved to model_eval/eval_reports/{name.replace(' ', '_')}_feature_importance.csv")
            else:
                 print(f"Feature importances not directly available for {name}.")
        except Exception as e:
            print(f"Could not extract/save feature importances for {name}: {e}")
        # --- Save ROC Curve ---
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))

            metrics_label = (f'{name}\n'
                             f'AUC: {auc:.2f}\n'
                             f'Acc: {acc:.2f}\n'
                             f'Prec: {precision:.2f}\n'
                             f'Recall: {recall:.2f}')
            plt.plot(fpr, tpr, label=metrics_label)
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.50)')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {name}")
            plt.legend(loc='lower right')
            plt.grid(True)
            roc_plot_path = f"model_eval/roc_curves/{name.replace(' ', '_')}_roc.png"
            plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"ROC curve (with metrics) saved to {roc_plot_path}")
        except Exception as e:
            print(f"Failed to plot/save ROC curve for {name}: {e}")

        # Append results for summary
        results.append({
            "Model": name,
            "Accuracy": acc,
            "AUC-ROC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Mean Confidence": mean_confidence,
            "Std Confidence": std_confidence
        })

    except Exception as e:
        print(f"Error during training/tuning/evaluation of {name}: {e}")
        import traceback
        print(traceback.format_exc())
        results.append({
            "Model": name,
            "Accuracy": np.nan,
            "AUC-ROC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1 Score": np.nan
        })

# Save Summary
print("\n Saving model performance summary...")
try:
    if not results:
        print(" No results to save.")
    else:
        # Save as CSV
        summary = pd.DataFrame(results)
        summary_sorted = summary.sort_values(by='AUC-ROC', ascending=False).reset_index(drop=True)
        summary_csv_path = "model_eval/eval_reports/model_summary.csv"
        summary_sorted.to_csv(summary_csv_path, index=False)
        print(f"Model summary saved to {summary_csv_path}")

        # Save as JSON for App
        json_results = {row['Model']: row.to_dict() for _, row in summary.iterrows()}
        # Remove 'Model' key from inner dict as it's the key
        for model_name in json_results:
            del json_results[model_name]['Model']
            
        json_path = "model/evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"Model evaluation results saved to {json_path}")

        # Display summary
        print("\nModel Performance Summary (sorted by AUC-ROC):")
        if not summary_sorted.empty:
             print(summary_sorted.round(4).to_string(index=False))
        else:
             print("   No valid results to display.")
        print("\nModel training, evaluation, and reporting complete.")
except Exception as e:
    print(f"Error saving model summary: {e}")
    import traceback
    print(traceback.format_exc())
