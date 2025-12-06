import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import os
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Load Data

print("Loading data...")
X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv").squeeze()
print(f"   Loaded X with shape {X.shape} and y with shape {y.shape}")

# Drop Columns with Too Much Missing Data
missing_threshold = 0.5
missing_ratios = X.isnull().mean()
drop_missing = missing_ratios[missing_ratios > missing_threshold].index.tolist()
if drop_missing:
    X.drop(columns=drop_missing, inplace=True)
    print(f" Dropped columns with >{missing_threshold*100}% missing values: {drop_missing}")
else:
    print(" No columns exceeded the missing data threshold.")


# Handle Missing Values + Scale

print(" Handling missing values and ensuring numeric data types...")
X = X.apply(pd.to_numeric, errors='coerce')  # Ensure numeric
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)
print(" Missing values imputed.")

# Remove zero-variance columns
initial_cols = X.shape[1]
X = X.loc[:, X.var() > 0]
final_cols = X.shape[1]
if initial_cols > final_cols:
    print(f"   🗑️ Removed {initial_cols - final_cols} zero-variance columns.")
else:
    print("   No zero-variance columns found.")

# Define RNR Categories with Domain Logic (updated to include multi-label features)
print("Defining RNR categories...")
RNR_RISK = [
    "Age", "Length of Current Sentence (yrs)", "Time Served (yrs)", "Infractions Count", "Solitary Time (days)",
    "Friends w/ Record", "Gang Affiliation_Yes", "Conflict in Jail_Yes", "Peer Influence_Yes",
    "Juvenile Records_Yes", "Prior Convictions_1 - Theft", "Prior Convictions_2 - Robbery",
    "Prior Convictions_3 - Homicide", "Jail Behavior Rating_Poor", "Psych Assessment Result_High",
    "Psych Assessment Result_Very High", "Impulsivity_High", "Manipulativeness_High",
    "Aggression_High", "Remorse_No", "Law Belief_Low"
    # Add multi-label features for Program Types, Stress Sources, Support Needed
] + [col for col in X.columns if col.startswith(('Program Types_', 'Stress Sources_', 'Support Needed_'))]

RNR_NEED = [
    "Rehab Programs_Yes", "Therapy Attendance_Yes", "Drug Rehab_Yes", "Education in Jail_Yes",
    "Vocational Training_Yes", "Family Relationship_Sstrained", "Domestic Trauma_Yes",
    "Family Reunification Intent_Yes", "Job History_Unemployed", "Living Before Arrest_Homeless",
    "Homelessness_Yes", "Juvenile Street Living_Yes", "Post-Release Housing_Homeless",
    "Post-Release Housing_Halfway House", "Post-Release Programs_Yes", "NGO Support_Yes",
    "Job After Release_Unemployed"
    # Add multi-label features for Program Types, Stress Sources, Support Needed
] + [col for col in X.columns if col.startswith(('Program Types_', 'Support Needed_'))]

RNR_RESPONSIVITY = [
    "Rehab Attitude_Willing", "Rehab Attitude_Cooperative", "Remorse_Yes", "Remorse_Some",
    "Law Belief_High", "Law Belief_Moderate", "Psych Assessment Result_Low",
    "Psych Assessment Result_Moderate", "Impulsivity_Low", "Impulsivity_Moderate",
    "Manipulativeness_Low", "Manipulativeness_Moderate", "Aggression_Low",
    "Aggression_Moderate", "Family Relationship_Strong", "Peer Type_Prosocial"
    # Add multi-label features for Stress Sources (as they might affect responsivity)
] + [col for col in X.columns if col.startswith('Stress Sources_')]

# Filter existing RNR features
RNR_RISK = [f for f in RNR_RISK if f in X.columns]
RNR_NEED = [f for f in RNR_NEED if f in X.columns]
RNR_RESPONSIVITY = [f for f in RNR_RESPONSIVITY if f in X.columns]

# Create Enhanced RNR Composite Scores
print("Creating RNR composite scores...")
RNR_ALL = [] # Initialize RNR_ALL to collect scores

if RNR_RISK:
    # Calculate risk score with multi-label features included
    risk_cols = [f for f in RNR_RISK if f not in ["Age"]]
    X["RNR_Risk_Score"] = (
        X["Age"] * 0.2 +
        (X[risk_cols].sum(axis=1) if risk_cols else 0) * 0.8
    )
    RNR_ALL.append("RNR_Risk_Score")
    print("   Created RNR_Risk_Score")
if RNR_NEED:
    # Calculate need score with multi-label features included
    need_cols = [f for f in RNR_NEED]
    X["RNR_Need_Score"] = X[need_cols].sum(axis=1) if need_cols else pd.Series([0] * len(X), index=X.index)
    RNR_ALL.append("RNR_Need_Score")
    print("   Created RNR_Need_Score")
if RNR_RESPONSIVITY:
    # Calculate responsivity score with multi-label features included
    resp_cols = [f for f in RNR_RESPONSIVITY]
    X["RNR_Responsivity_Score"] = X[resp_cols].sum(axis=1) if resp_cols else pd.Series([0] * len(X), index=X.index)
    RNR_ALL.append("RNR_Responsivity_Score")
    print("   Created RNR_Responsivity_Score")

# Update RNR_ALL with individual features (used for retention logic)
RNR_ALL_individual = RNR_RISK + RNR_NEED + RNR_RESPONSIVITY 
RNR_ALL.extend(RNR_ALL_individual) 

# Remove Potential Leakage Features (e.g., Barangay)
barangay_cols = [col for col in X.columns if 'barangay' in col.lower()]
if barangay_cols:
    X.drop(columns=barangay_cols, inplace=True)
    print(f" Removed potential leakage features: {barangay_cols}")
    # Update RNR_ALL to remove any dropped features
    RNR_ALL = [f for f in RNR_ALL if f not in barangay_cols]
    RNR_ALL_individual = [f for f in RNR_ALL_individual if f not in barangay_cols]
else:
    print("  No Barangay-related leakage features found.")

# Synthetic Data Augmentation (Add Realism) - FIXED
def augment_data(df_X, df_y, n_samples=200):

    print(f"🔄 Augmenting data with {n_samples} synthetic samples...")
    df_X_aug = df_X.copy()
    df_y_aug = df_y.copy()
    new_X_rows = []
    new_y_rows = []

    for _ in range(n_samples):
        sampled_idx = df_X.sample(1).index[0]
        row_X = df_X.loc[sampled_idx].copy()
        row_y = df_y.loc[sampled_idx]

        # Perturb numerical features (in X)
        row_X["Age"] = np.clip(row_X["Age"] + np.random.normal(0, 1), 18, 60)
        row_X["Length of Current Sentence (yrs)"] = np.clip(
            row_X["Length of Current Sentence (yrs)"] + np.random.normal(0, 0.5), 0, 20
        )

        # Apply small perturbations to multi-label features by randomly toggling some values
        for col in df_X.columns:
            if col.startswith(('Program Types_', 'Stress Sources_', 'Support Needed_')):
                # Randomly flip some values for multi-label features with low probability
                if np.random.random() < 0.05:  # 5% chance to flip
                    row_X[col] = 1 - row_X[col]

        new_X_rows.append(row_X)
        new_y_rows.append(row_y)

    # Concatenate the new rows to the original DataFrames
    if new_X_rows:
        df_X_new = pd.DataFrame(new_X_rows)
        df_y_new = pd.Series(new_y_rows, name=df_y.name)
        # Check if df_X_new is not empty before concatenating
        if not df_X_new.empty:
            df_X_aug = pd.concat([df_X_aug, df_X_new], ignore_index=True)
        # Check if df_y_new is not empty before concatenating
        if not df_y_new.empty:
            df_y_aug = pd.concat([df_y_aug, df_y_new], ignore_index=True)

    print(f" Data augmentation complete. New shapes - X: {df_X_aug.shape}, y: {df_y_aug.shape}")
    return df_X_aug, df_y_aug

X, y = augment_data(X, y, n_samples=200)

print("Balancing classes with SMOTE...")
print("   Class distribution before SMOTE:")
print(y.value_counts(normalize=True).rename("Proportion"))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y) # X, y are augmented here
print(" Class balancing complete.")
print("  Class distribution after SMOTE:")
print(y_resampled.value_counts(normalize=True).rename("Proportion"))

y_resampled.to_csv("data/y_trained.csv", index=False)
print("Saved resampled y_train to data/y_train_resampled.csv")


# SelectKBest Feature Selection

k = min(30, X_resampled.shape[1], X_resampled.shape[0] // 2) # Safer divisor
print(f"Attempting to select top {k} features.")
if (X_resampled < 0).any().any():
    print("Negative values detected — using mutual_info_classif.")
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
else:
    selector = SelectKBest(score_func=chi2, k=k)

try:
    selector.fit(X_resampled, y_resampled)
    # Use X.columns here because we want the original column names that survived preprocessing
    top_kbest = list(X.columns[selector.get_support()])
    print(f"Selected top {len(top_kbest)} features using SelectKBest.")
except Exception as e:
    print(f"Feature selection failed: {e}")
    top_kbest = list(X.columns)
    print(f"Using all {len(top_kbest)} available features.")


# Remove Highly Correlated Features

print("Removing highly correlated features...")
try:
    corr_matrix = X_resampled[top_kbest].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    filtered_features = [col for col in top_kbest if col not in correlated_drop]
    print(f"Removed {len(correlated_drop)} highly correlated features (threshold > 0.85).")
    print(f"Features remaining after correlation filter: {len(filtered_features)}")
except Exception as e:
    print(f"Correlation filtering failed: {e}")
    filtered_features = top_kbest
    print(f"Proceeding with {len(filtered_features)} features from SelectKBest.")

# Retain engineered RNR scores if they exist in the original X
rnr_scores_added = []
for f in RNR_ALL:
    if f not in filtered_features and f in X.columns and f not in RNR_ALL_individual:
        filtered_features.append(f)
        rnr_scores_added.append(f)
if rnr_scores_added:
    print(f" Retained RNR composite scores: {rnr_scores_added}")


# Logistic Regression p-value Ranking
print("Ranking features using Logistic Regression p-values...")
final_k = min(30, len(filtered_features), len(y_resampled) // 2) # Safer divisor
selected_for_lr = filtered_features[:final_k]
print(f"   Preparing {len(selected_for_lr)} features for Logistic Regression.")

if len(selected_for_lr) == 0:
    print("No features available for Logistic Regression ranking.")
    top_features = {20: []} # Only define top 20
else:
    try:
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_resampled[selected_for_lr]), columns=selected_for_lr)
        X_scaled_const = sm.add_constant(X_scaled)
        logit_model = sm.Logit(y_resampled, X_scaled_const)
        logit_result = logit_model.fit(disp=0) # Suppress output
        pvals = logit_result.pvalues.drop("const", errors='ignore') # Drop intercept if present
        if pvals.empty:
             raise ValueError("Logistic regression returned no p-values.")

        top_features = {
            20: pvals.nsmallest(min(20, len(pvals))).index.tolist()
        }
        print(" Logistic regression feature ranking completed.")
    except Exception as e:
        print(f" Logistic regression failed: {e}")
        top_features = {
            20: filtered_features[:min(20, len(filtered_features))]
        }
        print("   Using order from filtered features as fallback ranking.")

# Save Only Top 20 Results
print("💾 Saving top 20 features...")
os.makedirs("data", exist_ok=True)

k = 20
features_to_save = top_features.get(k, [])
if features_to_save:
    try:
        available_features = [f for f in features_to_save if f in X.columns]
        if not available_features:
            print(f"   ⚠️ No available features to save for top {k}.")
        else:
            pd.DataFrame(X_resampled, columns=X.columns)[available_features].to_csv(f"data/newtop{k}_features.csv", index=False)
            print(f"Saved top {k} features (from resampled data) to data/newtop{k}_features.csv")
    except Exception as e:
        print(f"Error saving top {k} features: {e}")
else:
     print(f"No top {k} features list found to save.")


# Feature Importance Scoring (SHAP) - Top 20
print("Calculating SHAP feature importances for top 20...")
try:
    # Split resampled data for SHAP analysis
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )
    print(f"   Split data for SHAP: Train {X_train.shape}, Test {X_test.shape}")

    # Train LightGBM model
    model = LGBMClassifier(
        learning_rate=0.01,
        num_leaves=15,
        n_estimators=500,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   ✅ LightGBM model trained for SHAP analysis.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("   📈 SHAP values calculated.")

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_use = shap_values[1]
        print("   Using SHAP values for the positive class (index 1).")
    else:
        shap_values_to_use = shap_values
        print("   Using provided SHAP values directly.")

    top_20_features_shap = top_features.get(20, [])
    if top_20_features_shap:
        X_test_top20 = X_test[top_20_features_shap]
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_to_use, X_test_top20, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot) - Top 20 Features")
            plt.tight_layout()
            plt.savefig("data/shap_summary_bar_top20.png", dpi=150)
            plt.close()
            print("SHAP bar plot for top 20 saved to data/shap_summary_bar_top20.png")
        except Exception as plot_error:
            print(f"Could not save SHAP bar plot for top 20: {plot_error}")

    shap_importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "SHAP_Importance": np.abs(shap_values_to_use).mean(axis=0)
    }).sort_values(by="SHAP_Importance", ascending=False)

    shap_importance_df.to_csv("data/shap_importance.csv", index=False)
    print("Full SHAP importance saved to data/shap_importance.csv")

    top_20_features = top_features.get(20, [])
    if not top_20_features:
         print("No top 20 features list found to calculate percentages.")
    else:
        # Get SHAP importances for the top 20 features
        top_20_shap = shap_importance_df[shap_importance_df['Feature'].isin(top_20_features)].copy()

        if top_20_shap.empty:
            print("None of the top 20 features were found in SHAP results.")
        else:
            # Normalize SHAP importances to get percentages
            total_importance_top20 = top_20_shap['SHAP_Importance'].sum()
            if total_importance_top20 > 0:
                top_20_shap['Percentage_Contribution'] = (top_20_shap['SHAP_Importance'] / total_importance_top20) * 100
                top_20_shap_sorted = top_20_shap.sort_values(by='Percentage_Contribution', ascending=False)

                # Save to CSV
                top_20_shap_sorted.to_csv("data/top_20_feature_percentages.csv", index=False)
                print("Saved top 20 feature percentages to data/top_20_feature_percentages.csv")

                # Save to TXT file
                with open("data/top_20_feature_percentages.txt", "w") as f:
                    f.write("Top 20 Features Contribution Percentages (based on SHAP)\n")
                    f.write("=" * 60 + "\n")
                    for index, row in top_20_shap_sorted.iterrows():
                        f.write(f"{row['Feature']:<40} {row['Percentage_Contribution']:.2f}%\n")
                print("Saved top 20 feature percentages to data/top_20_feature_percentages.txt")
            else:
                 print("Total SHAP importance for top 20 features is zero.")

except Exception as e:
    print(f"Error during SHAP analysis or percentage calculation: {e}")

# Save the final processed data
print("\nFeature engineering, SHAP analysis, and percentage calculation complete.")
top_20_list = top_features.get(20, [])
if top_20_list:
    print(f"Top 20 predictors (ranked by Logistic Regression p-value):")
    for i, feature in enumerate(top_20_list, 1):
        print(f"   {i:2d}. {feature}")
else:
    print("Top 20 predictors list is empty.")
