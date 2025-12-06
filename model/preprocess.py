import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer
import os
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv("data/syntheticdataset.csv")
target_col = "History of Incarceration"
y = df[target_col].map({'No': 0, 'Yes': 1}).astype(int)
X = df.drop(columns=[target_col])

# Drop columns with >5% missing
missing_pct = X.isnull().mean()
X = X.loc[:, missing_pct <= 0.05]

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Identify multi-label columns that need MLB processing
mlb_cols = [col for col in cat_cols if col in ['Program Types', 'Stress Sources', 'Support Needed']]
regular_cat_cols = [col for col in cat_cols if col not in mlb_cols]

print(f"Multi-label columns found: {mlb_cols}")
print(f"Regular categorical columns: {regular_cat_cols}")

# Group rare categories in categorical columns
def group_rare_categories(df, cols, threshold=0.01):
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, 'N/A')
    return df

X = group_rare_categories(X, regular_cat_cols)

# Split first to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Check for class balance
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))

# Define preprocessing for numeric and categorical features
def clip_outliers(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        low = df[col].quantile(lower)
        high = df[col].quantile(upper)
        df[col] = df[col].clip(low, high)
    return df

X_train = clip_outliers(X_train.copy(), num_cols)
X_test = clip_outliers(X_test.copy(), num_cols)

# Process multi-label columns with MultiLabelBinarizer
X_train_mlb = pd.DataFrame()
X_test_mlb = pd.DataFrame()

# Create directory for MLB encoders if it doesn't exist
os.makedirs("model/mlb", exist_ok=True)

for col in mlb_cols:
    print(f"Processing multi-label column: {col}")
    # Convert string representations of lists to actual lists for training data
    X_train[col] = X_train[col].apply(lambda x: x.split(', ') if pd.notnull(x) and isinstance(x, str) else [])
    X_test[col] = X_test[col].apply(lambda x: x.split(', ') if pd.notnull(x) and isinstance(x, str) else [])
    
    # Fit MLB on training data
    mlb = MultiLabelBinarizer()
    X_train_mlb_temp = mlb.fit_transform(X_train[col])
    X_test_mlb_temp = mlb.transform(X_test[col])
    
    # Create DataFrame with proper column names
    X_train_mlb_temp = pd.DataFrame(X_train_mlb_temp, columns=[f"{col}_{class_}" for class_ in mlb.classes_], index=X_train.index)
    X_test_mlb_temp = pd.DataFrame(X_test_mlb_temp, columns=[f"{col}_{class_}" for class_ in mlb.classes_], index=X_test.index)
    
    # Concatenate to the main DataFrame safely
    if not X_train_mlb_temp.empty:
        if X_train_mlb.empty:
            X_train_mlb = X_train_mlb_temp
        else:
            X_train_mlb = pd.concat([X_train_mlb, X_train_mlb_temp], axis=1)
    if not X_test_mlb_temp.empty:
        if X_test_mlb.empty:
            X_test_mlb = X_test_mlb_temp
        else:
            X_test_mlb = pd.concat([X_test_mlb, X_test_mlb_temp], axis=1)
    
    # Save the fitted MLB encoder
    joblib.dump(mlb, f"model/mlb/mlb_{col}.pkl")
    print(f"Saved MLB encoder for {col}")

# Build pipeline for numeric and regular categorical features
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", MinMaxScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Process numeric and regular categorical features with ColumnTransformer
if num_cols or regular_cat_cols:
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, regular_cat_cols)
    ])

    # Fit on training and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    cat_encoded_names = []
    if regular_cat_cols:
        cat_encoded_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(regular_cat_cols)
    processed_columns = num_cols + list(cat_encoded_names)

    # Convert to DataFrame
    X_train_regular = pd.DataFrame(X_train_processed, columns=processed_columns, index=X_train.index)
    X_test_regular = pd.DataFrame(X_test_processed, columns=processed_columns, index=X_test.index)
else:
    # If no regular features, create empty DataFrames with the same index
    X_train_regular = pd.DataFrame(index=X_train.index)
    X_test_regular = pd.DataFrame(index=X_test.index)

# Combine regular features with MLB features
# Check if both DataFrames are empty before concatenating
if X_train_regular.empty and X_train_mlb.empty:
    # If both are empty, create an empty DataFrame with appropriate index
    X_train_final = pd.DataFrame(index=X_train.index)
elif X_train_mlb.empty:
    X_train_final = X_train_regular
elif X_train_regular.empty:
    X_train_final = X_train_mlb
else:
    X_train_final = pd.concat([X_train_regular, X_train_mlb], axis=1)

if X_test_regular.empty and X_test_mlb.empty:
    # If both are empty, create an empty DataFrame with appropriate index
    X_test_final = pd.DataFrame(index=X_test.index)
elif X_test_mlb.empty:
    X_test_final = X_test_regular
elif X_test_regular.empty:
    X_test_final = X_test_mlb
else:
    X_test_final = pd.concat([X_test_regular, X_test_mlb], axis=1)

# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train_final)
X_test_selected = selector.transform(X_test_final)

selected_columns = X_train_final.columns[selector.get_support()]
X_train_final = pd.DataFrame(X_train_selected, columns=selected_columns, index=X_train.index)
X_test_final = pd.DataFrame(X_test_selected, columns=selected_columns, index=X_test.index)

# Apply SMOTE to handle class imbalance (only on training data!)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

# Save the resampled training data
X_train_final = pd.DataFrame(X_train_resampled, columns=X_train_final.columns)
y_train = pd.Series(y_train_resampled)

# Save results
os.makedirs("data", exist_ok=True)
X_train_final.to_csv("data/X_train.csv", index=False)
X_test_final.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save the preprocessor
if 'preprocessor' in locals():
    joblib.dump(preprocessor, "model/preprocessor.pkl")
    print("Saved preprocessor to model/preprocessor.pkl")
else:
    print("Warning: preprocessor not found, skipping save.")

print("\n Preprocessing complete.")
print(f"Final shapes — Train: {X_train_final.shape}, Test: {X_test_final.shape}")
print("New class distribution in resampled training set:")
print(y_train.value_counts(normalize=True))
print(f"Features after preprocessing: {X_train_final.shape[1]}")
