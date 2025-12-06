import pandas as pd
import joblib
import numpy as np

# Load model
model_path = 'model/final_rf_model.pkl'
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)
print(f"Model type: {type(model)}")

# Load feature names
try:
    model_columns = joblib.load('model/model_columns.pkl')
    feature_names = model_columns
except:
    feature_names = [
        'Civil Status_Widowed', 'Infractions Count', 'Jail Behavior Rating_Poor',
        'Prior Convictions_2 - Robbery', 'Job History_Clerk', 'Prior Convictions_0 - Assault',
        'Prior Convictions_0 - Estafa', 'Prior Convictions_0 - Fraud', 'Vocational Training_No',
        'Post-Release Housing_Homeless', 'Homelessness_No', 'Prior Convictions_2 - Assault',
        'Civil Status_Separated', 'Therapy Attendance_No', 'Prior Convictions_1 - Theft',
        'Type of Current Offense_Illegal Drugs', 'Rehab Attitude_Cooperative',
        'Job History_Unemployed', 'Prior Convictions_3 - Illegal Drugs', 'Prior Convictions_3 - Theft'
    ]

# Check if it's a Pipeline
if hasattr(model, 'steps'):
    # Assume the last step is the classifier
    classifier = model.steps[-1][1]
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        # Create DataFrame
        feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
        
        print("TOP FEATURES:")
        print(feature_imp.head(10).to_string())
        
        print("\nHIGH RISK FEATURES:")
        high_risk_features = [
            'Infractions Count', 'Jail Behavior Rating_Poor', 'Post-Release Housing_Homeless',
            'Vocational Training_No', 'Therapy Attendance_No', 'Type of Current Offense_Illegal Drugs',
            'Job History_Unemployed', 'Prior Convictions_3 - Illegal Drugs'
        ]
        for f in high_risk_features:
            if f in feature_imp['Feature'].values:
                imp = feature_imp[feature_imp['Feature'] == f]['Importance'].values[0]
                print(f"{f}: {imp:.4f}")
    else:
        print("NO IMPORTANCES")
else:
    print("NOT PIPELINE")
