# Enhanced Model Training Guide

## 📚 Overview

The `enhanced_training.py` script provides comprehensive model training with:
- ✅ Hyperparameter tuning for LR, SVM, and RF
- ✅ SMOTE for class balancing
- ✅ Ensemble stacking for combined predictions
- ✅ Detailed performance metrics
- ✅ Model comparison and selection

## 🚀 How to Run

```bash
# Navigate to project directory
cd "c:\Users\admin\Desktop\Recidivisim Predictor"

# Activate virtual environment
.\.venv\Scripts\activate

# Run the enhanced training script
python model/enhanced_training.py
```

## ⏱️ Expected Runtime

- **Logistic Regression**: ~2-3 minutes
- **SVM**: ~5-10 minutes (most time-intensive)
- **Random Forest**: ~3-5 minutes
- **Stacked Ensemble**: ~1-2 minutes
- **Total**: ~15-20 minutes

## 📊 What Gets Saved

After training completes, you'll find these files in the `model/` directory:

### 1. **final_rf_model.pkl**
   - Best Random Forest model (current production model)
   - Used by the web application

### 2. **stacked_ensemble_model.pkl**
   - Ensemble of LR + SVM + RF
   - Often achieves highest performance
   - Can be used for production if better than RF alone

### 3. **preprocessor.pkl**
   - StandardScaler for feature normalization
   - Required for making predictions

### 4. **model_columns.pkl**
   - Feature column names
   - Ensures correct feature alignment

### 5. **evaluation_results.json**
   - Performance metrics for all models
   - Includes: Accuracy, Precision, Recall, F1,AUC-ROC, Confidence

### 6. **best_hyperparameters.json**
   - Optimized hyperparameters for each model
   - Useful for reproducing results

## 📈 Understanding the Output

The script will print:

### 1. Data Loading
```
✓ Loaded X: (1000, 20) | y: (1000,)
📊 Class Distribution:
   Class 0 (No Risk): 500 (50.0%)
   Class 1 (Risk):    500 (50.0%)
```

### 2. Model Training Progress
```
1️⃣  Training Logistic Regression...
   ✓ Best CV AUC-ROC: 0.8234
   ✓ Best params: {'clf__C': 1, 'clf__penalty': 'l2'}
```

### 3. Final Comparison
```
MODEL COMPARISON SUMMARY
                      Accuracy  Precision  Recall  F1 Score  AUC-ROC
Logistic Regression     0.7892     0.7654  0.8123    0.7882   0.8234
SVM                     0.8123     0.7981  0.8234    0.8106   0.8567
Random Forest           0.8456     0.8312  0.8678    0.8491   0.8923
Stacked Ensemble        0.8567     0.8423  0.8789    0.8602   0.9012

🏆 BEST MODEL: Stacked Ensemble
```

## 🎯 Expected Improvements

Compared to basic training:

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| **AUC-ROC** | 89.0% | 91-95% | +2-6% |
| **Accuracy** | 79.0% | 83-87% | +4-8% |
| **F1-Score** | 79.2% | 83-88% | +4-9% |

## 🔧 Using the Stacked Ensemble in Production

If the stacked ensemble performs better, update your app to use it:

### Option 1: Quick Test
```python
# In app.py, add after line 44:
STACKED_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/stacked_ensemble_model.pkl')

# Update load_model() function to try loading stacked model first
if os.path.exists(STACKED_MODEL_PATH):
    model = joblib.load(STACKED_MODEL_PATH)
    logger.info("Stacked ensemble model loaded successfully.")
elif os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info("Random Forest model loaded successfully.")
```

### Option 2: Replace Current Model
```bash
# Backup current model
copy model\final_rf_model.pkl model\final_rf_model_backup.pkl

# Use stacked ensemble as primary model
copy model\stacked_ensemble_model.pkl model\final_rf_model.pkl
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: newtop20_features.csv"
**Solution**: Run feature engineering first:
```bash
python model/feature_engineering.py
```

### Issue: Training takes too long (>30 minutes)
**Solution**: Reduce hyperparameter grid size in the script:
```python
# Change these lines (around line 84-110):
rf_param_grid = {
    'clf__n_estimators': [200, 300],  # Reduced from [100, 200, 300, 500]
    'clf__max_depth': [20, None],      # Reduced from [10, 20, 30, None]
    # ... etc
}
```

### Issue: Low memory / Computer freezing
**Solution**: Reduce parallelization:
```python
# Change line 116:
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring='roc_auc', n_jobs=2, verbose=0)
# Changed n_jobs=-1 to n_jobs=2
```

## 📝 Next Steps After Training

1. **Review Results**: Check `evaluation_results.json` for performance metrics
2. **Compare Models**: See which model performs best on your data
3. **Update Production**: If stacked ensemble is better, deploy it
4. **Test in App**: Perform new assessments and verify predictions
5. **Monitor Performance**: Track real-world accuracy over time

## 💡 Tips for Even Better Performance

1. **Generate More Data**: Increase synthetic dataset from 1,000 to 3,000+ samples
2. **Feature Engineering**: Add interaction features (age × prior convictions)
3. **Collect Real Data**: Replace synthetic data with actual recidivism records
4. **Regular Retraining**: Retrain monthly with new real-world data

## 🆘 Need Help?

If you encounter issues:
1. Check the console output for error messages
2. Verify all required files exist in `data/` directory
3. Ensure virtual environment is activated
4. Check Python version (should be 3.8+)
