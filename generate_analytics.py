"""
Comprehensive Data Analytics Report Generator
Analyzes the training dataset and outputs insights to a text file.
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('data/syntheticdataset.csv')

# Create recidivism target from 'Reoffend Time (mos)' - if has reoffend time, they reoffended
if 'Reoffend Time (mos)' in df.columns:
    df['Recidivism'] = (df['Reoffend Time (mos)'].notna() & (df['Reoffend Time (mos)'] > 0)).astype(int)
    has_target = True
elif 'Time to Reoffend (mos)' in df.columns:
    df['Recidivism'] = (df['Time to Reoffend (mos)'].notna() & (df['Time to Reoffend (mos)'] > 0)).astype(int)
    has_target = True
else:
    has_target = False

# Open output file
with open('data/data_analytics_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("       RECIDIVISM PREDICTION SYSTEM - DATA ANALYTICS REPORT\n")
    f.write(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    # ====================
    # 1. DATASET OVERVIEW
    # ====================
    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 40 + "\n")
    f.write(f"   Total Records: {len(df)}\n")
    f.write(f"   Total Features: {len(df.columns)}\n")
    
    # Target variable analysis
    if has_target:
        recid_counts = df['Recidivism'].value_counts()
        recid_rate = df['Recidivism'].mean() * 100
        f.write(f"\n   Target Variable (Recidivism):\n")
        f.write(f"      Reoffended (1): {recid_counts.get(1, 0)} ({recid_counts.get(1, 0)/len(df)*100:.1f}%)\n")
        f.write(f"      Did Not Reoffend (0): {recid_counts.get(0, 0)} ({recid_counts.get(0, 0)/len(df)*100:.1f}%)\n")
        f.write(f"\n   *** OVERALL RECIDIVISM RATE: {recid_rate:.1f}% ***\n")
    
    # ====================
    # 2. DEMOGRAPHIC ANALYSIS
    # ====================
    f.write("\n\n2. DEMOGRAPHIC BREAKDOWN\n")
    f.write("-" * 40 + "\n")
    
    # Gender
    if 'Gender' in df.columns:
        f.write("\n   A. Gender Distribution:\n")
        gender_counts = df['Gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = count / len(df) * 100
            f.write(f"      {gender}: {count} ({pct:.1f}%)\n")
        
        # Recidivism by Gender
        if has_target:
            f.write("\n      ** Recidivism Rate by Gender: **\n")
            gender_recid = df.groupby('Gender')['Recidivism'].agg(['sum', 'count', 'mean'])
            for gender in gender_recid.index:
                rate = gender_recid.loc[gender, 'mean'] * 100
                reoffended = int(gender_recid.loc[gender, 'sum'])
                total = int(gender_recid.loc[gender, 'count'])
                f.write(f"      {gender}: {reoffended}/{total} reoffended ({rate:.1f}%)\n")
    
    # Civil Status
    if 'Civil Status' in df.columns:
        f.write("\n   B. Civil Status Distribution:\n")
        cs_counts = df['Civil Status'].value_counts()
        for status, count in cs_counts.items():
            pct = count / len(df) * 100
            f.write(f"      {status}: {count} ({pct:.1f}%)\n")
        
        # Recidivism by Civil Status
        if has_target:
            f.write("\n      ** Recidivism Rate by Civil Status (Sorted by Risk): **\n")
            cs_recid = df.groupby('Civil Status')['Recidivism'].agg(['sum', 'count', 'mean'])
            cs_recid = cs_recid.sort_values('mean', ascending=False)
            for status in cs_recid.index:
                rate = cs_recid.loc[status, 'mean'] * 100
                reoffended = int(cs_recid.loc[status, 'sum'])
                total = int(cs_recid.loc[status, 'count'])
                f.write(f"      {status}: {reoffended}/{total} reoffended ({rate:.1f}%)\n")
    
    # ====================
    # 3. CROSS-TABULATION: WIDOWED BY GENDER (Expert's Question)
    # ====================
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("3. DEEP DIVE: WIDOWED STATUS BY GENDER (EXPERT'S QUESTION)\n")
    f.write("=" * 80 + "\n")
    f.write("   Question: Why is 'Widowed' a top predictor? Is it male or female?\n\n")
    
    if 'Civil Status' in df.columns and 'Gender' in df.columns:
        widowed = df[df['Civil Status'] == 'Widowed']
        f.write(f"   Total Widowed in Dataset: {len(widowed)} ({len(widowed)/len(df)*100:.1f}% of all)\n\n")
        
        if len(widowed) > 0:
            f.write("   A. Widowed Population by Gender:\n")
            widowed_gender = widowed['Gender'].value_counts()
            for gender, count in widowed_gender.items():
                pct = count / len(widowed) * 100
                f.write(f"      {gender}: {count} ({pct:.1f}% of widowed)\n")
            
            if has_target:
                f.write("\n   B. *** RECIDIVISM RATE FOR WIDOWED BY GENDER: ***\n")
                widowed_recid = widowed.groupby('Gender')['Recidivism'].agg(['sum', 'count', 'mean'])
                for gender in widowed_recid.index:
                    rate = widowed_recid.loc[gender, 'mean'] * 100
                    reoffended = int(widowed_recid.loc[gender, 'sum'])
                    total = int(widowed_recid.loc[gender, 'count'])
                    f.write(f"      Widowed {gender}: {reoffended}/{total} reoffended ({rate:.1f}%)\n")
                
                # Compare to non-widowed
                f.write("\n   C. Comparison with Non-Widowed (Same Gender):\n")
                not_widowed = df[df['Civil Status'] != 'Widowed']
                for gender in df['Gender'].unique():
                    nw_subset = not_widowed[not_widowed['Gender'] == gender]
                    w_subset = widowed[widowed['Gender'] == gender]
                    if len(nw_subset) > 0 and len(w_subset) > 0:
                        nw_rate = nw_subset['Recidivism'].mean() * 100
                        w_rate = w_subset['Recidivism'].mean() * 100
                        diff = w_rate - nw_rate
                        f.write(f"      {gender}:\n")
                        f.write(f"         Non-Widowed: {nw_rate:.1f}% recidivism\n")
                        f.write(f"         Widowed: {w_rate:.1f}% recidivism\n")
                        f.write(f"         Difference: {'+' if diff > 0 else ''}{diff:.1f}%\n")
                
                f.write("\n   D. CONCLUSION:\n")
                male_widowed = widowed[widowed['Gender'] == 'Male']
                female_widowed = widowed[widowed['Gender'] == 'Female']
                if len(male_widowed) > 0 and len(female_widowed) > 0:
                    m_rate = male_widowed['Recidivism'].mean() * 100
                    f_rate = female_widowed['Recidivism'].mean() * 100
                    if m_rate > f_rate:
                        f.write(f"      Widowed MALES have HIGHER recidivism ({m_rate:.1f}%) than\n")
                        f.write(f"      Widowed FEMALES ({f_rate:.1f}%).\n")
                    else:
                        f.write(f"      Widowed FEMALES have HIGHER recidivism ({f_rate:.1f}%) than\n")
                        f.write(f"      Widowed MALES ({m_rate:.1f}%).\n")
    
    # ====================
    # 4. AGE ANALYSIS
    # ====================
    f.write("\n\n4. AGE ANALYSIS\n")
    f.write("-" * 40 + "\n")
    
    if 'Age' in df.columns:
        f.write(f"   Age Range: {df['Age'].min()} - {df['Age'].max()} years\n")
        f.write(f"   Mean Age: {df['Age'].mean():.1f} years\n")
        f.write(f"   Median Age: {df['Age'].median():.1f} years\n")
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                  labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        f.write("\n   Age Group Distribution:\n")
        age_counts = df['Age_Group'].value_counts().sort_index()
        for age_grp, count in age_counts.items():
            pct = count / len(df) * 100
            f.write(f"      {age_grp}: {count} ({pct:.1f}%)\n")
        
        if has_target:
            f.write("\n   ** Recidivism Rate by Age Group: **\n")
            age_recid = df.groupby('Age_Group')['Recidivism'].agg(['sum', 'count', 'mean'])
            for age_grp in age_recid.index:
                rate = age_recid.loc[age_grp, 'mean'] * 100
                reoff = int(age_recid.loc[age_grp, 'sum'])
                total = int(age_recid.loc[age_grp, 'count'])
                f.write(f"      {age_grp}: {reoff}/{total} ({rate:.1f}%)\n")
    
    # ====================
    # 5. OFFENSE ANALYSIS
    # ====================
    f.write("\n\n5. OFFENSE TYPE ANALYSIS\n")
    f.write("-" * 40 + "\n")
    
    offense_col = None
    for col in ['Type of Current Offense', 'Offense Type', 'Current Offense']:
        if col in df.columns:
            offense_col = col
            break
    
    if offense_col:
        f.write(f"   Offense Type Distribution:\n")
        offense_counts = df[offense_col].value_counts()
        for offense, count in offense_counts.items():
            pct = count / len(df) * 100
            f.write(f"      {offense}: {count} ({pct:.1f}%)\n")
        
        if has_target:
            f.write("\n   ** Recidivism Rate by Offense Type (Sorted by Risk): **\n")
            off_recid = df.groupby(offense_col)['Recidivism'].agg(['sum', 'count', 'mean'])
            off_recid = off_recid.sort_values('mean', ascending=False)
            for offense in off_recid.index:
                rate = off_recid.loc[offense, 'mean'] * 100
                reoffended = int(off_recid.loc[offense, 'sum'])
                total = int(off_recid.loc[offense, 'count'])
                f.write(f"      {offense}: {reoffended}/{total} ({rate:.1f}%)\n")
    
    # ====================
    # 6. RISK FACTOR ANALYSIS
    # ====================
    f.write("\n\n6. KEY RISK FACTORS ANALYSIS\n")
    f.write("-" * 40 + "\n")
    
    risk_factors = [
        ('Gang Affiliation', 'Gang Affiliation'),
        ('Homelessness', 'Homelessness'),
        ('Vocational Training', 'Vocational Training'),
        ('Jail Behavior Rating', 'Jail Behavior'),
        ('Rehab Attitude', 'Rehab Attitude'),
    ]
    
    for col, label in risk_factors:
        if col in df.columns:
            f.write(f"\n   {label}:\n")
            val_counts = df[col].value_counts()
            for val, count in val_counts.items():
                pct = count / len(df) * 100
                f.write(f"      {val}: {count} ({pct:.1f}%)\n")
            
            if has_target:
                f.write(f"      ** Recidivism Rate: **\n")
                factor_recid = df.groupby(col)['Recidivism'].agg(['sum', 'count', 'mean'])
                factor_recid = factor_recid.sort_values('mean', ascending=False)
                for val in factor_recid.index:
                    rate = factor_recid.loc[val, 'mean'] * 100
                    reoff = int(factor_recid.loc[val, 'sum'])
                    total = int(factor_recid.loc[val, 'count'])
                    f.write(f"         {val}: {reoff}/{total} ({rate:.1f}%)\n")
    
    # ====================
    # 7. TOP PREDICTORS DETAILED BREAKDOWN
    # ====================
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("7. TOP 5 SHAP PREDICTORS - DETAILED BREAKDOWN\n")
    f.write("=" * 80 + "\n")
    
    # Load SHAP importance
    try:
        shap_df = pd.read_csv('data/shap_importance.csv')
        top_5 = shap_df.head(5)
        
        for idx, row in top_5.iterrows():
            feature = row['Feature']
            importance = row['SHAP_Importance'] * 100
            f.write(f"\n   {idx+1}. {feature} (SHAP Importance: {importance:.2f}%)\n")
            f.write(f"      ----------------------------------------\n")
            
            # Try to find the base column and value
            if '_' in feature:
                parts = feature.rsplit('_', 1)
                base_col = parts[0]
                value = parts[1] if len(parts) > 1 else None
                
                # Check if base column exists
                if base_col in df.columns and value:
                    subset = df[df[base_col] == value]
                    total = len(subset)
                    f.write(f"      Total with '{value}': {total} ({total/len(df)*100:.1f}% of dataset)\n")
                    
                    if has_target and len(subset) > 0:
                        rate = subset['Recidivism'].mean() * 100
                        reoff = int(subset['Recidivism'].sum())
                        f.write(f"      Recidivism: {reoff}/{total} ({rate:.1f}%)\n")
                        
                        # Breakdown by gender
                        if 'Gender' in df.columns:
                            f.write(f"      By Gender:\n")
                            gender_recid = subset.groupby('Gender')['Recidivism'].agg(['sum', 'count', 'mean'])
                            for gender in gender_recid.index:
                                g_rate = gender_recid.loc[gender, 'mean'] * 100
                                g_count = int(gender_recid.loc[gender, 'count'])
                                g_reoff = int(gender_recid.loc[gender, 'sum'])
                                f.write(f"         {gender}: {g_reoff}/{g_count} ({g_rate:.1f}%)\n")
            else:
                # Numeric feature
                if feature in df.columns:
                    f.write(f"      Min: {df[feature].min()}, Max: {df[feature].max()}\n")
                    f.write(f"      Mean: {df[feature].mean():.2f}, Median: {df[feature].median():.2f}\n")
                    
                    if has_target:
                        # Correlation with target
                        corr = df[feature].corr(df['Recidivism'])
                        f.write(f"      Correlation with Recidivism: {corr:.3f}\n")
                        
                        # High vs Low
                        median = df[feature].median()
                        high = df[df[feature] > median]
                        low = df[df[feature] <= median]
                        if len(high) > 0 and len(low) > 0:
                            high_rate = high['Recidivism'].mean() * 100
                            low_rate = low['Recidivism'].mean() * 100
                            f.write(f"      Above Median ({median}): {high_rate:.1f}% recidivism\n")
                            f.write(f"      Below Median: {low_rate:.1f}% recidivism\n")
    except Exception as e:
        f.write(f"   Could not load SHAP data: {e}\n")
    
    # ====================
    # 8. SUMMARY STATISTICS
    # ====================
    f.write("\n\n8. NUMERICAL FEATURES SUMMARY\n")
    f.write("-" * 40 + "\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Recidivism' in numeric_cols:
        numeric_cols.remove('Recidivism')
    
    for col in numeric_cols[:10]:  # First 10 numeric columns
        f.write(f"\n   {col}:\n")
        f.write(f"      Min: {df[col].min()}, Max: {df[col].max()}\n")
        f.write(f"      Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}\n")
        if has_target:
            corr = df[col].corr(df['Recidivism'])
            f.write(f"      Correlation with Recidivism: {corr:.3f}\n")
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("                         END OF REPORT\n")
    f.write("=" * 80 + "\n")

print("Analytics report generated: data/data_analytics_report.txt")
