import pandas as pd
import numpy as np
import os
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# Start tracking memory
tracemalloc.start()

folder_path = "C:/Users/User/CIC_IoMT_2024"

# 1. LOAD FILES & SAMPLE TO PREVENT MEMORY ERROR
print("Step 1: Loading files and sampling 30,000 rows per file...")
files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f not in ['train_set.csv', 'val_set.csv', 'test_set.csv']]
df_list = []

for file in files:
    try:
        df = pd.read_csv(os.path.join(folder_path, file), low_memory=False, nrows=30000)
        if 'normal' in file.lower() or 'benign' in file.lower():
            df['Label'] = 0
        else:
            df['Label'] = 1
        df_list.append(df)
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

master_df = pd.concat(df_list, ignore_index=True)
master_df.columns = master_df.columns.str.strip()
print(f"Total rows sampled: {len(master_df)}, Attacks: {master_df['Label'].sum()}, Normal: {(master_df['Label'] == 0).sum()}")

# 2. STRATIFIED SPLIT
print("Step 2: Splitting data (Stratified)...")
train_df, temp_df = train_test_split(master_df, test_size=0.40, stratify=master_df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['Label'], random_state=42)

# 3. PREPARE FEATURES
print("Step 3: Preparing features...")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != 'Label']

X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train_df['Label']
X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_val = val_df['Label']
X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = test_df['Label']

# 4. SCALE FEATURES
print("Step 4: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. TRAIN MODEL 
print("Step 5: Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. PREDICT AND CALIBRATE
print("Step 6: Predicting and calibrating...")
probs_val = model.predict_proba(X_val_scaled)[:, 1]
probs_test = model.predict_proba(X_test_scaled)[:, 1]
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(probs_val, y_val)
probs_cal = iso_reg.predict(probs_test)

# 7. CALCULATE METRICS
print("Step 7: Calculating metrics...")
def calculate_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = y_true[in_bin].mean()
            avg_conf_in_bin = y_prob[in_bin].mean()
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece

auroc = roc_auc_score(y_test, probs_cal)
f1 = f1_score(y_test, (probs_cal > 0.5).astype(int))
brier = brier_score_loss(y_test, probs_cal)
ece = calculate_ece(y_test.values, probs_cal)

print("\n" + "="*40)
print("FINAL BENCHMARK METRICS (CALIBRATED)")
print("="*40)
print(f"AUROC       : {auroc:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"Brier Score : {brier:.4f}")
print(f"ECE         : {ece:.4f}")
print("="*40)

# 8. BOOTSTRAP CONFIDENCE INTERVALS
print("Step 8: Calculating Bootstrap 95% CIs...")
from sklearn.utils import resample
n_boot = 1000
auroc_boot, f1_boot, brier_boot, ece_boot = [], [], [], []
y_test_arr = y_test.values

for i in range(n_boot):
    idx = resample(np.arange(len(y_test_arr)), replace=True, n_samples=len(y_test_arr))
    if len(np.unique(y_test_arr[idx])) == 2:
        auroc_boot.append(roc_auc_score(y_test_arr[idx], probs_cal[idx]))
        f1_boot.append(f1_score(y_test_arr[idx], (probs_cal[idx] > 0.5).astype(int)))
        brier_boot.append(brier_score_loss(y_test_arr[idx], probs_cal[idx]))
        ece_boot.append(calculate_ece(y_test_arr[idx], probs_cal[idx]))

auroc_ci = np.percentile(auroc_boot, [2.5, 97.5])
f1_ci = np.percentile(f1_boot, [2.5, 97.5])
brier_ci = np.percentile(brier_boot, [2.5, 97.5])
ece_ci = np.percentile(ece_boot, [2.5, 97.5])

print("\n" + "="*40)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("="*40)
print(f"AUROC       : {auroc:.4f} [95% CI: {auroc_ci[0]:.4f} - {auroc_ci[1]:.4f}]")
print(f"F1-Score    : {f1:.4f} [95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
print(f"Brier Score : {brier:.4f} [95% CI: {brier_ci[0]:.4f} - {brier_ci[1]:.4f}]")
print(f"ECE         : {ece:.4f} [95% CI: {ece_ci[0]:.4f} - {ece_ci[1]:.4f}]")
print("="*40)

# 9. MEMORY USAGE
current, peak = tracemalloc.get_traced_memory()
print(f"\nPeak Memory Usage: {peak / 10**6:.2f} MB")
tracemalloc.stop()
print("SUCCESS! Copy these numbers into your paper.")