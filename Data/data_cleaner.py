"""
Credit Dataset Cleaner
======================
Detects and fixes common data quality issues in the credit scoring dataset.
Generates a cleaned CSV.
"""

import pandas as pd
import numpy as np
import re
from copy import deepcopy

# ── 1. LOAD ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/train.csv", low_memory=False)
original_df = df.copy()
report = []          # will collect human-readable log entries
change_counts = {}   # column → number of cells changed

def log(msg):
    print(msg)
    report.append(msg)

log(f"\n{'='*60}")
log(f"  CREDIT DATA CLEANING REPORT")
log(f"{'='*60}")
log(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")


# ── 2. HELPERS ───────────────────────────────────────────────────────────────
def strip_trailing_junk(series):
    """Remove trailing underscores / non-numeric suffixes from numeric strings."""
    return series.astype(str).str.replace(r'[_\s]+$', '', regex=True)

def is_corrupted_string(val, pattern=r'^[#@!%\$\^&\*]+|__'):
    """True if the string looks like garbage (special-char noise or __ prefix)."""
    return bool(re.search(pattern, str(val)))

def count_changes(col_before, col_after):
    return (col_before.astype(str) != col_after.astype(str)).sum()


# ── 3. GENERIC PLACEHOLDER → NaN ─────────────────────────────────────────────
log("\n[1] Converting placeholder markers to NaN")
placeholders = ['_', 'NA', 'NM', '_______', '__10000__', 'nan', 'NaN', '']

for col in df.columns:
    before = df[col].copy()
    df[col] = df[col].replace(placeholders, np.nan)
    # Also catch strings that are only underscores
    mask = df[col].astype(str).str.fullmatch(r'_+')
    df.loc[mask, col] = np.nan
    changed = count_changes(before, df[col])
    if changed:
        log(f"   {col}: {changed} placeholder(s) → NaN")
        change_counts[col] = change_counts.get(col, 0) + changed


# ── 4. CORRUPTED STRINGS → NaN ───────────────────────────────────────────────
log("\n[2] Removing corrupted / garbage strings")
string_cols = ['SSN', 'Payment_Behaviour', 'Occupation', 'Credit_Mix']
corrupt_pattern = r'[#@!%\$\^&\*]|^__'

for col in string_cols:
    if col not in df.columns:
        continue
    before = df[col].copy()
    mask = df[col].astype(str).str.contains(corrupt_pattern, regex=True, na=False)
    df.loc[mask, col] = np.nan
    changed = mask.sum()
    if changed:
        log(f"   {col}: {changed} corrupted value(s) → NaN")
        change_counts[col] = change_counts.get(col, 0) + changed


# ── 5. STRIP TRAILING UNDERSCORES FROM NUMERIC COLUMNS ───────────────────────
log("\n[3] Stripping trailing underscores from numeric fields")
numeric_candidates = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance'
]

for col in numeric_candidates:
    if col not in df.columns:
        continue
    before = df[col].copy()
    cleaned = df[col].astype(str).str.replace(r'_+$', '', regex=True)
    df[col] = pd.to_numeric(cleaned, errors='coerce')
    changed = count_changes(before, df[col])
    if changed:
        log(f"   {col}: {changed} value(s) stripped/coerced")
        change_counts[col] = change_counts.get(col, 0) + changed

# Make the rest of the numeric cols proper numeric too
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# ── 6. DOMAIN-SPECIFIC VALIDATION ────────────────────────────────────────────
log("\n[4] Domain validation – clamping / nullifying out-of-range values")

def nullify_out_of_range(col, low=None, high=None, label=""):
    if col not in df.columns:
        return
    before = df[col].copy()
    mask = pd.Series(False, index=df.index)
    if low is not None:
        mask |= df[col] < low
    if high is not None:
        mask |= df[col] > high
    df.loc[mask, col] = np.nan
    changed = mask.sum()
    if changed:
        log(f"   {col}: {changed} value(s) out of range {label} → NaN")
        change_counts[col] = change_counts.get(col, 0) + changed

# Age: must be 18–100
nullify_out_of_range('Age', low=18, high=100, label="[18, 100]")

# Interest Rate: must be > 0 and realistic (≤ 60%)
nullify_out_of_range('Interest_Rate', low=0.00, high=60, label="(0, 60]")

# Num_Credit_Card: realistic max ~20
nullify_out_of_range('Num_Credit_Card', low=0, high=20, label="[0, 20]")

# Num_of_Loan: 0–20
nullify_out_of_range('Num_of_Loan', low=0, high=20, label="[0, 20]")

# Delay_from_due_date: can be negative (paid early) but not extreme
nullify_out_of_range('Delay_from_due_date', low=-30, high=365, label="[-30, 365]")

# Num_of_Delayed_Payment: 0–100
nullify_out_of_range('Num_of_Delayed_Payment', low=0, high=100, label="[0, 100]")

# Annual_Income: > 0 and < 10M
nullify_out_of_range('Annual_Income', low=1, high=10_000_000, label="[1, 10M]")

# Monthly_Inhand_Salary: > 0 and < 1M
nullify_out_of_range('Monthly_Inhand_Salary', low=1, high=1_000_000, label="[1, 1M]")

# Amount_invested_monthly: ≥ 0
nullify_out_of_range('Amount_invested_monthly', low=0, label="[0, ∞)")

# Credit_Utilization_Ratio: 0–100 %
nullify_out_of_range('Credit_Utilization_Ratio', low=0, high=100, label="[0, 100]")


# ── 7. PAYMENT_OF_MIN_AMOUNT → YES/NO ────────────────────────────────────────
log("\n[5] Standardising Payment_of_Min_Amount (NM → NaN)")
if 'Payment_of_Min_Amount' in df.columns:
    before = df['Payment_of_Min_Amount'].copy()
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace({'NM': np.nan})
    valid_values = {'Yes', 'No'}
    mask = ~df['Payment_of_Min_Amount'].isin(valid_values) & df['Payment_of_Min_Amount'].notna()
    df.loc[mask, 'Payment_of_Min_Amount'] = np.nan
    changed = count_changes(before, df['Payment_of_Min_Amount'])
    if changed:
        log(f"   Payment_of_Min_Amount: {changed} invalid value(s) → NaN")
        change_counts['Payment_of_Min_Amount'] = change_counts.get('Payment_of_Min_Amount', 0) + changed


# ── 8. IMPUTATION (per-customer medians for numeric cols) ────────────────────
log("\n[6] Imputing NaN values with per-customer median (numeric) / mode (categorical)")

# Per-customer numeric imputation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    null_count = df[col].isna().sum()
    if null_count == 0:
        continue
    if 'Customer_ID' in df.columns:
        df[col] = df.groupby('Customer_ID')[col].transform(
            lambda x: x.fillna(x.median())
        )
    remaining = df[col].isna().sum()
    # Global median fallback
    if remaining > 0:
        df[col] = df[col].fillna(df[col].median())
    filled = null_count - df[col].isna().sum()
    if filled > 0:
        log(f"   {col}: {filled} NaN(s) filled via per-customer median")

# Per-customer categorical imputation
cat_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
for col in cat_cols:
    if col not in df.columns:
        continue
    null_count = df[col].isna().sum()
    if null_count == 0:
        continue
    if 'Customer_ID' in df.columns:
        df[col] = df.groupby('Customer_ID')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
        )
    remaining = df[col].isna().sum()
    if remaining > 0:
        global_mode = df[col].mode()
        df[col] = df[col].fillna(global_mode[0] if not global_mode.empty else np.nan)
    filled = null_count - df[col].isna().sum()
    if filled > 0:
        log(f"   {col}: {filled} NaN(s) filled via per-customer mode")


# ── FURTHER CHANGES ─────────────────────────────────────────────────────────
# This is used to make into numerical values the Credit_History_Age column, which is in the format "X Years and Y Months".
def parse_credit_history_simple(text):
    if pd.isna(text):
        return np.nan
    
    try:
        # Split by "Years and"
        parts = str(text).split('Years and')
        years = int(parts[0].strip())
        
        # Extract months
        months = int(parts[1].replace('Months', '').strip())
        
        return years + (months / 12)
    except:
        return np.nan

df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_credit_history_simple)

# Use to fill the NaN values on the column Credit_History_Age by backfilling and subtracting 1 month for each month back.
def bfill_subtract_monthly(group):
    """
    For leading NaNs, backfill by subtracting 1/12 from the first valid value
    """
    # Find first non-null value
    first_valid_idx = group.first_valid_index()
    
    if first_valid_idx is None:
        return group  # All NaN, can't fill
    
    # Get position of first valid value
    first_value = group.loc[first_valid_idx]
    first_position = group.index.get_loc(first_valid_idx)
    
    # Fill all positions before first valid value
    for i in range(first_position - 1, -1, -1):
        idx = group.index[i]
        if pd.isna(group.loc[idx]):
            months_back = first_position - i
            group.loc[idx] = first_value - (months_back / 12)
    
    return group

# Filling NaN values in a forward way by customer by adding 1 month.
df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(
    lambda x: x.interpolate(method='linear')
)

# Filling NaN values in a backward way by customer by subtracting 1 month.
df['Credit_History_Age'] = df.groupby('Customer_ID', group_keys=False)['Credit_History_Age'].apply(
    bfill_subtract_monthly
)

# --- FILL BLANK SPACES ---
# Step 1: fill from same Customer_ID
df['Type_of_Loan'] = df.groupby('Customer_ID')['Type_of_Loan'].transform(lambda x: x.ffill().bfill())

# Step 2: remaining blanks → "Not Specified"
df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Not Specified')

# --- PERSISTENT ERRORS ---
cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan']

for c in cols:
    df[c] = (
        df.groupby('Customer_ID')[c]
          .transform(lambda x: x.value_counts().idxmax())
    )
    
# --- STRAY APOSTROPHES DELETION ---
def delete_stray_apostrophes(column):
    return (
        pd.to_numeric(
            column.astype(str)
                  .str.strip()
                  .str.lstrip("'"),   # removes leading apostrophes only
            errors="coerce"
        )
    )
    
cols = [
    'Monthly_Inhand_Salary',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance',
    'Credit_History_Age'
]

df[cols] = df[cols].apply(delete_stray_apostrophes)

# ── 9. DUPLICATE ROWS ─────────────────────────────────────────────────────────
log("\n[7] Checking for duplicate rows")
dupes = df.duplicated().sum()
if dupes:
    df.drop_duplicates(inplace=True)
    log(f"   Removed {dupes} duplicate row(s)")
else:
    log("   No duplicates found")


# ── 10. SUMMARY ──────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log(f"  SUMMARY")
log(f"{'='*60}")
log(f"Original rows : {len(original_df):,}")
log(f"Cleaned rows  : {len(df):,}")
log(f"Remaining NaN : {df.isna().sum().sum():,}")
log(f"\nColumns modified:")
for col, cnt in sorted(change_counts.items(), key=lambda x: -x[1]):
    log(f"   {col:40s} {cnt:>6} cell(s) changed")
log(f"{'='*60}\n")

for col in ['Name', 'SSN']:
    before = df[col].isna().sum()
    df[col] = df.groupby('Customer_ID')[col].transform(lambda x: x.ffill().bfill())
    after = df[col].isna().sum()
    print(f"{col}: {before} blanks filled → {after} remaining")


# ── 11. SAVE ─────────────────────────────────────────────────────────────────
df.to_csv("Data/clean_train.csv", index=False)