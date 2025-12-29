import pandas as pd
import numpy as np

# 1. Load the original PaySim data
print("Loading PaySim Data...")
df = pd.read_csv('./experiment2/PS_20174392719_1491204439457_log.csv')

# 2. FILTER: PaySim documentation says fraud ONLY happens in 'TRANSFER' and 'CASH_OUT'
# We drop PAYMENT, CASH_IN, DEBIT because they are 100% safe (easy mode).
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

# 3. UNDERSAMPLING (The Magic Fix)
# We separate Fraud and Legit
frauds = df[df['isFraud'] == 1]
legits = df[df['isFraud'] == 0]

print(f"Total Frauds Available: {len(frauds)}")
print(f"Total Legits Available: {len(legits)}")

# We take ALL the frauds, and randomly sample the SAME amount of legits.
# This creates a 50/50 split (Balanced Dataset).
legits_balanced = legits.sample(n=len(frauds), random_state=42)
balanced_df = pd.concat([frauds, legits_balanced])

# Shuffle the rows so they aren't in order
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"New Balanced Dataset Size: {len(balanced_df)} rows")
print("Class Distribution: 50% Fraud, 50% Legit")

# 4. SPLIT into Banks (Non-IID Slices)

# Bank A (The "Rich" Bank): Gets 60% of the data (Standard Schema)
bank_a = balanced_df.iloc[: int(len(balanced_df)*0.6)].copy()

# Bank B (The "Weird" Bank): Gets 20% (Renamed Columns)
bank_b = balanced_df.iloc[int(len(balanced_df)*0.6) : int(len(balanced_df)*0.8)].copy()
bank_b = bank_b.rename(columns={'amount': 'txn_val', 'oldbalanceOrg': 'old_bal'})

# Bank C (The "Small" Bank): Gets 20% (Wrong Scale - Cents)
bank_c = balanced_df.iloc[int(len(balanced_df)*0.8) :].copy()
monetary_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
for col in monetary_cols:
    bank_c[col] = bank_c[col] * 100 

bank_c = bank_c.rename(columns={'amount': 'amount_cents'})

# 5. Save Files
bank_a.to_csv('./experiment2/bank_a_raw.csv', index=False)
bank_b.to_csv('./experiment2/bank_b_raw.csv', index=False)
bank_c.to_csv('./experiment2/bank_c_raw.csv', index=False)

print("\nDATA GENERATION COMPLETE")
print("Files saved to ./experiment2/")