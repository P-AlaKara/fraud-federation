import pandas as pd
import numpy as np

# 1. Load the PaySim Dataset (Simulating a National Switch)
# Download from Kaggle first: https://www.kaggle.com/datasets/ealaxi/paysim1
print("Loading PaySim Data...")
# Make sure you unzip the kaggle file to get the .csv
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Reduce size for rapid testing (Use 100k rows instead of 6 million)
df = df.sample(n=100000, random_state=42)

# 2. Simulate 3 Different Institutions (The "Silos")
# We split based on transaction type to create "Non-IID" data (Realistic Skew)

# Bank A (e.g., M-Pesa): Mostly Cash-In/Cash-Out
bank_a = df[df['type'].isin(['CASH_IN', 'CASH_OUT'])].copy()

# Bank B (e.g., Equity): Mostly Transfers/Payments
bank_b = df[df['type'].isin(['PAYMENT', 'TRANSFER'])].copy()

# Bank C (e.g., Small Sacco): Gets a random 10% slice of whatever is left
# This simulates a small player with very little data
bank_c = df[~df.index.isin(bank_a.index) & ~df.index.isin(bank_b.index)].copy()

print(f"Bank A Rows: {len(bank_a)} (Specialty: Cash Operations)")
print(f"Bank B Rows: {len(bank_b)} (Specialty: Transfers)")
print(f"Bank C Rows: {len(bank_c)} (Specialty: Mixed)")

# 3. Create the "Real World" Problems (The "Adapter" Challenge)
# We intentionally BREAK the data so you can fix it later with your Adapter.

# Break Bank B: Rename columns and delete 'isFlaggedFraud'
# This simulates a bank that uses different column names
bank_b = bank_b.rename(columns={'amount': 'txn_val', 'oldbalanceOrg': 'old_bal'})
bank_b = bank_b.drop(columns=['isFlaggedFraud'])

# Break Bank C: Convert Amount to Cents (Scale Mismatch)
# This simulates a bank that stores values differently
bank_c['amount'] = bank_c['amount'] * 100 
bank_c = bank_c.rename(columns={'amount': 'amount_cents'})

# 4. Save the "Siloed" Files
bank_a.to_csv('bank_a_raw.csv', index=False)
bank_b.to_csv('bank_b_raw.csv', index=False)
bank_c.to_csv('bank_c_raw.csv', index=False)

print("\nData Split Complete.")
print("Challenge 1 Created: Bank B has wrong column names.")
print("Challenge 2 Created: Bank C has wrong number scale.")
print("Ready for Step 2: Building the MindSpore Fraud-Net.")