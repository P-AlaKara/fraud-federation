import pandas as pd
import numpy as np

class DataHarmonizer:
    def __init__(self, schema_map=None, scaling_factor=1.0):
        """
        schema_map: Dictionary to rename local columns to global standard.
        scaling_factor: Float to convert local currency to standard (e.g., 0.01 for cents).
        """
        self.map = schema_map if schema_map else {}
        self.scale = scaling_factor
        
        # The GLOBAL STANDARD columns the model expects
        self.standard_cols = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
            'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud'
        ]

    def process(self, df):
        # 1. RENAME columns based on config
        if self.map:
            df = df.rename(columns=self.map)
        
        # 2. FILL MISSING columns (Zero-Masking)
        # Ensure all standard columns exist before we try to do math on them
        for col in self.standard_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # 3. SCALE Currency (Fix Units: e.g. Cents -> Shillings)
        # We check if columns exist before multiplying to avoid errors
        if self.scale != 1.0:
            cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            for c in cols_to_scale:
                 # We know they exist because of Step 2, but safety first
                if c in df.columns:
                    df[c] = df[c] * self.scale

        # 4. NORMALIZATION (Log1p Transformation) - NEW!
        # This squashes massive numbers (millions) into a range the AI can handle (0-20)
        numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in numeric_cols:
            # clip(lower=0) ensures we don't crash on negative numbers
            # np.log1p calculates log(x + 1) to handle 0s correctly
            df[col] = np.log1p(df[col].clip(lower=0))

        # 5. ENCODE 'type' (Transaction Type) to Numbers
        type_map = {'PAYMENT': 1, 'TRANSFER': 2, 'CASH_OUT': 3, 'CASH_IN': 4, 'DEBIT': 5}
        df['type'] = df['type'].map(type_map).fillna(0)

        # 6. SELECT final features for the model
        features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        
        # This creates the clean Tensor for MindSpore
        X = df[features].values.astype(np.float32)
        
        # 7. Extract Target (Labels)
        if 'isFraud' in df.columns:
            y = df['isFraud'].values.astype(np.int32)
        else:
            y = np.zeros(len(df), dtype=np.int32)
            
        return X, y