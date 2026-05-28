import pandas as pd
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv('D:/New folder (3)/project/data/raw/train.csv')
    print("Success loading train.csv")
    print(f"Shape: {df.shape}")
    print(f"Label groups: {df['label_group'].nunique()}")
    
    # Try stratify
    try:
        val_df, test_df = train_test_split(
            df,
            test_size=0.8,
            random_state=42,
            stratify=df['label_group'].values
        )
        print("Stratify split success!")
    except Exception as e:
        print(f"Stratify split failed: {e}")
        
except Exception as e:
    print(f"Error: {e}")
