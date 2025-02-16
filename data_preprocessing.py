import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.impute import KNNImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    print("\n📌 Columns in dataset:", df.columns)

    # Ensure correct column for customer names
    customer_name_col = "Name"  
    if customer_name_col not in df.columns:
        raise KeyError(f"⚠ Column '{customer_name_col}' not found in dataset!")

    # ✅ **1. Check Missing Values Before Cleaning**
    missing_values_before = df.isnull().sum()
    print("\n🔍 Missing Values (Before Cleaning):\n", missing_values_before)

    # Fill missing values
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # ✅ **2. Check Missing Values After Cleaning**
    missing_values_after = df.isnull().sum()
    print("\n✅ Missing Values (After Cleaning):\n", missing_values_after)

    # ✅ **3. Count Duplicates Before Matching**
    duplicate_count_before = df.duplicated(subset=[customer_name_col]).sum()
    print(f"\n🔍 Duplicate Names Before Matching: {duplicate_count_before}")

    # ✅ **4. Optimized Fuzzy Matching with a Progress Counter**
    unique_names = df[customer_name_col].unique()
    name_map = {}
    
    print("\n🚀 Performing Fuzzy Matching on Names...")
    for index, name in enumerate(unique_names):
        best_match = process.extractOne(name, unique_names, score_cutoff=85)
        if best_match:
            name_map[name] = best_match[0]

        # ✅ **5. Progress Counter**
        if index % 500 == 0:  # Print progress every 500 names
            print(f"✅ Processed {index}/{len(unique_names)} names")

    df[customer_name_col] = df[customer_name_col].map(name_map)

    # ✅ **6. Count Duplicates After Matching**
    duplicate_count_after = df.duplicated(subset=[customer_name_col]).sum()
    print(f"\n✅ Duplicate Names After Matching: {duplicate_count_after}")

    return df

if __name__ == "__main__":
    # Load dataset
    data = load_data("large_crm.csv")

    # Process and clean data
    cleaned_data = clean_data(data)

    # Save cleaned data
    cleaned_data.to_csv("cleaned_crm.csv", index=False)
    print("\n🎉 Data cleaned and saved successfully!")
