"""
Data preparation script for Fertility Risk Prediction
Downloads, cleans, and partitions data for federated learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def load_and_preprocess_ahs_data(filepath_or_dir):
    """
    Load and preprocess AHS Woman dataset from Kaggle
    Can handle single file or directory with multiple CSVs
    Note: AHS datasets use pipe (|) as delimiter
    """
    print("Loading AHS dataset...")
    
    # Check if it's a directory or single file
    if os.path.isdir(filepath_or_dir):
        # Load and merge all CSV files in directory
        csv_files = [f for f in os.listdir(filepath_or_dir) 
                     if f.endswith('.csv') and f.startswith('AHS_Woman')]
        print(f"Found {len(csv_files)} AHS CSV files")
        
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(filepath_or_dir, csv_file)
            print(f"  Loading {csv_file}...")
            try:
                # AHS files use pipe delimiter
                df_temp = pd.read_csv(file_path, delimiter='|', low_memory=False, on_bad_lines='skip')
                
                if len(df_temp) > 0 and len(df_temp.columns) > 1:
                    dfs.append(df_temp)
                    print(f"    ✓ Loaded {len(df_temp)} rows, {len(df_temp.columns)} columns")
                else:
                    print(f"    ✗ Skipping - insufficient data")
                    
            except Exception as e:
                print(f"    ✗ Error: {str(e)[:100]}")
                continue
        
        if not dfs:
            raise ValueError("No valid CSV files could be loaded!")
        
        # Merge all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Merged {len(csv_files)} state datasets into {len(df)} total rows, {len(df.columns)} columns")
    else:
        # Single file - use pipe delimiter
        df = pd.read_csv(filepath_or_dir, delimiter='|', low_memory=False, on_bad_lines='skip')
        print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Print dataset info
    print(f"\nDataset has {len(df.columns)} columns")
    print(f"Sample columns: {list(df.columns[:10])}")
    
    # Select relevant fertility-related features based on AHS data dictionary
    # These are confirmed columns from the AHS dataset
    fertility_features = [
        'age', 'marital_status', 'delivered_any_baby',
        'born_alive_female', 'born_alive_male', 'born_alive_total',
        'surviving_female', 'surviving_male', 'surviving_total',
        'mother_age_when_baby_was_born', 'outcome_pregnancy',
        'is_currently_pregnant', 'pregnant_month', 'is_anc_registered',
        'is_tubectomy', 'is_vasectomy', 'is_copper_t',
        'is_pills_daily', 'is_condom', 'is_contraceptive',
        'want_more_childern', 'religion', 'social_group_code',
        'highest_qualification', 'rural', 'house_structure',
        'drinking_water_source', 'household_have_electricity',
        'ever_conceived', 'no_of_times_conceived', 'age_at_first_conception',
        'aware_abt_rti', 'aware_abt_hiv', 'aware_of_the_danger_signs'
    ]
    
    # Filter columns that exist in the dataset
    available_features = [col for col in fertility_features if col in df.columns]
    
    if len(available_features) == 0:
        print("Warning: Predefined features not found, using all numeric columns")
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()[:30]
    
    print(f"\nSelected {len(available_features)} features for modeling")
    print(f"Features: {available_features[:15]}...")
    
    # CRITICAL: Convert all features to numeric FIRST, before creating target
    print("\nCleaning and converting data types...")
    for col in available_features:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with median for numeric columns
    for col in available_features:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
    
    print(f"After cleaning: {len(df)} rows")
    
    # Create fertility risk target variable based on multiple risk factors
    print("\nCreating fertility risk target variable...")
    
    # Define risk factors based on medical literature
    df['fertility_risk'] = 0  # Initialize with low risk
    
    # Risk Factor 1: High parity (>4 children)
    if 'born_alive_total' in df.columns:
        df['fertility_risk'] += (df['born_alive_total'] > 4).astype(int)
        print("  ✓ Added high parity risk factor")
    
    # Risk Factor 2: Very young or old maternal age (<18 or >35)
    if 'age' in df.columns:
        df['fertility_risk'] += ((df['age'] < 18) | (df['age'] > 35)).astype(int)
        print("  ✓ Added maternal age risk factor")
    
    # Risk Factor 3: Short interpregnancy interval (mother age at first birth <18)
    if 'mother_age_when_baby_was_born' in df.columns:
        df['fertility_risk'] += (df['mother_age_when_baby_was_born'] < 18).astype(int)
        print("  ✓ Added young first birth risk factor")
    
    # Risk Factor 4: No contraceptive use despite not wanting more children
    if 'want_more_childern' in df.columns and 'is_contraceptive' in df.columns:
        df['fertility_risk'] += ((df['want_more_childern'] == 2) & 
                                  (df['is_contraceptive'] == 2)).astype(int)
        print("  ✓ Added contraceptive use risk factor")
    
    # Risk Factor 5: Pregnancy without ANC registration
    if 'is_currently_pregnant' in df.columns and 'is_anc_registered' in df.columns:
        df['fertility_risk'] += ((df['is_currently_pregnant'] == 1) & 
                                  (df['is_anc_registered'] == 2)).astype(int)
        print("  ✓ Added ANC registration risk factor")
    
    # Convert to binary: High risk if 2+ risk factors present
    df['fertility_risk'] = (df['fertility_risk'] >= 2).astype(int)
    
    target = 'fertility_risk'
    
    # Print target distribution
    risk_counts = df[target].value_counts()
    print(f"\nTarget variable created:")
    print(f"  Low Risk (0): {risk_counts.get(0, 0):,} ({risk_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  High Risk (1): {risk_counts.get(1, 0):,} ({risk_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Extract features and target
    X = df[available_features].values
    y = df[target].values
    
    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y, available_features

def create_federated_partitions(X, y, num_clients=5, iid=True):
    """
    Partition data across clients (hospitals)
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of federated clients (hospitals)
        iid: If True, data is distributed uniformly (IID)
             If False, creates non-IID distribution (more realistic)
    """
    print(f"\nCreating {num_clients} federated partitions (IID={iid})...")
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if iid:
        # Shuffle and split evenly
        np.random.shuffle(indices)
        partitions = np.array_split(indices, num_clients)
    else:
        # Non-IID: Sort by label and create imbalanced partitions
        sorted_indices = indices[np.argsort(y)]
        partitions = []
        
        # Create skewed partitions
        partition_sizes = np.random.dirichlet(np.ones(num_clients) * 0.5, 1)[0]
        partition_sizes = (partition_sizes * n_samples).astype(int)
        
        start_idx = 0
        for size in partition_sizes[:-1]:
            partitions.append(sorted_indices[start_idx:start_idx + size])
            start_idx += size
        partitions.append(sorted_indices[start_idx:])
    
    return partitions

def save_federated_data(X, y, feature_names, partitions, output_dir='data/processed_dp'):
    """
    Save partitioned data for federated learning
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save global test set (20% of data)
    print("\nCreating train/test split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"  ✓ Saved global test set: {len(X_test):,} samples")
    
    # Recalculate partitions on remaining 80%
    new_partitions = create_federated_partitions(
        X_temp, y_temp, num_clients=len(partitions), iid=True
    )
    
    # Save client partitions
    print("\nSaving client partitions...")
    for i, partition_indices in enumerate(new_partitions):
        X_client = X_temp[partition_indices]
        y_client = y_temp[partition_indices]
        
        # Split into train/val for each client
        X_train, X_val, y_train, y_val = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )
        
        client_dir = os.path.join(output_dir, f'client_{i}')
        os.makedirs(client_dir, exist_ok=True)
        
        np.save(os.path.join(client_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(client_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(client_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(client_dir, 'y_val.npy'), y_val)
        
        print(f"  Client {i}: {len(X_train):,} train, {len(X_val):,} val samples")
    
    # Save feature names and metadata
    metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'num_classes': len(np.unique(y)),
        'num_clients': len(partitions)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Data saved to {output_dir}")
    print(f"  Global test set: {len(X_test):,} samples")
    print(f"  Total clients: {len(partitions)}")
    
    return metadata

def main():
    """
    Main data preparation pipeline
    """
    print("="*60)
    print("Fertility Risk Prediction - Data Preparation")
    print("="*60)
    
    # Check for data in raw directory
    raw_dir = 'data/raw'
    
    # Look for CSV files directly in data/raw
    if os.path.exists(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) 
                     if f.endswith('.csv') and os.path.getsize(os.path.join(raw_dir, f)) > 1000]
        
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV files in {raw_dir}:")
            for f in csv_files:
                size_mb = os.path.getsize(os.path.join(raw_dir, f)) / (1024*1024)
                print(f"  - {f} ({size_mb:.1f} MB)")
            
            # Load and merge all CSV files
            print(f"\nLoading and merging all CSV files...")
            X, y, features = load_and_preprocess_ahs_data(raw_dir)
        else:
            print("\nERROR: No valid CSV files found in data/raw/")
            print("Please download the dataset and place CSV files in data/raw/")
            return
    else:
        print(f"\nERROR: Directory {raw_dir} does not exist!")
        print("Please create the directory and place your CSV files there.")
        return
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Create federated partitions
    num_clients = 5  # Number of hospitals
    partitions = create_federated_partitions(X, y, num_clients=num_clients, iid=True)
    
    # Save processed data
    metadata = save_federated_data(X, y, features, partitions)
    
    print("\n" + "="*60)
    print("✓ Data preparation complete!")
    print("="*60)
    print(f"Number of features: {metadata['num_features']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Number of clients: {metadata['num_clients']}")
    print("\nNext steps:")
    print("1. Review the processed data in data/processed_dp/")
    print("2. Run: flwr run")
    print("="*60)

if __name__ == "__main__":
    main()