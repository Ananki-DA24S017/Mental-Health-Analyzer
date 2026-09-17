# src/data_processing/preprocess.py
import pandas as pd
import numpy as np
import time
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/preprocess_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_preprocessing")

def preprocess_data(input_file, output_dir):
    """
    Preprocess the mental health dataset and record throughput and speed metrics
    """
    # Start measuring time
    start_time = time.time()
    records_processed = 0
    
    logger.info(f"Starting preprocessing of {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    data_load_start = time.time()
    df = pd.read_csv(input_file)
    data_load_time = time.time() - data_load_start
    
    records_processed = len(df)
    load_throughput = records_processed / data_load_time if data_load_time > 0 else 0
    
    logger.info(f"Loaded {records_processed} records in {data_load_time:.2f} seconds")
    logger.info(f"Load throughput: {load_throughput:.2f} records/second")
    
    # Clean the data
    clean_start = time.time()
    
    # Handle missing values
    if 'statement' in df.columns:
        df['statement'] = df['statement'].fillna('')
        logger.info(f"Number of NaN values in 'statement': {df['statement'].isnull().sum()}")
    if 'status' in df.columns:
        df = df.dropna(subset=['status'])
    if 'sr' in df.columns:
        df['sr'] = df['sr'].fillna(0)
    
    df = df.fillna('')
    logger.info(f"Total NaN values in dataset after cleaning: {df.isnull().sum().sum()}")

    # Basic text preprocessing
    if 'statement' in df.columns:
        # Convert to lowercase
        df['statement'] = df['statement'].str.lower()
        # Remove extra whitespace
        df['statement'] = df['statement'].str.strip()
    
    # Create status code mapping
    if 'status' in df.columns:
        categories = df['status'].unique()
        category_mapping = {cat: i for i, cat in enumerate(sorted(categories))}
        df['status_code'] = df['status'].map(category_mapping)
        
        # Save category mapping for later use
        with open(f"{output_dir}/category_mapping.json", 'w') as f:
            json.dump(category_mapping, f)
    
    clean_time = time.time() - clean_start
    clean_throughput = records_processed / clean_time if clean_time > 0 else 0
    
    logger.info(f"Cleaned data in {clean_time:.2f} seconds")
    logger.info(f"Clean throughput: {clean_throughput:.2f} records/second")
    
    # Split the data
    split_start = time.time()
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    # Save the processed datasets
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    split_time = time.time() - split_start
    split_throughput = records_processed / split_time if split_time > 0 else 0
    
    logger.info(f"Split data in {split_time:.2f} seconds")
    logger.info(f"Split throughput: {split_throughput:.2f} records/second")
    
    # Calculate overall statistics
    total_time = time.time() - start_time
    overall_throughput = records_processed / total_time if total_time > 0 else 0
    
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Overall throughput: {overall_throughput:.2f} records/second")
    
    # Save metrics
    metrics = {
        "records_processed": records_processed,
        "load_time": data_load_time,
        "load_throughput": load_throughput,
        "clean_time": clean_time,
        "clean_throughput": clean_throughput,
        "split_time": split_time,
        "split_throughput": split_throughput,
        "total_time": total_time,
        "overall_throughput": overall_throughput
    }
    
    with open(f"{output_dir}/pipeline_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save as CSV for easier plotting
    pd.DataFrame([metrics]).to_csv(f"{output_dir}/pipeline_metrics.csv", index=False)
    
    return metrics

if __name__ == "__main__":
    input_file = "data/mental_health_data.csv"
    output_dir = "data/processed"
    
    metrics = preprocess_data(input_file, output_dir)
    
    print("\nPipeline Performance Summary:")
    print(f"Records processed: {metrics['records_processed']}")
    print(f"Total processing time: {metrics['total_time']:.2f} seconds")
    print(f"Overall throughput: {metrics['overall_throughput']:.2f} records/second")