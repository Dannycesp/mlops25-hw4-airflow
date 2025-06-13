#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import argparse
import os

def read_data(filename):
    """Reads and preprocesses taxi trip data from a parquet file."""
    print(f"Reading data from {filename}...")
    try:
        df = pd.read_parquet(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
    
    # Define categorical features, consistent with the model training
    categorical = ['PULocationID', 'DOLocationID']
    
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out trips with unreasonable durations
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Prepare categorical features for the model
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    """Main function to run the batch prediction process."""
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output_script/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    # Create output directory if it doesn't exist
    os.makedirs('output_script', exist_ok=True)

    # Load the model and DictVectorizer
    # This assumes 'model.bin' is in the same directory as the script
    print("Loading model and vectorizer...")
    try:
        with open('models/model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print("Error: model.bin not found. Please ensure it's in the same directory.")
        return

    df = read_data(input_file)
    if df is None:
        return

    # Prepare data for prediction
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    print("Applying the model to make predictions...")
    y_pred = model.predict(X_val)
    
    # --- Homework Q1: Standard Deviation ---
    std_dev = np.std(y_pred)
    print(f"Q1 Answer: Standard deviation of predicted durations = {std_dev:.2f}")

    # --- Homework Q5: Mean Prediction ---
    mean_pred = np.mean(y_pred)
    print(f"Q5 Answer: Mean of predicted durations = {mean_pred:.2f}")

    # Prepare the results dataframe
    print("Creating result dataframe with ride_id...")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    # Save results to a parquet file
    print(f"Saving predictions to {output_file}...")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    # --- Homework Q2: File Size ---
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Q2 Answer: Size of the output file = {file_size_mb:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch prediction for taxi ride durations.')
    parser.add_argument('--year', type=int, required=True, help='The year of the taxi data (e.g., 2023).')
    parser.add_argument('--month', type=int, required=True, help='The month of the taxi data (e.g., 3 for March).')
    args = parser.parse_args()
    
    main(args.year, args.month)
