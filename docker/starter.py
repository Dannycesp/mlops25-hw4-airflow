import sys
import os
import pickle
import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def main(year, month, step):
    input_file = f"/data/output/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    raw_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"/data/output/predictions_{year:04d}_{month:02d}.parquet"

    os.makedirs('/data/output', exist_ok=True)

    if step == 'download':
        logger.info(f"Downloading and cleaning data for {year}-{month}...")
        df = read_data(raw_file)
        df.to_parquet(input_file, engine='pyarrow', index=False)
        logger.info("Download step complete.")

    elif step == 'predict':
        logger.info("Loading model and vectorizer...")
        with open('model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)

        logger.info(f"Reading input data from {input_file}...")
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded dataframe with shape: {df.shape}")
        
        dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(dicts)
        
        # ADDED LOGGING: Log the shape of the transformed features
        logger.info(f"Feature matrix X_val has shape: {X_val.shape}")

        logger.info("Making predictions...")
        y_pred = model.predict(X_val)
        
        # ADDED LOGGING: Log the statistics of the raw predictions
        logger.info(f"Prediction stats: mean={np.mean(y_pred):.2f}, std={np.std(y_pred):.2f}, min={np.min(y_pred):.2f}, max={np.max(y_pred):.2f}")

        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
        df['predicted_duration'] = y_pred

        logger.info(f"Saving predictions to {output_file}...")
        df[['ride_id', 'predicted_duration']].to_parquet(output_file, engine='pyarrow', index=False)
        logger.info("Prediction step complete.")

    elif step == 'save':
        logger.info(f"Calculating stats for predictions from {output_file}")
        df = pd.read_parquet(output_file)
        logger.info(f"Mean predicted duration: {df.predicted_duration.mean():.2f}")
        logger.info(f"Std dev of predicted duration: {df.predicted_duration.std():.2f}")
        
        # ADDED LOGGING: Log the file size
        try:
            file_size_bytes = os.path.getsize(output_file)
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"Size of output file {output_file}: {file_size_mb:.2f} MB")
        except FileNotFoundError:
            logger.error(f"Output file not found: {output_file}")

        logger.info(f"Stats calculation complete. Predictions saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--step', type=str, required=True, choices=['download', 'predict', 'save'])
    args = parser.parse_args()
    main(args.year, args.month, args.step)