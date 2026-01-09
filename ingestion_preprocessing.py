"""
Agrisense – Data Ingestion & Preprocessing Script

- Loads raw sensor data
- Applies the same preprocessing used during model training
- Produces clean feature matrices for inference or storage

Designed to integrate with:
- CSV batch ingestion (current)
- InfluxDB / streaming pipelines (future)
"""

import pandas as pd
import joblib
from typing import Union

# -----------------------------
# Configuration
# -----------------------------
FEATURES = ["temperature", "humidity", "co2"]
PIPELINE_PATH = "storage_quality_pipeline.pkl"


# -----------------------------
# Data Ingestion
# -----------------------------
def load_sensor_data(source: Union[str, dict]) -> pd.DataFrame:
    """
    Load sensor data from different sources.

    Parameters
    ----------
    source : str or dict
        - CSV file path (batch mode)
        - Dictionary (real-time ingestion)

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, str):
        # CSV ingestion
        df = pd.read_csv(source)

    elif isinstance(source, dict):
        # Streaming / API ingestion
        df = pd.DataFrame([source])

    else:
        raise ValueError("Unsupported data source type")

    return df


# -----------------------------
# Standardize Columns
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column names and units match training schema.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "temp": "temperature",
        "humid (%)": "humidity",
        "co2 (pmm)": "co2"
    }

    df.rename(columns=rename_map, inplace=True)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[FEATURES]


# -----------------------------
# Preprocessing + Inference
# -----------------------------
def preprocess_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply trained preprocessing pipeline and predict quality.
    """
    pipeline = joblib.load(PIPELINE_PATH)

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    output = df.copy()
    output["prediction"] = predictions
    output["confidence"] = probabilities

    output["prediction_label"] = output["prediction"].map(
        {1: "GOOD", 0: "BAD"}
    )

    return output


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":

    
    raw_df = load_sensor_data("Dataset.csv")
    clean_df = standardize_columns(raw_df)
    results = preprocess_and_predict(clean_df)

    print(results.head())

    # Example 2: Real-time ingestion (InfluxDB-style)
    realtime_sample = {
        "temperature": 12.8,
        "humidity": 87.5,
        "co2": 410,
        
    }

    rt_df = load_sensor_data(realtime_sample)
    rt_df = standardize_columns(rt_df)
    rt_result = preprocess_and_predict(rt_df)

    print("\nReal-time prediction:")
    print(rt_result)
