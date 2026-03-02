"""
AgriSense ML Module - Main Pipeline Orchestrator

Complete training and evaluation pipeline:
1. Load real data
2. Generate synthetic data (Digital Twin + TVAE)
3. Feature engineering
4. Hierarchical model training
5. SHAP explainability setup
6. Model evaluation and export
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Import AgriSense Modules
# =============================================================================

from config import get_config, AgriSenseConfig
from physics_engine import CropType, CROP_PHYSICS_PARAMS
from digital_twin import SyntheticDataGenerator, ScenarioGenerator
from synthetic_data import DatasetBuilder
from feature_engineering import FeaturePipeline, FeatureEngineer
from hierarchical_model import HierarchicalEnsemble, ModelConfig
from explainability import ExplainabilityService


# =============================================================================
# Pipeline Stages
# =============================================================================

class DataIngestionStage:
    """Stage 1: Load and preprocess real data."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, data_path: str = "Dataset.csv") -> pd.DataFrame:
        """Load real dataset."""
        logger.info("Stage 1: Data Ingestion")
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(data_path)
        logger.info(f"  Loaded {len(df)} samples from {data_path}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Standardize column names
        column_mapping = {
            'Temp': 'temperature',
            'Humid': 'humidity',
            'CO2': 'co2_ppm',
            'Light': 'light_lux',
            'Class': 'quality_label',
            'Fruit': 'crop_type'
        }
        df = df.rename(columns=column_mapping)
        
        # Map quality labels
        if 'quality_label' in df.columns:
            label_mapping = {
                'Good': 'GOOD',
                'Bad': 'SPOILED',
                'Marginal': 'MARGINAL',
                'At Risk': 'AT_RISK',
                'Critical': 'CRITICAL'
            }
            df['quality_label'] = df['quality_label'].map(
                lambda x: label_mapping.get(x, x)
            )
        
        # Map crop types
        if 'crop_type' in df.columns:
            crop_mapping = self.config.crop.legacy_crop_mapping
            df['crop_type'] = df['crop_type'].str.lower().map(
                lambda x: crop_mapping.get(x, 'mango')
            )
        
        logger.info(f"  Processed {len(df)} samples")
        return df


class SyntheticDataStage:
    """Stage 2: Generate synthetic data."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, real_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data combining digital twin and TVAE."""
        logger.info("Stage 2: Synthetic Data Generation")
        
        # Initialize generators
        dt_generator = SyntheticDataGenerator()
        dataset_builder = DatasetBuilder()
        
        # Generate digital twin data
        logger.info("  Generating Digital Twin samples...")
        dt_samples = min(
            self.config.synthetic_data.digital_twin_samples,
            30000  # Limit for demonstration
        )
        dt_data = dt_generator.generate_dataset(total_samples=dt_samples)
        dt_df = pd.DataFrame(dt_data)
        logger.info(f"    Generated {len(dt_df)} digital twin samples")
        
        # Build combined dataset
        logger.info("  Building combined dataset...")
        combined_df = dataset_builder.build_dataset(
            real_df=real_df,
            digital_twin_df=dt_df,
            use_tvae=False  # Disable TVAE for faster demo
        )
        
        # Validate
        if dataset_builder.validate_dataset(combined_df):
            logger.info(f"  Dataset validation passed")
        else:
            logger.warning("  Dataset validation had warnings")
        
        logger.info(f"  Combined dataset: {len(combined_df)} samples")
        return combined_df


class FeatureEngineeringStage:
    """Stage 3: Feature engineering."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, df: pd.DataFrame) -> tuple:
        """Apply feature engineering pipeline."""
        logger.info("Stage 3: Feature Engineering")
        
        pipeline = FeaturePipeline()
        engineer = FeatureEngineer()
        
        # Extract features for all samples
        features_list = []
        crop_types = []
        labels = []
        
        for idx, row in df.iterrows():
            try:
                # Build sensor dict
                sensor_dict = {
                    'temperature': row.get('temperature', 20),
                    'humidity': row.get('humidity', 80),
                    'co2_ppm': row.get('co2_ppm', 400),
                    'light_lux': row.get('light_lux', 0)
                }
                
                crop_type = row.get('crop_type', 'mango')
                quality = row.get('quality_label', 'GOOD')
                
                # Extract features
                features = engineer.extract_features(sensor_dict, crop_type)
                features_list.append(features)
                crop_types.append(crop_type)
                labels.append(quality)
                
            except Exception as e:
                logger.debug(f"  Skipping row {idx}: {e}")
        
        # Convert to arrays
        feature_names = engineer.FEATURE_NAMES
        X = np.array([
            [f.get(name, 0.0) for name in feature_names]
            for f in features_list
        ])
        
        # Encode labels
        label_to_int = {
            'GOOD': 0, 'MARGINAL': 1, 'AT_RISK': 2,
            'CRITICAL': 3, 'SPOILED': 4
        }
        y = np.array([label_to_int.get(l, 0) for l in labels])
        crop_types = np.array(crop_types)
        
        # Fit pipeline (normalization)
        pipeline.fit(X)
        X_normalized = pipeline.transform(X)
        
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Labels shape: {y.shape}")
        logger.info(f"  Label distribution: {np.bincount(y)}")
        
        return X_normalized, y, crop_types, feature_names, pipeline


class ModelTrainingStage:
    """Stage 4: Hierarchical model training."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, X: np.ndarray, y: np.ndarray, 
            crop_types: np.ndarray, feature_names: list,
            pipeline: FeaturePipeline) -> HierarchicalEnsemble:
        """Train hierarchical ensemble model."""
        logger.info("Stage 4: Model Training")
        
        # Create model config
        model_config = ModelConfig(
            optuna_trials=self.config.model.optuna_trials,
            num_features=len(feature_names),
            target_metric='f1_weighted'
        )
        
        # Initialize ensemble
        ensemble = HierarchicalEnsemble(config=model_config)
        ensemble.feature_names = feature_names
        ensemble.feature_pipeline = pipeline
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, crops_train, crops_val = train_test_split(
            X, y, crop_types, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Validation set: {len(X_val)} samples")
        
        # Train ensemble
        logger.info("  Training hierarchical ensemble...")
        ensemble.fit(X_train, y_train, crops_train)
        
        # Evaluate
        metrics = ensemble.evaluate(X_val, y_val, crops_val)
        
        logger.info(f"  Validation Results:")
        logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        return ensemble


class ExplainabilityStage:
    """Stage 5: Setup explainability service."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, ensemble: HierarchicalEnsemble, 
            X_sample: np.ndarray) -> ExplainabilityService:
        """Setup SHAP explainability."""
        logger.info("Stage 5: Explainability Setup")
        
        explainer = ExplainabilityService(
            model=ensemble,
            feature_names=ensemble.feature_names
        )
        
        # Initialize SHAP with background data
        explainer.initialize(X_sample[:100])
        
        logger.info("  SHAP explainer initialized")
        return explainer


class ModelExportStage:
    """Stage 6: Export trained model."""
    
    def __init__(self, config: AgriSenseConfig):
        self.config = config
        
    def run(self, ensemble: HierarchicalEnsemble, 
            explainer: ExplainabilityService) -> str:
        """Export model and artifacts."""
        logger.info("Stage 6: Model Export")
        
        model_dir = self.config.model.model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble
        ensemble.save(model_dir)
        
        # Save metadata
        metadata = {
            'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'feature_count': len(ensemble.feature_names),
            'crops_supported': list(self.config.crop.supported_crops),
            'quality_levels': list(self.config.crop.quality_levels),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Model exported to {model_dir}")
        return model_dir


# =============================================================================
# Main Pipeline
# =============================================================================

class AgriSensePipeline:
    """Main orchestrator for AgriSense ML pipeline."""
    
    def __init__(self, config: Optional[AgriSenseConfig] = None):
        self.config = config or get_config()
        
        # Initialize stages
        self.data_stage = DataIngestionStage(self.config)
        self.synthetic_stage = SyntheticDataStage(self.config)
        self.feature_stage = FeatureEngineeringStage(self.config)
        self.training_stage = ModelTrainingStage(self.config)
        self.explainability_stage = ExplainabilityStage(self.config)
        self.export_stage = ModelExportStage(self.config)
        
    def run(self, data_path: str = "Dataset.csv", 
            skip_synthetic: bool = False) -> Dict:
        """Run complete pipeline."""
        logger.info("=" * 60)
        logger.info("AgriSense ML Pipeline")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # Stage 1: Data Ingestion
        real_df = self.data_stage.run(data_path)
        results['real_samples'] = len(real_df)
        
        # Stage 2: Synthetic Data
        if skip_synthetic:
            combined_df = real_df
            logger.info("Skipping synthetic data generation")
        else:
            combined_df = self.synthetic_stage.run(real_df)
        results['total_samples'] = len(combined_df)
        
        # Stage 3: Feature Engineering
        X, y, crop_types, feature_names, pipeline = self.feature_stage.run(combined_df)
        results['feature_count'] = len(feature_names)
        
        # Stage 4: Model Training
        ensemble = self.training_stage.run(X, y, crop_types, feature_names, pipeline)
        
        # Stage 5: Explainability
        explainer = self.explainability_stage.run(ensemble, X)
        
        # Stage 6: Export
        model_dir = self.export_stage.run(ensemble, explainer)
        results['model_dir'] = model_dir
        
        # Summary
        elapsed = time.time() - start_time
        results['elapsed_seconds'] = elapsed
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info(f"  Total time: {elapsed:.1f} seconds")
        logger.info(f"  Samples used: {results['total_samples']}")
        logger.info(f"  Features: {results['feature_count']}")
        logger.info(f"  Model saved to: {model_dir}")
        logger.info("=" * 60)
        
        return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgriSense ML Pipeline")
    parser.add_argument(
        '--data', 
        type=str, 
        default='Dataset.csv',
        help='Path to input dataset'
    )
    parser.add_argument(
        '--skip-synthetic',
        action='store_true',
        help='Skip synthetic data generation'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=50,
        help='Number of Optuna optimization trials'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = get_config()
    config.model.model_dir = args.model_dir
    config.model.optuna_trials = args.optuna_trials
    
    # Run pipeline
    pipeline = AgriSensePipeline(config)
    results = pipeline.run(
        data_path=args.data,
        skip_synthetic=args.skip_synthetic
    )
    
    print("\nResults:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
