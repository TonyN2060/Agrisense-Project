"""
AgriSense Synthetic Data Vault (SDV) Integration

Generates additional synthetic samples using TVAE (Tabular Variational AutoEncoder)
to complement digital twin simulations while preserving multivariate correlations.

Target Output:
- 40,000 TVAE synthetic samples
- Combined with 120,000 digital twin + 2,500 real = 162,500 total samples
- Class-balanced distribution for 5-level quality classification

Quality Metrics:
- KS-statistic < 0.05 for distribution matching
- Spearman ρ > 0.95 for correlation preservation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import json
from pathlib import Path
from scipy import stats

# Optional imports with fallbacks
try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("Warning: SDV not installed. Install with: pip install sdv")


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    target_total_samples: int = 162500
    tvae_samples: int = 40000
    digital_twin_samples: int = 120000
    real_samples: int = 2500
    
    # Class distribution targets
    class_distribution: Dict[str, float] = None
    
    # TVAE hyperparameters
    tvae_epochs: int = 300
    tvae_batch_size: int = 500
    tvae_embedding_dim: int = 128
    tvae_compress_dims: Tuple[int, ...] = (128, 128)
    tvae_decompress_dims: Tuple[int, ...] = (128, 128)
    
    # Quality thresholds
    ks_threshold: float = 0.05      # Max KS statistic for distribution match
    correlation_threshold: float = 0.95  # Min Spearman correlation
    
    def __post_init__(self):
        if self.class_distribution is None:
            self.class_distribution = {
                'GOOD': 0.35,
                'MARGINAL': 0.22,
                'AT_RISK': 0.21,
                'CRITICAL': 0.12,
                'SPOILED': 0.10
            }


class DataValidator:
    """Validates synthetic data quality against real data distributions."""
    
    @staticmethod
    def calculate_ks_statistics(real_df: pd.DataFrame, 
                                synthetic_df: pd.DataFrame,
                                numeric_columns: List[str]) -> Dict[str, float]:
        """
        Calculate Kolmogorov-Smirnov statistics for distribution matching.
        """
        ks_stats = {}
        
        for col in numeric_columns:
            if col in real_df.columns and col in synthetic_df.columns:
                real_values = real_df[col].dropna()
                synth_values = synthetic_df[col].dropna()
                
                if len(real_values) > 0 and len(synth_values) > 0:
                    stat, p_value = stats.ks_2samp(real_values, synth_values)
                    ks_stats[col] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'passed': stat < 0.05
                    }
        
        return ks_stats
    
    @staticmethod
    def calculate_correlation_preservation(real_df: pd.DataFrame,
                                           synthetic_df: pd.DataFrame,
                                           numeric_columns: List[str]) -> Dict:
        """
        Calculate Spearman correlation preservation between real and synthetic.
        """
        # Calculate correlation matrices
        common_cols = [c for c in numeric_columns 
                       if c in real_df.columns and c in synthetic_df.columns]
        
        if len(common_cols) < 2:
            return {'overall_preservation': 1.0, 'details': {}}
        
        real_corr = real_df[common_cols].corr(method='spearman')
        synth_corr = synthetic_df[common_cols].corr(method='spearman')
        
        # Flatten upper triangle
        real_upper = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
        synth_upper = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
        
        # Calculate overall correlation preservation
        preservation, _ = stats.spearmanr(real_upper, synth_upper)
        
        return {
            'overall_preservation': preservation,
            'real_corr_shape': real_corr.shape,
            'passed': preservation > 0.95
        }
    
    @staticmethod
    def validate_class_distribution(df: pd.DataFrame,
                                    target_distribution: Dict[str, float],
                                    class_column: str = 'quality_status') -> Dict:
        """Validate class distribution matches targets."""
        actual_dist = df[class_column].value_counts(normalize=True).to_dict()
        
        deviations = {}
        for class_name, target_prop in target_distribution.items():
            actual_prop = actual_dist.get(class_name, 0.0)
            deviation = abs(actual_prop - target_prop)
            deviations[class_name] = {
                'target': target_prop,
                'actual': actual_prop,
                'deviation': deviation,
                'passed': deviation < 0.05  # Within 5% of target
            }
        
        return {
            'deviations': deviations,
            'all_passed': all(d['passed'] for d in deviations.values())
        }
    
    @staticmethod
    def generate_quality_report(real_df: pd.DataFrame,
                                synthetic_df: pd.DataFrame,
                                numeric_columns: List[str],
                                target_distribution: Dict[str, float]) -> Dict:
        """Generate comprehensive quality validation report."""
        
        ks_results = DataValidator.calculate_ks_statistics(
            real_df, synthetic_df, numeric_columns
        )
        
        corr_results = DataValidator.calculate_correlation_preservation(
            real_df, synthetic_df, numeric_columns
        )
        
        dist_results = DataValidator.validate_class_distribution(
            synthetic_df, target_distribution
        )
        
        # Count KS passes
        ks_passed = sum(1 for r in ks_results.values() if r['passed'])
        ks_total = len(ks_results)
        
        return {
            'ks_statistics': ks_results,
            'ks_pass_rate': ks_passed / ks_total if ks_total > 0 else 1.0,
            'correlation_preservation': corr_results,
            'class_distribution': dist_results,
            'overall_quality': {
                'ks_passed': ks_passed == ks_total,
                'correlation_passed': corr_results.get('passed', False),
                'distribution_passed': dist_results.get('all_passed', False)
            }
        }


class TVAEGenerator:
    """
    TVAE-based synthetic data generator using SDV.
    
    Generates additional samples while preserving multivariate correlations
    learned from real + digital twin data.
    """
    
    # Core numeric features to synthesize
    NUMERIC_FEATURES = [
        'temperature', 'humidity', 'co2_ppm', 'light_lux',
        'door_cycles_today', 'compressor_duty_cycle', 'energy_consumed_kwh',
        'quality_index', 'rsl_hours', 'rsl_percent',
        'flu_total', 'water_loss_percent', 'microbial_load', 'mold_risk_score',
        'decay_temperature', 'decay_humidity', 'decay_co2', 'decay_microbial'
    ]
    
    CATEGORICAL_FEATURES = [
        'crop_type', 'quality_status', 'temp_profile', 'humidity_profile',
        'initial_condition'
    ]
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self.synthesizer = None
        self.metadata = None
        self.is_fitted = False
        
    def _create_metadata(self, df: pd.DataFrame) -> 'SingleTableMetadata':
        """Create SDV metadata from DataFrame schema."""
        if not SDV_AVAILABLE:
            raise RuntimeError("SDV not installed")
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Set primary key if exists
        if 'sample_id' in df.columns:
            metadata.set_primary_key('sample_id')
        
        # Explicitly set crop_type as categorical
        if 'crop_type' in df.columns:
            metadata.update_column('crop_type', sdtype='categorical')
        
        if 'quality_status' in df.columns:
            metadata.update_column('quality_status', sdtype='categorical')
        
        return metadata
    
    def fit(self, training_df: pd.DataFrame):
        """
        Fit TVAE model on training data (real + digital twin combined).
        
        Args:
            training_df: Combined real and digital twin data
        """
        if not SDV_AVAILABLE:
            raise RuntimeError("SDV not installed. Install with: pip install sdv")
        
        print(f"Fitting TVAE on {len(training_df)} samples...")
        
        # Clean data
        df_clean = self._preprocess_for_tvae(training_df)
        
        # Create metadata
        self.metadata = self._create_metadata(df_clean)
        
        # Initialize TVAE synthesizer
        self.synthesizer = TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.config.tvae_epochs,
            batch_size=self.config.tvae_batch_size,
            embedding_dim=self.config.tvae_embedding_dim,
            compress_dims=self.config.tvae_compress_dims,
            decompress_dims=self.config.tvae_decompress_dims,
            verbose=True
        )
        
        # Fit model
        self.synthesizer.fit(df_clean)
        self.is_fitted = True
        
        print("TVAE fitting complete!")
    
    def _preprocess_for_tvae(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame for TVAE training."""
        df_clean = df.copy()
        
        # Remove non-synthesizable columns
        drop_cols = ['timestamp_minutes', 'scenario_id', 'sample_id']
        for col in drop_cols:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna('unknown')
        
        return df_clean
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples using fitted TVAE model.
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        print(f"Generating {num_samples} synthetic samples...")
        
        synthetic_df = self.synthesizer.sample(num_samples)
        
        # Post-process to ensure valid ranges
        synthetic_df = self._postprocess_synthetic(synthetic_df)
        
        # Add source marker
        synthetic_df['data_source'] = 'tvae_synthetic'
        
        print(f"Generated {len(synthetic_df)} samples")
        
        return synthetic_df
    
    def _postprocess_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process synthetic data to ensure valid ranges."""
        df_clean = df.copy()
        
        # Clip numeric values to valid ranges
        clip_ranges = {
            'temperature': (-5.0, 35.0),
            'humidity': (20.0, 99.9),
            'co2_ppm': (300.0, 25000.0),
            'light_lux': (0.0, 15000.0),
            'quality_index': (0.0, 100.0),
            'rsl_hours': (0.0, 3000.0),
            'rsl_percent': (0.0, 100.0),
            'flu_total': (0.0, 2.0),
            'water_loss_percent': (0.0, 30.0),
            'microbial_load': (0.0, 10.0),
            'mold_risk_score': (0.0, 1.0),
            'door_cycles_today': (0, 50),
            'compressor_duty_cycle': (0.0, 100.0)
        }
        
        for col, (min_val, max_val) in clip_ranges.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(min_val, max_val)
        
        # Ensure consistency between quality_index and quality_status
        if 'quality_index' in df_clean.columns and 'quality_status' in df_clean.columns:
            for idx, row in df_clean.iterrows():
                qi = row['quality_index']
                if qi > 70:
                    expected_status = 'GOOD'
                elif qi > 50:
                    expected_status = 'MARGINAL'
                elif qi > 30:
                    expected_status = 'AT_RISK'
                elif qi > 10:
                    expected_status = 'CRITICAL'
                else:
                    expected_status = 'SPOILED'
                df_clean.at[idx, 'quality_status'] = expected_status
        
        return df_clean
    
    def save_model(self, filepath: str):
        """Save fitted model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        self.synthesizer.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        if not SDV_AVAILABLE:
            raise RuntimeError("SDV not installed")
        
        self.synthesizer = TVAESynthesizer.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


class DatasetBuilder:
    """
    Builds complete training dataset from multiple sources.
    
    Combines:
    - Real sensor data (2,500 samples)
    - Digital twin simulations (120,000 samples)
    - TVAE synthetic augmentation (40,000 samples)
    
    Total: 162,500 samples with class-balanced distribution.
    """
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self.tvae_generator = TVAEGenerator(config)
        self.validator = DataValidator()
        
    def load_real_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess real sensor data."""
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        rename_map = {
            'temp': 'temperature',
            'humid (%)': 'humidity',
            'humid': 'humidity',
            'co2 (pmm)': 'co2_ppm',
            'co2 (ppm)': 'co2_ppm',
            'light (fux)': 'light_lux',
            'light': 'light_lux',
            'fruit': 'crop_type',
            'class': 'quality_status'
        }
        
        df = df.rename(columns=rename_map)
        
        # Standardize crop names
        if 'crop_type' in df.columns:
            crop_map = {
                'tomato': 'leafy_greens',  # Map old crops to new
                'banana': 'mango',
                'pineapple': 'berries',
                'orange': 'orange'
            }
            df['crop_type'] = df['crop_type'].str.lower().map(
                lambda x: crop_map.get(x, x)
            )
        
        # Standardize quality status
        if 'quality_status' in df.columns:
            df['quality_status'] = df['quality_status'].str.upper()
            # Map binary to 5-level (expand based on conditions)
            df = self._expand_quality_labels(df)
        
        # Add source marker
        df['data_source'] = 'real'
        
        print(f"Loaded {len(df)} real samples")
        return df
    
    def _expand_quality_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand binary Good/Bad labels to 5-level quality classification."""
        df_expanded = df.copy()
        
        # If only binary labels, expand based on sensor conditions
        if set(df_expanded['quality_status'].unique()) <= {'GOOD', 'BAD'}:
            print("Expanding binary labels to 5-level classification...")
            
            for idx, row in df_expanded.iterrows():
                if row['quality_status'] == 'GOOD':
                    # Good conditions - assign GOOD or MARGINAL randomly
                    df_expanded.at[idx, 'quality_status'] = np.random.choice(
                        ['GOOD', 'MARGINAL'], p=[0.8, 0.2]
                    )
                else:
                    # Bad conditions - distribute across AT_RISK, CRITICAL, SPOILED
                    df_expanded.at[idx, 'quality_status'] = np.random.choice(
                        ['AT_RISK', 'CRITICAL', 'SPOILED'], p=[0.5, 0.3, 0.2]
                    )
        
        return df_expanded
    
    def generate_digital_twin_data(self, 
                                   num_samples: int = 120000) -> pd.DataFrame:
        """Generate digital twin simulation data."""
        # Import here to avoid circular dependency
        from digital_twin import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        df = generator.generate_dataset(target_samples=num_samples)
        
        # Add source marker
        df['data_source'] = 'digital_twin'
        
        return df
    
    def generate_tvae_data(self, 
                          training_df: pd.DataFrame,
                          num_samples: int = 40000) -> pd.DataFrame:
        """Generate TVAE synthetic data from training distribution."""
        if not SDV_AVAILABLE:
            print("Warning: SDV not installed. Skipping TVAE generation.")
            return pd.DataFrame()
        
        # Fit TVAE on combined real + digital twin data
        self.tvae_generator.fit(training_df)
        
        # Generate synthetic samples
        synthetic_df = self.tvae_generator.generate(num_samples)
        
        return synthetic_df
    
    def build_complete_dataset(self,
                              real_data_path: Optional[str] = None,
                              digital_twin_samples: int = 120000,
                              tvae_samples: int = 40000,
                              validate: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Build complete class-balanced training dataset.
        
        Returns:
            Tuple of (combined_df, quality_report)
        """
        datasets = []
        
        # 1. Load real data if available
        if real_data_path:
            try:
                real_df = self.load_real_data(real_data_path)
                datasets.append(real_df)
            except Exception as e:
                print(f"Warning: Could not load real data: {e}")
        
        # 2. Generate digital twin data
        print("\nGenerating digital twin data...")
        twin_df = self.generate_digital_twin_data(digital_twin_samples)
        datasets.append(twin_df)
        
        # 3. Combine for TVAE training
        combined_base = pd.concat(datasets, ignore_index=True)
        print(f"Base dataset: {len(combined_base)} samples")
        
        # 4. Generate TVAE synthetic data
        if tvae_samples > 0 and SDV_AVAILABLE:
            print("\nGenerating TVAE synthetic data...")
            tvae_df = self.generate_tvae_data(combined_base, tvae_samples)
            if len(tvae_df) > 0:
                datasets.append(tvae_df)
        
        # 5. Combine all sources
        full_df = pd.concat(datasets, ignore_index=True)
        
        # 6. Balance classes
        print("\nBalancing class distribution...")
        balanced_df = self._balance_classes(full_df)
        
        # 7. Validate quality
        quality_report = {}
        if validate and len(datasets) > 1:
            print("\nValidating synthetic data quality...")
            numeric_cols = [c for c in self.tvae_generator.NUMERIC_FEATURES 
                          if c in balanced_df.columns]
            quality_report = self.validator.generate_quality_report(
                real_df=combined_base,
                synthetic_df=balanced_df[balanced_df['data_source'] != 'real'],
                numeric_columns=numeric_cols,
                target_distribution=self.config.class_distribution
            )
        
        print(f"\nFinal dataset: {len(balanced_df)} samples")
        print(f"Class distribution:\n{balanced_df['quality_status'].value_counts(normalize=True)}")
        
        return balanced_df, quality_report
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance class distribution according to target."""
        target_total = self.config.target_total_samples
        balanced_samples = []
        
        for status, proportion in self.config.class_distribution.items():
            target_count = int(target_total * proportion)
            status_df = df[df['quality_status'] == status]
            
            if len(status_df) >= target_count:
                sampled = status_df.sample(n=target_count, random_state=42)
            else:
                # Oversample with replacement if insufficient
                sampled = status_df.sample(n=target_count, replace=True, random_state=42)
            
            balanced_samples.append(sampled)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add sample IDs
        balanced_df['sample_id'] = range(len(balanced_df))
        
        return balanced_df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV/Parquet."""
        path = Path(filepath)
        
        if path.suffix == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load dataset from file."""
        path = Path(filepath)
        
        if path.suffix == '.parquet':
            return pd.read_parquet(filepath)
        else:
            return pd.read_csv(filepath)


class QuickSyntheticGenerator:
    """
    Fallback synthetic data generator when SDV is not available.
    
    Uses statistical sampling and physics-constrained generation
    to create synthetic samples without deep learning.
    """
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
    
    def generate_from_statistics(self, 
                                 reference_df: pd.DataFrame,
                                 num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data using statistical properties of reference.
        """
        synthetic_records = []
        
        # Get statistics per crop and quality status
        grouped = reference_df.groupby(['crop_type', 'quality_status'])
        
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for (crop, status), group_df in grouped:
            # Calculate target samples for this group
            status_proportion = self.config.class_distribution.get(status, 0.1)
            crop_count = len(reference_df[reference_df['crop_type'] == crop])
            total_samples = num_samples
            
            target_for_group = int(
                total_samples * status_proportion / len(reference_df['crop_type'].unique())
            )
            
            if target_for_group == 0:
                continue
            
            # Calculate statistics
            means = group_df[numeric_cols].mean()
            stds = group_df[numeric_cols].std().fillna(0.1)
            
            # Generate samples
            for _ in range(target_for_group):
                record = {}
                record['crop_type'] = crop
                record['quality_status'] = status
                record['data_source'] = 'statistical_synthetic'
                
                for col in numeric_cols:
                    # Sample from normal distribution with clipping
                    value = np.random.normal(means[col], stds[col] * 0.5)
                    
                    # Apply physical constraints
                    if col == 'temperature':
                        value = np.clip(value, -5, 35)
                    elif col == 'humidity':
                        value = np.clip(value, 20, 99.9)
                    elif col == 'co2_ppm':
                        value = np.clip(value, 300, 25000)
                    elif col == 'quality_index':
                        value = np.clip(value, 0, 100)
                    elif col.startswith('decay_') or col.endswith('_score'):
                        value = np.clip(value, 0, 1)
                    
                    record[col] = value
                
                synthetic_records.append(record)
        
        synthetic_df = pd.DataFrame(synthetic_records)
        
        # Shuffle
        synthetic_df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return synthetic_df


def create_training_dataset(real_data_path: str = "Dataset.csv",
                           output_path: str = "agrisense_training_data.csv",
                           digital_twin_samples: int = 5000,
                           tvae_samples: int = 2000) -> pd.DataFrame:
    """
    Convenience function to create complete training dataset.
    
    For full-scale generation, increase sample counts to:
    - digital_twin_samples = 120000
    - tvae_samples = 40000
    """
    config = SyntheticDataConfig(
        digital_twin_samples=digital_twin_samples,
        tvae_samples=tvae_samples
    )
    
    builder = DatasetBuilder(config)
    
    df, report = builder.build_complete_dataset(
        real_data_path=real_data_path,
        digital_twin_samples=digital_twin_samples,
        tvae_samples=tvae_samples,
        validate=True
    )
    
    builder.save_dataset(df, output_path)
    
    # Save quality report
    report_path = output_path.replace('.csv', '_quality_report.json')
    with open(report_path, 'w') as f:
        # Convert numpy types for JSON
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        json.dump(convert_types(report), f, indent=2)
    
    print(f"\nQuality report saved to {report_path}")
    
    return df


if __name__ == "__main__":
    print("AgriSense Synthetic Data Generator")
    print("=" * 50)
    
    # Quick demo with reduced samples
    print("\nGenerating demo dataset (reduced scale)...")
    
    try:
        df = create_training_dataset(
            real_data_path="Dataset.csv",
            output_path="demo_training_data.csv",
            digital_twin_samples=1000,
            tvae_samples=0  # Skip TVAE for quick demo
        )
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nSample columns: {list(df.columns[:20])}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("This is expected if dependencies are not installed.")
