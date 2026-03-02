"""
AgriSense Feature Engineering Pipeline

Transforms raw sensor readings into 39 high-level agricultural indicators:
- Raw Sensors (4): temperature, humidity, CO2, light
- Temporal Features (12): 24-hour lookback statistics
- Psychrometric Features (4): dew point, VPD, condensation risk, enthalpy
- Crop-Specific Weighted Features (4): importance-weighted sensor values
- Interaction Features (7): combined stress indicators
- Infrastructure Features (5): door cycles, energy, pressure
- Sensor Quality Features (3): anomaly detection, data quality scoring

Total: 39 engineered features for hierarchical ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import math

from physics_engine import CropType, CROP_PHYSICS_PARAMS, CropPhysicsParams


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    temporal_lookback_hours: int = 24
    timestep_minutes: int = 5
    
    # Psychrometric constants
    pressure_kpa: float = 101.325  # Standard atmospheric pressure
    
    # Anomaly detection thresholds
    temperature_jump_threshold: float = 5.0  # °C in 5 minutes
    humidity_jump_threshold: float = 10.0    # % in 5 minutes
    co2_jump_threshold: float = 500.0        # ppm in 5 minutes
    
    # Infrastructure baselines
    baseline_door_cycles_per_day: int = 5
    max_door_cycles_per_day: int = 30
    expected_compressor_duty_cycle: float = 50.0


class TemporalFeatureBuffer:
    """
    Maintains rolling buffer for temporal feature computation.
    
    Stores historical readings for computing 24-hour statistics.
    """
    
    def __init__(self, max_samples: int = 288):  # 24 hours at 5-min intervals
        self.max_samples = max_samples
        self.temperature_buffer = deque(maxlen=max_samples)
        self.humidity_buffer = deque(maxlen=max_samples)
        self.co2_buffer = deque(maxlen=max_samples)
        self.light_buffer = deque(maxlen=max_samples)
        
    def add_reading(self, temperature: float, humidity: float, 
                   co2: float, light: float):
        """Add new sensor reading to buffers."""
        self.temperature_buffer.append(temperature)
        self.humidity_buffer.append(humidity)
        self.co2_buffer.append(co2)
        self.light_buffer.append(light)
    
    def get_temperature_stats(self) -> Dict[str, float]:
        """Get 24-hour temperature statistics."""
        if len(self.temperature_buffer) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'trend': 0}
        
        values = np.array(self.temperature_buffer)
        
        # Calculate trend (change over last hour, 12 samples)
        if len(values) >= 12:
            recent = values[-12:]
            trend = recent[-1] - recent[0]
        else:
            trend = 0.0
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'trend': float(trend)
        }
    
    def get_humidity_stats(self) -> Dict[str, float]:
        """Get 24-hour humidity statistics."""
        if len(self.humidity_buffer) == 0:
            return {'mean': 0, 'min': 0, 'deficit_cumulative': 0}
        
        values = np.array(self.humidity_buffer)
        
        # Humidity deficit = cumulative time below 100%
        deficit = sum(100 - h for h in values)  # Simplified deficit calculation
        
        return {
            'mean': float(np.mean(values)),
            'min': float(np.min(values)),
            'deficit_cumulative': float(deficit)
        }
    
    def get_co2_stats(self) -> Dict[str, float]:
        """Get 24-hour CO2 statistics."""
        if len(self.co2_buffer) == 0:
            return {'rate_of_change': 0, 'accumulation': 0, 'time_above_600': 0}
        
        values = np.array(self.co2_buffer)
        
        # Rate of change (ppm/hour)
        if len(values) >= 12:
            roc = (values[-1] - values[-12]) / 1.0  # per hour
        else:
            roc = 0.0
        
        # Accumulation (sum of excess above baseline)
        accumulation = sum(max(0, c - 400) for c in values)
        
        # Time above 600 ppm (in timesteps)
        time_above = sum(1 for c in values if c > 600)
        
        return {
            'rate_of_change': float(roc),
            'accumulation': float(accumulation),
            'time_above_600': int(time_above)
        }
    
    def get_light_stats(self) -> Dict[str, float]:
        """Get 24-hour light statistics."""
        if len(self.light_buffer) == 0:
            return {'cumulative': 0}
        
        values = np.array(self.light_buffer)
        
        # Cumulative lux-hours (for photodegradation)
        cumulative = sum(values) * (5/60)  # Convert to lux-hours
        
        return {'cumulative': float(cumulative)}
    
    def reset(self):
        """Clear all buffers."""
        self.temperature_buffer.clear()
        self.humidity_buffer.clear()
        self.co2_buffer.clear()
        self.light_buffer.clear()


class PsychrometricCalculator:
    """
    Calculates psychrometric properties from temperature and humidity.
    
    Based on ASHRAE fundamentals for moist air calculations.
    """
    
    @staticmethod
    def calculate_saturation_pressure(temperature_c: float) -> float:
        """
        Calculate saturation vapor pressure (kPa) using Magnus formula.
        """
        if temperature_c >= 0:
            a, b, c = 0.61094, 17.625, 243.04
        else:
            a, b, c = 0.61115, 22.452, 272.55
        
        return a * np.exp((b * temperature_c) / (c + temperature_c))
    
    @staticmethod
    def calculate_dew_point(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate dew point temperature (°C).
        """
        if relative_humidity <= 0:
            return -50.0  # Extremely low
        
        a, b = 17.625, 243.04
        
        alpha = ((a * temperature_c) / (b + temperature_c)) + np.log(relative_humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return float(dew_point)
    
    @staticmethod
    def calculate_vapor_pressure_deficit(temperature_c: float, 
                                         relative_humidity: float) -> float:
        """
        Calculate Vapor Pressure Deficit (VPD) in kPa.
        
        High VPD = drying stress on produce
        """
        sat_pressure = PsychrometricCalculator.calculate_saturation_pressure(temperature_c)
        actual_pressure = sat_pressure * (relative_humidity / 100.0)
        vpd = sat_pressure - actual_pressure
        
        return float(max(0, vpd))
    
    @staticmethod
    def calculate_condensation_risk(temperature_c: float, 
                                   relative_humidity: float,
                                   surface_temp_offset: float = -2.0) -> float:
        """
        Calculate probability of condensation on cold surfaces.
        
        Returns value 0-1 indicating condensation risk.
        """
        dew_point = PsychrometricCalculator.calculate_dew_point(
            temperature_c, relative_humidity
        )
        
        # Coldest surface is typically colder than ambient
        surface_temp = temperature_c + surface_temp_offset
        
        # Risk increases as surface approaches dew point
        temp_margin = surface_temp - dew_point
        
        if temp_margin <= 0:
            return 1.0  # Definitely condensing
        elif temp_margin >= 5:
            return 0.0  # Safe margin
        else:
            return float(1.0 - (temp_margin / 5.0))
    
    @staticmethod
    def calculate_enthalpy(temperature_c: float, 
                          relative_humidity: float,
                          pressure_kpa: float = 101.325) -> float:
        """
        Calculate specific enthalpy of moist air (kJ/kg dry air).
        
        Correlates with spoilage rate - higher enthalpy = faster spoilage.
        """
        sat_pressure = PsychrometricCalculator.calculate_saturation_pressure(temperature_c)
        vapor_pressure = sat_pressure * (relative_humidity / 100.0)
        
        # Humidity ratio (kg water / kg dry air)
        W = 0.622 * (vapor_pressure / (pressure_kpa - vapor_pressure))
        
        # Enthalpy calculation
        h = (1.006 * temperature_c) + W * (2501 + 1.86 * temperature_c)
        
        return float(h)


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for AgriSense ML models.
    
    Transforms raw sensor data into 39 engineered features optimized
    for crop-specific spoilage prediction.
    """
    
    # Define all 39 features
    FEATURE_NAMES = [
        # Raw Sensors (4)
        'temperature', 'humidity', 'co2', 'light',
        
        # Temporal Features (12)
        'temperature_mean_24h', 'temperature_std_24h', 
        'temperature_min_24h', 'temperature_max_24h', 'temperature_trend',
        'humidity_mean_24h', 'humidity_min_24h', 'humidity_deficit_cumulative',
        'co2_rate_of_change', 'co2_accumulation_24h', 'co2_time_above_600',
        'light_cumulative_24h',
        
        # Psychrometric Features (4)
        'dew_point', 'vapor_pressure_deficit', 'condensation_risk', 'enthalpy',
        
        # Crop-Specific Weighted Features (4)
        'temperature_weighted', 'humidity_weighted', 'co2_weighted', 'light_weighted',
        
        # Interaction Features (7)
        'temperature_humidity_interaction', 'temperature_co2_interaction',
        'humidity_co2_ratio', 'high_temp_low_humidity_flag',
        'low_temp_high_humidity_flag', 'anaerobic_risk_flag', 'stress_index',
        
        # Infrastructure Features (5)
        'door_cycle_frequency', 'door_cycle_stress', 'energy_anomaly_score',
        'compressor_runtime_pct', 'pressure_stability_score',
        
        # Sensor Quality Features (3)
        'sensor_anomaly_composite', 'data_quality_score', 'feasibility_score'
    ]
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.temporal_buffer = TemporalFeatureBuffer()
        self.psychro = PsychrometricCalculator()
        
        # Previous readings for anomaly detection
        self.prev_temperature = None
        self.prev_humidity = None
        self.prev_co2 = None
        
    def extract_features(self, 
                        sensor_reading: Dict,
                        crop_type: Union[str, CropType] = None) -> Dict[str, float]:
        """
        Extract all 39 features from a sensor reading.
        
        Args:
            sensor_reading: Dict with sensor values
            crop_type: Crop type for crop-specific weighting
            
        Returns:
            Dict with all 39 engineered features
        """
        # Get raw values
        temperature = sensor_reading.get('temperature', 20.0)
        humidity = sensor_reading.get('humidity', 70.0)
        co2 = sensor_reading.get('co2_ppm', sensor_reading.get('co2', 400.0))
        light = sensor_reading.get('light_lux', sensor_reading.get('light', 0.0))
        
        # Update temporal buffer
        self.temporal_buffer.add_reading(temperature, humidity, co2, light)
        
        # Get crop parameters
        if crop_type:
            if isinstance(crop_type, str):
                crop_type = self._parse_crop_type(crop_type)
            crop_params = CROP_PHYSICS_PARAMS.get(crop_type)
        else:
            crop_params = None
        
        features = {}
        
        # 1. Raw Sensors (4)
        features['temperature'] = temperature
        features['humidity'] = humidity
        features['co2'] = co2
        features['light'] = light
        
        # 2. Temporal Features (12)
        features.update(self._extract_temporal_features())
        
        # 3. Psychrometric Features (4)
        features.update(self._extract_psychrometric_features(temperature, humidity))
        
        # 4. Crop-Specific Weighted Features (4)
        features.update(self._extract_crop_weighted_features(
            temperature, humidity, co2, light, crop_params
        ))
        
        # 5. Interaction Features (7)
        features.update(self._extract_interaction_features(
            temperature, humidity, co2, crop_params
        ))
        
        # 6. Infrastructure Features (5)
        features.update(self._extract_infrastructure_features(sensor_reading))
        
        # 7. Sensor Quality Features (3)
        features.update(self._extract_quality_features(
            temperature, humidity, co2, crop_params
        ))
        
        # Update previous values for next iteration
        self.prev_temperature = temperature
        self.prev_humidity = humidity
        self.prev_co2 = co2
        
        return features
    
    def _parse_crop_type(self, crop_name: str) -> CropType:
        """Parse crop name string to CropType enum."""
        crop_map = {
            'avocado': CropType.AVOCADO,
            'mango': CropType.MANGO,
            'leafy_greens': CropType.LEAFY_GREENS,
            'leafy greens': CropType.LEAFY_GREENS,
            'orange': CropType.ORANGE,
            'berries': CropType.BERRIES,
            'berry': CropType.BERRIES
        }
        return crop_map.get(crop_name.lower(), CropType.AVOCADO)
    
    def _extract_temporal_features(self) -> Dict[str, float]:
        """Extract 12 temporal features from buffer."""
        temp_stats = self.temporal_buffer.get_temperature_stats()
        humidity_stats = self.temporal_buffer.get_humidity_stats()
        co2_stats = self.temporal_buffer.get_co2_stats()
        light_stats = self.temporal_buffer.get_light_stats()
        
        return {
            'temperature_mean_24h': temp_stats['mean'],
            'temperature_std_24h': temp_stats['std'],
            'temperature_min_24h': temp_stats['min'],
            'temperature_max_24h': temp_stats['max'],
            'temperature_trend': temp_stats['trend'],
            'humidity_mean_24h': humidity_stats['mean'],
            'humidity_min_24h': humidity_stats['min'],
            'humidity_deficit_cumulative': humidity_stats['deficit_cumulative'],
            'co2_rate_of_change': co2_stats['rate_of_change'],
            'co2_accumulation_24h': co2_stats['accumulation'],
            'co2_time_above_600': co2_stats['time_above_600'],
            'light_cumulative_24h': light_stats['cumulative']
        }
    
    def _extract_psychrometric_features(self, temperature: float, 
                                        humidity: float) -> Dict[str, float]:
        """Extract 4 psychrometric features."""
        return {
            'dew_point': self.psychro.calculate_dew_point(temperature, humidity),
            'vapor_pressure_deficit': self.psychro.calculate_vapor_pressure_deficit(
                temperature, humidity
            ),
            'condensation_risk': self.psychro.calculate_condensation_risk(
                temperature, humidity
            ),
            'enthalpy': self.psychro.calculate_enthalpy(temperature, humidity)
        }
    
    def _extract_crop_weighted_features(self, temperature: float, humidity: float,
                                        co2: float, light: float,
                                        crop_params: Optional[CropPhysicsParams]
                                       ) -> Dict[str, float]:
        """Extract 4 crop-specific weighted features."""
        if crop_params is None:
            # Default weights if no crop specified
            T_weight, H_weight, CO2_weight, L_weight = 0.4, 0.3, 0.2, 0.1
        else:
            T_weight = crop_params.T_weight
            H_weight = crop_params.H_weight
            CO2_weight = crop_params.CO2_weight
            L_weight = crop_params.light_weight
        
        # Normalize values to 0-1 range for weighting
        temp_norm = (temperature + 5) / 40.0  # -5 to 35°C -> 0-1
        humidity_norm = humidity / 100.0
        co2_norm = min(co2 / 5000.0, 1.0)  # Cap at 5000
        light_norm = min(light / 10000.0, 1.0)  # Cap at 10000
        
        return {
            'temperature_weighted': temp_norm * T_weight,
            'humidity_weighted': humidity_norm * H_weight,
            'co2_weighted': co2_norm * CO2_weight,
            'light_weighted': light_norm * L_weight
        }
    
    def _extract_interaction_features(self, temperature: float, humidity: float,
                                      co2: float,
                                      crop_params: Optional[CropPhysicsParams]
                                     ) -> Dict[str, float]:
        """Extract 7 interaction features."""
        # Temperature-Humidity interaction
        temp_hum_interaction = temperature * humidity / 1000.0  # Scale down
        
        # Temperature-CO2 interaction
        temp_co2_interaction = temperature * (co2 / 1000.0)
        
        # Humidity-CO2 ratio (ventilation effectiveness)
        humidity_co2_ratio = humidity / max(co2, 1.0) * 100.0
        
        # Binary flags
        high_temp_low_humidity = int(temperature > 15 and humidity < 80)
        low_temp_high_humidity = int(temperature < 5 and humidity > 95)
        anaerobic_risk = int(co2 > 5000 and humidity > 95)
        
        # Stress index (composite deviation from optimal)
        stress_index = 0.0
        if crop_params:
            T_opt_mid = (crop_params.T_optimal[0] + crop_params.T_optimal[1]) / 2
            H_opt_mid = (crop_params.H_optimal[0] + crop_params.H_optimal[1]) / 2
            CO2_opt_mid = (crop_params.CO2_optimal[0] + crop_params.CO2_optimal[1]) / 2
            
            # Weighted deviation from optimum
            T_dev = abs(temperature - T_opt_mid) / 10.0 * crop_params.T_weight
            H_dev = abs(humidity - H_opt_mid) / 20.0 * crop_params.H_weight
            CO2_dev = abs(co2 - CO2_opt_mid) / 1000.0 * crop_params.CO2_weight
            
            stress_index = T_dev + H_dev + CO2_dev
        else:
            # Generic stress calculation
            stress_index = (
                abs(temperature - 10) / 15.0 * 0.4 +
                abs(humidity - 90) / 30.0 * 0.3 +
                max(0, (co2 - 500) / 2000.0) * 0.3
            )
        
        return {
            'temperature_humidity_interaction': temp_hum_interaction,
            'temperature_co2_interaction': temp_co2_interaction,
            'humidity_co2_ratio': humidity_co2_ratio,
            'high_temp_low_humidity_flag': high_temp_low_humidity,
            'low_temp_high_humidity_flag': low_temp_high_humidity,
            'anaerobic_risk_flag': anaerobic_risk,
            'stress_index': min(1.0, stress_index)
        }
    
    def _extract_infrastructure_features(self, 
                                         sensor_reading: Dict) -> Dict[str, float]:
        """Extract 5 infrastructure features."""
        # Door cycle frequency (cycles per hour)
        door_cycles = sensor_reading.get('door_cycles_today', 0)
        door_frequency = door_cycles / 24.0  # Approximate hourly rate
        
        # Door cycle stress (cumulative impact)
        baseline = self.config.baseline_door_cycles_per_day
        max_cycles = self.config.max_door_cycles_per_day
        door_stress = max(0, (door_cycles - baseline) / (max_cycles - baseline))
        
        # Energy anomaly
        energy_anomaly = sensor_reading.get('energy_anomaly_score', 0.0)
        
        # Compressor runtime
        compressor_runtime = sensor_reading.get('compressor_duty_cycle', 50.0)
        
        # Pressure stability
        pressure_stability = sensor_reading.get('pressure_stability', 1.0)
        
        return {
            'door_cycle_frequency': door_frequency,
            'door_cycle_stress': min(1.0, door_stress),
            'energy_anomaly_score': min(1.0, energy_anomaly),
            'compressor_runtime_pct': compressor_runtime,
            'pressure_stability_score': pressure_stability
        }
    
    def _extract_quality_features(self, temperature: float, humidity: float,
                                  co2: float,
                                  crop_params: Optional[CropPhysicsParams]
                                 ) -> Dict[str, float]:
        """Extract 3 sensor quality features."""
        anomaly_flags = []
        
        # Temperature jump detection
        if self.prev_temperature is not None:
            temp_jump = abs(temperature - self.prev_temperature)
            if temp_jump > self.config.temperature_jump_threshold:
                anomaly_flags.append(0.5)
            else:
                anomaly_flags.append(0.0)
        
        # Humidity jump detection
        if self.prev_humidity is not None:
            hum_jump = abs(humidity - self.prev_humidity)
            if hum_jump > self.config.humidity_jump_threshold:
                anomaly_flags.append(0.3)
            else:
                anomaly_flags.append(0.0)
        
        # CO2 jump detection
        if self.prev_co2 is not None:
            co2_jump = abs(co2 - self.prev_co2)
            if co2_jump > self.config.co2_jump_threshold:
                anomaly_flags.append(0.2)
            else:
                anomaly_flags.append(0.0)
        
        # Composite anomaly score
        sensor_anomaly = sum(anomaly_flags) if anomaly_flags else 0.0
        
        # Data quality score (inverse of anomaly - higher is better)
        data_quality = max(0.0, 1.0 - sensor_anomaly)
        
        # Physical feasibility check
        feasibility = 1.0
        if crop_params:
            # Check if values are within reasonable range for crop
            if temperature < crop_params.chilling_threshold - 10:
                feasibility *= 0.7  # Unlikely but possible
            if temperature > crop_params.heat_threshold + 15:
                feasibility *= 0.5  # Very unusual
            if humidity < 10 or humidity > 100:
                feasibility *= 0.3  # Sensor error likely
        
        return {
            'sensor_anomaly_composite': sensor_anomaly,
            'data_quality_score': data_quality,
            'feasibility_score': feasibility
        }
    
    def extract_batch_features(self, 
                              df: pd.DataFrame,
                              crop_column: str = 'crop_type') -> pd.DataFrame:
        """
        Extract features for entire DataFrame.
        
        Args:
            df: DataFrame with sensor readings
            crop_column: Column name containing crop type
            
        Returns:
            DataFrame with all 39 engineered features
        """
        all_features = []
        
        # Reset buffer for batch processing
        self.temporal_buffer.reset()
        self.prev_temperature = None
        self.prev_humidity = None
        self.prev_co2 = None
        
        for idx, row in df.iterrows():
            crop_type = row.get(crop_column) if crop_column in df.columns else None
            
            features = self.extract_features(row.to_dict(), crop_type)
            all_features.append(features)
        
        feature_df = pd.DataFrame(all_features)
        
        # Preserve original columns not in features
        preserve_cols = ['sample_id', 'crop_type', 'quality_status', 'quality_class',
                        'quality_index', 'data_source', 'scenario_id']
        for col in preserve_cols:
            if col in df.columns:
                feature_df[col] = df[col].values
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of all 39 feature names."""
        return self.FEATURE_NAMES.copy()
    
    def reset(self):
        """Reset temporal buffer and previous readings."""
        self.temporal_buffer.reset()
        self.prev_temperature = None
        self.prev_humidity = None
        self.prev_co2 = None


class FeaturePipeline:
    """
    Complete feature engineering pipeline for ML training and inference.
    
    Provides methods for:
    - Training data preparation
    - Inference-time feature extraction
    - Feature standardization/normalization
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.engineer = FeatureEngineer(config)
        
        # Feature statistics for normalization
        self.feature_means: Optional[Dict[str, float]] = None
        self.feature_stds: Optional[Dict[str, float]] = None
        self.is_fitted = False
    
    def fit(self, training_df: pd.DataFrame):
        """
        Fit pipeline on training data to compute normalization statistics.
        """
        # Extract features
        feature_df = self.engineer.extract_batch_features(training_df)
        
        # Compute statistics for numeric features
        numeric_features = [f for f in self.engineer.FEATURE_NAMES 
                          if f in feature_df.columns]
        
        self.feature_means = {}
        self.feature_stds = {}
        
        for feature in numeric_features:
            self.feature_means[feature] = float(feature_df[feature].mean())
            self.feature_stds[feature] = float(feature_df[feature].std())
            
            # Prevent division by zero
            if self.feature_stds[feature] == 0:
                self.feature_stds[feature] = 1.0
        
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Transform DataFrame to engineered features.
        
        Args:
            df: Input DataFrame with sensor readings
            normalize: Whether to apply standardization
            
        Returns:
            DataFrame with 39 engineered features
        """
        # Extract features
        feature_df = self.engineer.extract_batch_features(df)
        
        # Normalize if requested and fitted
        if normalize and self.is_fitted:
            feature_df = self._normalize(feature_df)
        
        return feature_df
    
    def fit_transform(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, normalize)
    
    def _normalize(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        df_norm = feature_df.copy()
        
        for feature in self.feature_means.keys():
            if feature in df_norm.columns:
                mean = self.feature_means[feature]
                std = self.feature_stds[feature]
                df_norm[feature] = (df_norm[feature] - mean) / std
        
        return df_norm
    
    def get_feature_matrix(self, df: pd.DataFrame, 
                          normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Get feature matrix suitable for ML model input.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_df = self.transform(df, normalize)
        
        # Select only the 39 features
        features = [f for f in self.engineer.FEATURE_NAMES 
                   if f in feature_df.columns]
        
        X = feature_df[features].values
        
        return X, features
    
    def transform_single(self, sensor_reading: Dict, 
                        crop_type: str = None,
                        normalize: bool = True) -> np.ndarray:
        """
        Transform single sensor reading for real-time inference.
        
        Returns:
            1D numpy array with 39 features
        """
        features = self.engineer.extract_features(sensor_reading, crop_type)
        
        if normalize and self.is_fitted:
            for name, value in features.items():
                if name in self.feature_means:
                    mean = self.feature_means[name]
                    std = self.feature_stds[name]
                    features[name] = (value - mean) / std
        
        # Convert to array in consistent order
        feature_array = np.array([
            features.get(name, 0.0) for name in self.engineer.FEATURE_NAMES
        ])
        
        return feature_array
    
    def save(self, filepath: str):
        """Save fitted pipeline to disk."""
        import json
        
        state = {
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'is_fitted': self.is_fitted,
            'config': {
                'temporal_lookback_hours': self.config.temporal_lookback_hours,
                'timestep_minutes': self.config.timestep_minutes
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, filepath: str):
        """Load fitted pipeline from disk."""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.feature_means = state['feature_means']
        self.feature_stds = state['feature_stds']
        self.is_fitted = state['is_fitted']
    
    def reset(self):
        """Reset internal state."""
        self.engineer.reset()


def create_feature_pipeline(training_df: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None) -> FeaturePipeline:
    """
    Factory function to create and optionally fit a feature pipeline.
    """
    pipeline = FeaturePipeline()
    
    if training_df is not None:
        pipeline.fit(training_df)
        
        if save_path:
            pipeline.save(save_path)
    
    return pipeline


if __name__ == "__main__":
    print("AgriSense Feature Engineering Pipeline Demo")
    print("=" * 50)
    
    # Create sample sensor reading
    sample_reading = {
        'temperature': 8.5,
        'humidity': 92.0,
        'co2_ppm': 550,
        'light_lux': 15,
        'door_cycles_today': 6,
        'compressor_duty_cycle': 45.0,
        'energy_anomaly_score': 0.1,
        'pressure_stability': 0.95
    }
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Add some history for temporal features
    for i in range(50):
        temp = 8.0 + np.random.uniform(-0.5, 0.5)
        hum = 91.0 + np.random.uniform(-2, 2)
        co2 = 500 + np.random.uniform(-50, 50)
        light = 10 + np.random.uniform(0, 20)
        engineer.temporal_buffer.add_reading(temp, hum, co2, light)
    
    # Extract features
    features = engineer.extract_features(sample_reading, crop_type='avocado')
    
    print(f"\nExtracted {len(features)} features:")
    print("-" * 40)
    
    for name, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    print(f"\nFeature names: {engineer.get_feature_names()}")
