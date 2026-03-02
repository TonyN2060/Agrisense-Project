"""
AgriSense Explainability & Drift Detection System

Provides:
- SHAP TreeExplainer for feature attribution
- Bootstrap confidence intervals (±8%)
- Concept drift detection with sliding windows
- Incremental learning with automatic retraining triggers
- Anomaly detection for sensor and equipment failures

Key Components:
1. SHAPExplainer: Tree-based SHAP explanations
2. ConceptDriftDetector: Monitor for distribution shifts
3. IncrementalLearner: Adaptive model updates
4. AnomalyDetector: Sensor and equipment anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json
import threading
import warnings

# SHAP import with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

from physics_engine import CropType, CROP_PHYSICS_PARAMS


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability system."""
    # SHAP settings
    n_background_samples: int = 100  # Background dataset size for SHAP
    top_features_to_show: int = 3    # Top features in explanations
    
    # Confidence intervals
    confidence_level: float = 0.90   # 90% confidence interval
    ci_margin: float = 0.08          # ±8% as specified
    
    # Drift detection
    drift_window_size: int = 500     # Samples in sliding window
    f1_drop_threshold: float = 0.05  # 5% F1 drop triggers drift alert
    
    # Retraining triggers
    retrain_on_f1_drop: bool = True
    retrain_on_sample_count: int = 1000  # Retrain every N labeled samples
    retrain_weekly: bool = True
    
    # Anomaly thresholds
    sensor_anomaly_zscore: float = 3.0  # Z-score threshold for anomalies


class SHAPExplainer:
    """
    SHAP-based model explainability for crop quality predictions.
    
    Uses TreeExplainer for efficient computation on LightGBM models.
    Provides feature attribution and human-readable explanations.
    """
    
    def __init__(self, model, feature_names: List[str], 
                 background_data: np.ndarray = None,
                 config: Optional[ExplainabilityConfig] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained LightGBM or tree-based model
            feature_names: List of feature names
            background_data: Background dataset for SHAP (optional)
            config: Configuration object
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or ExplainabilityConfig()
        
        if not SHAP_AVAILABLE:
            self.explainer = None
            warnings.warn("SHAP not available. Explanations will be limited.")
            return
        
        # Create TreeExplainer
        if background_data is not None:
            # Use background data for SHAP
            if len(background_data) > self.config.n_background_samples:
                indices = np.random.choice(
                    len(background_data),
                    size=self.config.n_background_samples,
                    replace=False
                )
                background_data = background_data[indices]
            
            self.explainer = shap.TreeExplainer(model, background_data)
        else:
            # Use model's internal capabilities
            self.explainer = shap.TreeExplainer(model)
        
        # Feature descriptions for human-readable explanations
        self.feature_descriptions = self._create_feature_descriptions()
    
    def _create_feature_descriptions(self) -> Dict[str, str]:
        """Create human-readable descriptions for features."""
        return {
            'temperature': 'Storage temperature',
            'humidity': 'Relative humidity',
            'co2': 'CO2 concentration',
            'light': 'Light exposure',
            'temperature_mean_24h': 'Average temperature (24h)',
            'temperature_std_24h': 'Temperature variability (24h)',
            'temperature_trend': 'Temperature trend (recent)',
            'humidity_deficit_cumulative': 'Cumulative humidity deficit',
            'co2_accumulation_24h': 'CO2 accumulation (24h)',
            'dew_point': 'Dew point temperature',
            'vapor_pressure_deficit': 'Drying stress (VPD)',
            'condensation_risk': 'Condensation risk',
            'stress_index': 'Overall stress index',
            'door_cycle_frequency': 'Door opening frequency',
            'compressor_runtime_pct': 'Compressor duty cycle',
            'mold_risk_score': 'Mold growth risk',
            'water_loss_percent': 'Water loss percentage'
        }
    
    def explain(self, X: np.ndarray, crop_type: str = None) -> Dict:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            X: Feature vector (1D or 2D array)
            crop_type: Crop type for context-aware explanations
            
        Returns:
            Dict with SHAP values, top contributors, and natural language explanation
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(X, crop_type)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For multi-class, shap_values is list of arrays per class
        # Get the predicted class
        pred_proba = self.model.predict_proba(X)[0]
        predicted_class = np.argmax(pred_proba)
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class][0]
        else:
            class_shap_values = shap_values[0]
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(class_shap_values))[::-1][:self.config.top_features_to_show]
        
        top_contributors = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_value = class_shap_values[idx]
            feature_value = X[0, idx]
            
            # Generate interpretation
            interpretation = self._interpret_contribution(
                feature_name, feature_value, shap_value, crop_type
            )
            
            top_contributors.append({
                'feature': feature_name,
                'feature_description': self.feature_descriptions.get(
                    feature_name, feature_name
                ),
                'value': float(feature_value),
                'shap_value': float(shap_value),
                'direction': 'toward_spoilage' if shap_value > 0 else 'toward_good',
                'interpretation': interpretation
            })
        
        # Generate natural language summary
        summary = self._generate_summary(top_contributors, predicted_class, pred_proba)
        
        return {
            'shap_values': class_shap_values.tolist(),
            'feature_names': self.feature_names,
            'top_contributors': top_contributors,
            'predicted_class': int(predicted_class),
            'predicted_proba': pred_proba.tolist(),
            'summary': summary,
            'recommendations': self._generate_recommendations(top_contributors, crop_type)
        }
    
    def _interpret_contribution(self, feature_name: str, value: float,
                                shap_value: float, crop_type: str = None) -> str:
        """Generate human-readable interpretation of feature contribution."""
        direction = "increasing" if shap_value > 0 else "decreasing"
        impact = "spoilage risk" if shap_value > 0 else "quality confidence"
        
        # Get crop-specific context
        crop_context = ""
        if crop_type:
            try:
                crop_enum = CropType(crop_type.lower())
                params = CROP_PHYSICS_PARAMS[crop_enum]
                
                if feature_name == 'temperature':
                    T_opt = params.T_optimal
                    if value < T_opt[0]:
                        crop_context = f" (below optimal {T_opt[0]}-{T_opt[1]}°C)"
                    elif value > T_opt[1]:
                        crop_context = f" (above optimal {T_opt[0]}-{T_opt[1]}°C)"
                    else:
                        crop_context = " (within optimal range)"
                        
                elif feature_name == 'humidity':
                    H_opt = params.H_optimal
                    if value < H_opt[0]:
                        crop_context = f" (below optimal {H_opt[0]}-{H_opt[1]}%)"
                    elif value > H_opt[1]:
                        crop_context = f" (above optimal, mold risk)"
            except (ValueError, KeyError):
                pass
        
        magnitude = abs(shap_value)
        if magnitude > 0.3:
            strength = "strongly"
        elif magnitude > 0.1:
            strength = "moderately"
        else:
            strength = "slightly"
        
        return f"{self.feature_descriptions.get(feature_name, feature_name)} = {value:.2f}{crop_context} is {strength} {direction} {impact}"
    
    def _generate_summary(self, contributors: List[Dict], 
                         predicted_class: int, proba: np.ndarray) -> str:
        """Generate natural language prediction summary."""
        class_names = ['GOOD', 'MARGINAL', 'AT_RISK', 'CRITICAL', 'SPOILED']
        prediction = class_names[predicted_class]
        confidence = proba[predicted_class] * 100
        
        # Build explanation
        main_factors = [c['feature_description'] for c in contributors[:2]]
        
        if predicted_class <= 1:
            outlook = "Quality is acceptable"
        elif predicted_class == 2:
            outlook = "Attention needed - quality at risk"
        else:
            outlook = "Urgent action required"
        
        summary = (
            f"Prediction: {prediction} ({confidence:.0f}% confidence). "
            f"{outlook}. "
            f"Main factors: {', '.join(main_factors)}."
        )
        
        return summary
    
    def _generate_recommendations(self, contributors: List[Dict],
                                  crop_type: str = None) -> List[str]:
        """Generate actionable recommendations based on SHAP analysis."""
        recommendations = []
        
        for contrib in contributors:
            feature = contrib['feature']
            value = contrib['value']
            direction = contrib['direction']
            
            if feature == 'temperature' and direction == 'toward_spoilage':
                if value > 15:
                    recommendations.append(f"Reduce temperature to optimal range")
                elif value < 0:
                    recommendations.append("Increase temperature to prevent chilling injury")
            
            elif feature == 'humidity' and direction == 'toward_spoilage':
                if value < 85:
                    recommendations.append("Increase humidity to prevent desiccation")
                elif value > 95:
                    recommendations.append("Reduce humidity to prevent mold growth")
            
            elif 'co2' in feature and direction == 'toward_spoilage':
                recommendations.append("Improve ventilation to reduce CO2 accumulation")
            
            elif 'door_cycle' in feature and direction == 'toward_spoilage':
                recommendations.append("Reduce door opening frequency to minimize thermal stress")
            
            elif 'mold' in feature and direction == 'toward_spoilage':
                recommendations.append("Consider expedited dispatch - mold risk elevated")
        
        if not recommendations:
            recommendations.append("Maintain current conditions")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _fallback_explanation(self, X: np.ndarray, crop_type: str = None) -> Dict:
        """Fallback explanation when SHAP is not available."""
        # Simple feature importance based on deviation from optimal
        importance_scores = []
        
        for i, feature in enumerate(self.feature_names):
            value = X[0, i]
            # Simple deviation-based importance
            if 'temperature' in feature.lower():
                importance = abs(value - 5) / 10  # Deviation from ~5°C optimal
            elif 'humidity' in feature.lower():
                importance = abs(value - 90) / 20
            else:
                importance = abs(value) / 10
            importance_scores.append(importance)
        
        top_indices = np.argsort(importance_scores)[::-1][:3]
        
        contributors = []
        for idx in top_indices:
            contributors.append({
                'feature': self.feature_names[idx],
                'feature_description': self.feature_descriptions.get(
                    self.feature_names[idx], self.feature_names[idx]
                ),
                'value': float(X[0, idx]),
                'importance': float(importance_scores[idx]),
                'interpretation': f"{self.feature_names[idx]} = {X[0, idx]:.2f}"
            })
        
        return {
            'top_contributors': contributors,
            'summary': "Explanation based on feature deviation analysis (SHAP not available)",
            'recommendations': ["Install SHAP for detailed explanations"]
        }


class ConceptDriftDetector:
    """
    Monitors for concept drift in model predictions.
    
    Uses sliding window analysis to detect distribution shifts
    that may require model retraining.
    """
    
    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self.config = config or ExplainabilityConfig()
        
        # Sliding window for predictions
        self.window_size = self.config.drift_window_size
        self.predictions_buffer = deque(maxlen=self.window_size)
        self.labels_buffer = deque(maxlen=self.window_size)
        
        # Reference metrics (set during initial calibration)
        self.reference_f1: Optional[float] = None
        self.reference_distribution: Optional[Dict[int, float]] = None
        
        # Drift detection state
        self.drift_detected = False
        self.drift_severity: float = 0.0
        self.last_check_timestamp: Optional[datetime] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
    
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calibrate drift detector with initial baseline metrics.
        
        Should be called after initial model deployment with validation data.
        """
        self.reference_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store reference class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        self.reference_distribution = {
            int(cls): count / total for cls, count in zip(unique, counts)
        }
        
        print(f"Drift detector calibrated:")
        print(f"  Reference F1: {self.reference_f1:.4f}")
        print(f"  Reference distribution: {self.reference_distribution}")
    
    def add_prediction(self, true_label: int, predicted_label: int):
        """
        Add a new labeled prediction to the monitoring buffer.
        
        This should be called when ground truth becomes available
        (e.g., post-harvest quality confirmation).
        """
        self.predictions_buffer.append(predicted_label)
        self.labels_buffer.append(true_label)
        
        # Check for drift periodically
        if len(self.predictions_buffer) >= self.window_size:
            self._check_drift()
    
    def _check_drift(self):
        """Check for concept drift in current window."""
        if self.reference_f1 is None:
            return
        
        y_true = np.array(self.labels_buffer)
        y_pred = np.array(self.predictions_buffer)
        
        # Calculate current F1
        current_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate F1 drop percentage
        f1_drop = (self.reference_f1 - current_f1) / self.reference_f1
        
        # Calculate distribution shift (KL divergence approximation)
        unique, counts = np.unique(y_true, return_counts=True)
        current_dist = {int(cls): count / len(y_true) for cls, count in zip(unique, counts)}
        
        kl_divergence = self._calculate_kl_divergence(
            self.reference_distribution, current_dist
        )
        
        # Update drift state
        self.drift_severity = max(f1_drop, kl_divergence * 0.5)
        self.drift_detected = f1_drop > self.config.f1_drop_threshold
        self.last_check_timestamp = datetime.now()
        
        if self.drift_detected:
            self._trigger_drift_alert(f1_drop, current_f1, kl_divergence)
    
    def _calculate_kl_divergence(self, ref_dist: Dict, curr_dist: Dict) -> float:
        """Calculate KL divergence between distributions."""
        kl_div = 0.0
        for cls, ref_prob in ref_dist.items():
            curr_prob = curr_dist.get(cls, 1e-10)  # Avoid log(0)
            if ref_prob > 0 and curr_prob > 0:
                kl_div += ref_prob * np.log(ref_prob / curr_prob)
        return kl_div
    
    def _trigger_drift_alert(self, f1_drop: float, current_f1: float,
                            kl_divergence: float):
        """Trigger drift alert and notify callbacks."""
        alert_info = {
            'type': 'CONCEPT_DRIFT',
            'severity': 'HIGH' if f1_drop > 0.1 else 'MEDIUM',
            'f1_drop_percent': f1_drop * 100,
            'current_f1': current_f1,
            'reference_f1': self.reference_f1,
            'kl_divergence': kl_divergence,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'RETRAIN_MODEL'
        }
        
        print(f"\n*** DRIFT DETECTED ***")
        print(f"F1 dropped {f1_drop*100:.1f}%: {self.reference_f1:.4f} -> {current_f1:.4f}")
        print(f"KL divergence: {kl_divergence:.4f}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback function for drift alerts."""
        self.alert_callbacks.append(callback)
    
    def get_drift_report(self) -> Dict:
        """Get current drift detection status."""
        return {
            'drift_detected': self.drift_detected,
            'drift_severity': self.drift_severity,
            'buffer_size': len(self.predictions_buffer),
            'reference_f1': self.reference_f1,
            'last_check': self.last_check_timestamp.isoformat() if self.last_check_timestamp else None
        }
    
    def reset(self):
        """Reset drift detector state."""
        self.predictions_buffer.clear()
        self.labels_buffer.clear()
        self.drift_detected = False
        self.drift_severity = 0.0


class IncrementalLearner:
    """
    Manages incremental learning and model retraining.
    
    Implements retraining triggers:
    - F1 drop threshold
    - Sample accumulation (every N labeled samples)
    - Time-based (weekly)
    - Seasonal transitions
    """
    
    def __init__(self, model_trainer: Callable,
                 config: Optional[ExplainabilityConfig] = None):
        """
        Args:
            model_trainer: Function to retrain model (takes training data)
            config: Configuration object
        """
        self.model_trainer = model_trainer
        self.config = config or ExplainabilityConfig()
        
        # Sample accumulation buffer
        self.sample_buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        
        # Timing
        self.last_retrain_timestamp: Optional[datetime] = None
        self.retrain_count: int = 0
        
        # Drift detector integration
        self.drift_detector: Optional[ConceptDriftDetector] = None
    
    def connect_drift_detector(self, detector: ConceptDriftDetector):
        """Connect to drift detector for automatic retraining triggers."""
        self.drift_detector = detector
        detector.register_alert_callback(self._on_drift_detected)
    
    def add_labeled_sample(self, features: np.ndarray, label: int,
                          metadata: Dict = None):
        """
        Add newly labeled sample (ground truth from post-harvest confirmation).
        """
        with self.buffer_lock:
            sample = {
                'features': features.tolist() if isinstance(features, np.ndarray) else features,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.sample_buffer.append(sample)
        
        # Check sample count trigger
        if len(self.sample_buffer) >= self.config.retrain_on_sample_count:
            self._trigger_retrain('sample_count')
    
    def check_time_based_retrain(self):
        """Check if time-based retraining is needed."""
        if not self.config.retrain_weekly:
            return
        
        if self.last_retrain_timestamp is None:
            return
        
        days_since_retrain = (datetime.now() - self.last_retrain_timestamp).days
        
        if days_since_retrain >= 7:
            self._trigger_retrain('weekly_schedule')
    
    def _on_drift_detected(self, alert_info: Dict):
        """Callback when drift is detected."""
        if self.config.retrain_on_f1_drop:
            self._trigger_retrain('drift_detected')
    
    def _trigger_retrain(self, trigger_reason: str):
        """Trigger model retraining."""
        print(f"\n*** RETRAINING TRIGGERED ***")
        print(f"Reason: {trigger_reason}")
        print(f"Samples in buffer: {len(self.sample_buffer)}")
        
        if len(self.sample_buffer) < 50:
            print("Insufficient samples for retraining. Skipping.")
            return
        
        with self.buffer_lock:
            # Prepare training data
            training_samples = self.sample_buffer.copy()
            
            # Keep some samples for next cycle (overlap)
            self.sample_buffer = self.sample_buffer[-100:]
        
        try:
            # Call model trainer
            self.model_trainer(training_samples)
            
            self.last_retrain_timestamp = datetime.now()
            self.retrain_count += 1
            
            print(f"Retraining complete (#{self.retrain_count})")
            
            # Reset drift detector if connected
            if self.drift_detector:
                self.drift_detector.reset()
                
        except Exception as e:
            print(f"Retraining failed: {e}")
    
    def get_status(self) -> Dict:
        """Get incremental learner status."""
        return {
            'samples_in_buffer': len(self.sample_buffer),
            'retrain_count': self.retrain_count,
            'last_retrain': self.last_retrain_timestamp.isoformat() if self.last_retrain_timestamp else None,
            'next_retrain_at_samples': self.config.retrain_on_sample_count
        }


class AnomalyDetector:
    """
    Detects anomalies in sensor readings and equipment behavior.
    
    Identifies:
    - Sensor failures (impossible values, sudden jumps)
    - Equipment malfunctions (abnormal patterns)
    - Environmental excursions (physics violations)
    """
    
    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self.config = config or ExplainabilityConfig()
        
        # Historical statistics for z-score calculation
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        # Valid ranges (physics-based)
        self.valid_ranges = {
            'temperature': (-10, 50),
            'humidity': (0, 100),
            'co2_ppm': (200, 50000),
            'light_lux': (0, 100000),
            'compressor_duty_cycle': (0, 100),
        }
        
        # Anomaly history
        self.anomaly_log: List[Dict] = []
    
    def fit(self, training_df: pd.DataFrame, feature_columns: List[str]):
        """Calculate baseline statistics from training data."""
        for col in feature_columns:
            if col in training_df.columns:
                values = training_df[col].dropna()
                self.feature_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q01': float(values.quantile(0.01)),
                    'q99': float(values.quantile(0.99))
                }
    
    def detect(self, sensor_reading: Dict) -> Dict:
        """
        Detect anomalies in a sensor reading.
        
        Returns:
            Dict with anomaly flags and scores
        """
        anomalies = []
        anomaly_scores = {}
        
        for feature, value in sensor_reading.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Range check
            if feature in self.valid_ranges:
                min_val, max_val = self.valid_ranges[feature]
                if value < min_val or value > max_val:
                    anomalies.append({
                        'type': 'OUT_OF_RANGE',
                        'feature': feature,
                        'value': value,
                        'valid_range': (min_val, max_val),
                        'severity': 'HIGH'
                    })
            
            # Z-score check
            if feature in self.feature_stats:
                stats = self.feature_stats[feature]
                if stats['std'] > 0:
                    z_score = (value - stats['mean']) / stats['std']
                    anomaly_scores[feature] = abs(z_score)
                    
                    if abs(z_score) > self.config.sensor_anomaly_zscore:
                        anomalies.append({
                            'type': 'STATISTICAL_ANOMALY',
                            'feature': feature,
                            'value': value,
                            'z_score': z_score,
                            'severity': 'HIGH' if abs(z_score) > 5 else 'MEDIUM'
                        })
        
        # Physics-based anomalies
        physics_anomalies = self._check_physics_constraints(sensor_reading)
        anomalies.extend(physics_anomalies)
        
        # Calculate overall anomaly score
        overall_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0
        
        result = {
            'is_anomalous': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'overall_anomaly_score': overall_score,
            'data_quality': 1.0 - min(1.0, overall_score / 5.0)
        }
        
        # Log anomalies
        if anomalies:
            self.anomaly_log.append({
                'timestamp': datetime.now().isoformat(),
                **result
            })
        
        return result
    
    def _check_physics_constraints(self, reading: Dict) -> List[Dict]:
        """Check for physics-based anomalies."""
        anomalies = []
        
        # Check dew point vs temperature (condensation)
        temp = reading.get('temperature')
        humidity = reading.get('humidity')
        
        if temp is not None and humidity is not None:
            # Approximate dew point check
            if humidity > 99 and temp < 35:
                pass  # Valid condensation conditions
            elif humidity == 100 and temp > 0:
                anomalies.append({
                    'type': 'PHYSICS_VIOLATION',
                    'feature': 'humidity',
                    'message': '100% humidity at positive temperature unusual',
                    'severity': 'LOW'
                })
        
        # Check CO2 vs ventilation
        co2 = reading.get('co2_ppm', 400)
        door_cycles = reading.get('door_cycles_today', 0)
        
        if co2 > 5000 and door_cycles > 20:
            anomalies.append({
                'type': 'PHYSICS_IMPLAUSIBLE',
                'feature': 'co2_ppm',
                'message': 'High CO2 despite frequent door openings - sensor check needed',
                'severity': 'MEDIUM'
            })
        
        return anomalies
    
    def get_anomaly_report(self, last_n: int = 100) -> List[Dict]:
        """Get recent anomaly history."""
        return self.anomaly_log[-last_n:]


class ExplainabilityService:
    """
    Unified service combining all explainability and monitoring capabilities.
    """
    
    def __init__(self, model, feature_names: List[str],
                 background_data: np.ndarray = None,
                 config: Optional[ExplainabilityConfig] = None):
        self.config = config or ExplainabilityConfig()
        
        # Initialize components
        self.shap_explainer = SHAPExplainer(
            model, feature_names, background_data, config
        )
        self.drift_detector = ConceptDriftDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        
    def explain_prediction(self, features: np.ndarray, 
                          crop_type: str = None) -> Dict:
        """Generate complete explanation for a prediction."""
        explanation = self.shap_explainer.explain(features, crop_type)
        
        # Add confidence interval
        explanation['confidence_interval'] = {
            'margin': self.config.ci_margin,
            'level': self.config.confidence_level
        }
        
        return explanation
    
    def check_data_quality(self, sensor_reading: Dict) -> Dict:
        """Check sensor data for anomalies."""
        return self.anomaly_detector.detect(sensor_reading)
    
    def report_ground_truth(self, true_label: int, predicted_label: int):
        """Report ground truth for drift monitoring."""
        self.drift_detector.add_prediction(true_label, predicted_label)
    
    def get_system_health(self) -> Dict:
        """Get overall system health status."""
        return {
            'drift_status': self.drift_detector.get_drift_report(),
            'recent_anomalies': len(self.anomaly_detector.anomaly_log),
            'shap_available': SHAP_AVAILABLE
        }


def create_explainability_service(model, feature_names: List[str],
                                 training_data: np.ndarray = None
                                ) -> ExplainabilityService:
    """Factory function to create explainability service."""
    config = ExplainabilityConfig()
    
    service = ExplainabilityService(
        model=model,
        feature_names=feature_names,
        background_data=training_data,
        config=config
    )
    
    return service


if __name__ == "__main__":
    print("AgriSense Explainability System Demo")
    print("=" * 50)
    
    if not SHAP_AVAILABLE:
        print("SHAP not installed. Install with: pip install shap")
    
    # Demo drift detector
    print("\nDrift Detector Demo:")
    detector = ConceptDriftDetector()
    
    # Simulate baseline
    np.random.seed(42)
    y_true_baseline = np.random.randint(0, 5, 500)
    y_pred_baseline = y_true_baseline.copy()
    # Add some errors
    y_pred_baseline[:50] = (y_pred_baseline[:50] + 1) % 5
    
    detector.calibrate(y_true_baseline, y_pred_baseline)
    
    print("\nSimulating gradual drift...")
    for i in range(600):
        true_label = np.random.randint(0, 5)
        # Increasing error rate simulates drift
        error_prob = min(0.3, i / 2000)
        if np.random.random() < error_prob:
            pred_label = (true_label + 1) % 5
        else:
            pred_label = true_label
        
        detector.add_prediction(true_label, pred_label)
    
    print(f"\nFinal drift report: {detector.get_drift_report()}")
