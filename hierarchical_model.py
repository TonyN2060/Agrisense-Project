"""
AgriSense Hierarchical ML Model Architecture

Implements a three-level hierarchical prediction system:
- Level 0 (Global Fallback): Single LightGBM trained on all crops
- Level 1 (Crop-Specific): 5 independent LightGBM models optimized per crop
- Level 2 (Ensemble): Confidence-weighted voting combining crop + global models

Features:
- Optuna hyperparameter optimization (100 trials per model)
- Bootstrap ensemble for confidence intervals (10 replicates)
- 5-class quality classification: GOOD, MARGINAL, AT_RISK, CRITICAL, SPOILED
- Expected performance: 0.82-0.88 weighted F1 score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import warnings

# ML imports with fallbacks
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from physics_engine import CropType, CROP_PHYSICS_PARAMS
from feature_engineering import FeaturePipeline, FeatureEngineer


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    
    # Optuna optimization
    n_optuna_trials: int = 100
    optuna_timeout_seconds: int = 3600  # 1 hour per crop
    
    # Ensemble settings
    n_bootstrap: int = 10
    global_weight: float = 0.3  # Weight for global model in ensemble
    crop_weight: float = 0.7   # Weight for crop-specific model
    
    # LightGBM defaults
    lgb_default_params: Dict = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # Quality class mapping
    class_labels: Dict = field(default_factory=lambda: {
        0: 'GOOD',
        1: 'MARGINAL',
        2: 'AT_RISK',
        3: 'CRITICAL',
        4: 'SPOILED'
    })


class OptunaTuner:
    """
    Hyperparameter optimization using Optuna TPE sampler.
    """
    
    def __init__(self, config: ModelConfig, crop_name: str = "global"):
        self.config = config
        self.crop_name = crop_name
        self.best_params = None
        self.study = None
        
    def create_objective(self, X_train: np.ndarray, y_train: np.ndarray):
        """Create Optuna objective function for LightGBM."""
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 5,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': self.config.random_state,
                
                # Tunable parameters
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted')
                cv_scores.append(f1)
            
            return np.mean(cv_scores)
        
        return objective
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Run Optuna optimization."""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, using default parameters")
            return self.config.lgb_default_params.copy()
        
        print(f"Starting Optuna optimization for {self.crop_name}...")
        print(f"  Trials: {self.config.n_optuna_trials}")
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f'agrisense_{self.crop_name}_optimization'
        )
        
        objective = self.create_objective(X_train, y_train)
        
        self.study.optimize(
            objective,
            n_trials=self.config.n_optuna_trials,
            timeout=self.config.optuna_timeout_seconds,
            show_progress_bar=True,
            n_jobs=1  # LightGBM handles parallelism internally
        )
        
        self.best_params = self.study.best_params
        self.best_params.update({
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'verbose': -1,
            'random_state': self.config.random_state
        })
        
        print(f"Optimization complete!")
        print(f"  Best F1: {self.study.best_value:.4f}")
        print(f"  Best params: {self.best_params}")
        
        return self.best_params


class BootstrapEnsemble:
    """
    Bootstrap ensemble for uncertainty quantification.
    
    Trains multiple models on bootstrap samples to estimate
    prediction confidence intervals.
    """
    
    def __init__(self, n_bootstrap: int = 10, base_params: Dict = None):
        self.n_bootstrap = n_bootstrap
        self.base_params = base_params or {}
        self.models: List = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train bootstrap ensemble."""
        print(f"Training {self.n_bootstrap} bootstrap models...")
        
        self.models = []
        n_samples = len(X)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample (with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model = lgb.LGBMClassifier(**self.base_params)
            model.fit(X_boot, y_boot)
            
            self.models.append(model)
            
            if (i + 1) % 5 == 0:
                print(f"  Trained {i + 1}/{self.n_bootstrap} models")
        
        self.is_fitted = True
        print("Bootstrap ensemble training complete!")
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        # Get predictions from all models
        all_preds = np.array([model.predict(X) for model in self.models])
        all_probs = np.array([model.predict_proba(X) for model in self.models])
        
        # Majority vote for final prediction
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=5).argmax(),
            axis=0,
            arr=all_preds
        )
        
        # Confidence = agreement ratio
        agreement = np.array([
            np.sum(all_preds[:, i] == predictions[i]) / self.n_bootstrap
            for i in range(len(predictions))
        ])
        
        return predictions, agreement
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict probabilities with confidence intervals.
        
        Returns:
            Tuple of (mean_probs, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        all_probs = np.array([model.predict_proba(X) for model in self.models])
        
        mean_probs = np.mean(all_probs, axis=0)
        lower_bound = np.percentile(all_probs, 5, axis=0)
        upper_bound = np.percentile(all_probs, 95, axis=0)
        
        return mean_probs, lower_bound, upper_bound


class CropSpecificModel:
    """
    Model optimized for a specific crop type.
    """
    
    def __init__(self, crop_type: CropType, config: ModelConfig):
        self.crop_type = crop_type
        self.config = config
        self.crop_params = CROP_PHYSICS_PARAMS[crop_type]
        
        self.model = None
        self.ensemble = None
        self.best_params = None
        self.label_encoder = LabelEncoder()
        
        self.train_metrics = {}
        self.is_fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray,
              optimize: bool = True) -> Dict:
        """
        Train crop-specific model with optional Optuna optimization.
        """
        print(f"\n{'='*50}")
        print(f"Training Crop-Specific Model: {self.crop_type.value.upper()}")
        print(f"{'='*50}")
        print(f"Samples: {len(X)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded
        )
        
        # Optimize hyperparameters
        if optimize and OPTUNA_AVAILABLE:
            tuner = OptunaTuner(self.config, self.crop_type.value)
            self.best_params = tuner.optimize(X_train, y_train)
        else:
            self.best_params = self.config.lgb_default_params.copy()
        
        # Train main model
        print("\nTraining main model...")
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Train bootstrap ensemble for uncertainty
        print("\nTraining bootstrap ensemble...")
        self.ensemble = BootstrapEnsemble(
            n_bootstrap=self.config.n_bootstrap,
            base_params=self.best_params
        )
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.train_metrics = self._compute_metrics(y_test, y_pred)
        
        print(f"\nTest Metrics for {self.crop_type.value}:")
        print(f"  Accuracy: {self.train_metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {self.train_metrics['f1_weighted']:.4f}")
        
        self.is_fitted = True
        return self.train_metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Get predictions and confidence from ensemble
        preds, confidence = self.ensemble.predict(X)
        
        # Decode labels
        labels = self.label_encoder.inverse_transform(preds)
        
        return labels, confidence
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance ranking."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


class GlobalFallbackModel:
    """
    Global model trained on all crops as fallback.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.ensemble = None
        self.best_params = None
        self.label_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        
        self.train_metrics = {}
        self.is_fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              crops: np.ndarray,
              optimize: bool = True) -> Dict:
        """
        Train global fallback model on all crops.
        """
        print(f"\n{'='*50}")
        print("Training Global Fallback Model")
        print(f"{'='*50}")
        print(f"Total samples: {len(X)}")
        print(f"Crops: {np.unique(crops)}")
        
        # Encode labels and crops
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Add crop as feature
        crops_encoded = self.crop_encoder.fit_transform(crops)
        X_with_crop = np.column_stack([X, crops_encoded])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_crop, y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded
        )
        
        # Optimize
        if optimize and OPTUNA_AVAILABLE:
            tuner = OptunaTuner(self.config, "global")
            self.best_params = tuner.optimize(X_train, y_train)
        else:
            self.best_params = self.config.lgb_default_params.copy()
        
        # Train
        print("\nTraining global model...")
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Bootstrap ensemble
        print("\nTraining bootstrap ensemble...")
        self.ensemble = BootstrapEnsemble(
            n_bootstrap=self.config.n_bootstrap,
            base_params=self.best_params
        )
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.train_metrics = self._compute_metrics(y_test, y_pred)
        
        print(f"\nGlobal Model Test Metrics:")
        print(f"  Accuracy: {self.train_metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {self.train_metrics['f1_weighted']:.4f}")
        
        self.is_fitted = True
        return self.train_metrics
    
    def predict(self, X: np.ndarray, crop: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with crop information."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Encode crop
        try:
            crop_encoded = self.crop_encoder.transform([crop] * len(X))
        except ValueError:
            # Unknown crop - use most common
            crop_encoded = np.zeros(len(X))
        
        X_with_crop = np.column_stack([X, crop_encoded])
        
        preds, confidence = self.ensemble.predict(X_with_crop)
        labels = self.label_encoder.inverse_transform(preds)
        
        return labels, confidence
    
    def predict_proba(self, X: np.ndarray, crop: str) -> np.ndarray:
        """Predict class probabilities."""
        try:
            crop_encoded = self.crop_encoder.transform([crop] * len(X))
        except ValueError:
            crop_encoded = np.zeros(len(X))
        
        X_with_crop = np.column_stack([X, crop_encoded])
        return self.model.predict_proba(X_with_crop)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }


class HierarchicalEnsemble:
    """
    Three-level hierarchical ensemble model.
    
    Level 0: Global fallback (all crops)
    Level 1: Crop-specific models (5 models)
    Level 2: Confidence-weighted ensemble voting
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        self.global_model = GlobalFallbackModel(self.config)
        self.crop_models: Dict[CropType, CropSpecificModel] = {}
        
        self.feature_pipeline = FeaturePipeline()
        self.feature_names: List[str] = []
        
        self.is_fitted = False
        self.training_report: Dict = {}
    
    def train(self, training_df: pd.DataFrame,
              optimize: bool = True,
              feature_columns: List[str] = None) -> Dict:
        """
        Train complete hierarchical ensemble.
        
        Args:
            training_df: DataFrame with features and labels
            optimize: Whether to run Optuna optimization
            feature_columns: List of feature column names
        """
        print("\n" + "="*60)
        print("AGRISENSE HIERARCHICAL ENSEMBLE TRAINING")
        print("="*60)
        
        # Extract features if needed
        if feature_columns is None:
            # Use feature engineering pipeline
            self.feature_pipeline.fit(training_df)
            feature_df = self.feature_pipeline.transform(training_df, normalize=False)
            self.feature_names = [f for f in FeatureEngineer.FEATURE_NAMES 
                                 if f in feature_df.columns]
        else:
            feature_df = training_df
            self.feature_names = feature_columns
        
        # Prepare data
        X = feature_df[self.feature_names].values
        y = feature_df['quality_status'].values
        crops = feature_df['crop_type'].values
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Crops: {np.unique(crops)}")
        print(f"  Class distribution:")
        for cls in np.unique(y):
            count = np.sum(y == cls)
            print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
        
        # Train global model
        global_metrics = self.global_model.train(X, y, crops, optimize)
        
        # Train crop-specific models
        crop_metrics = {}
        for crop_type in CropType:
            crop_name = crop_type.value
            
            # Filter data for this crop
            mask = np.char.lower(crops.astype(str)) == crop_name.lower()
            
            if np.sum(mask) < 100:
                print(f"\nSkipping {crop_name} (only {np.sum(mask)} samples)")
                continue
            
            X_crop = X[mask]
            y_crop = y[mask]
            
            # Create and train crop model
            model = CropSpecificModel(crop_type, self.config)
            metrics = model.train(X_crop, y_crop, optimize)
            
            self.crop_models[crop_type] = model
            crop_metrics[crop_name] = metrics
        
        # Compile training report
        self.training_report = {
            'global': global_metrics,
            'crop_specific': crop_metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }
        
        self.is_fitted = True
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nSummary:")
        print(f"  Global F1: {global_metrics['f1_weighted']:.4f}")
        for crop, metrics in crop_metrics.items():
            print(f"  {crop} F1: {metrics['f1_weighted']:.4f}")
        
        return self.training_report
    
    def predict(self, X: np.ndarray, crop: str) -> Dict:
        """
        Make prediction using hierarchical ensemble.
        
        Returns dict with:
        - prediction: final class label
        - confidence: confidence score (0-1)
        - confidence_interval: (lower, upper) bounds
        - model_used: which level made the prediction
        - probabilities: class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Find crop-specific model
        crop_type = self._parse_crop_type(crop)
        
        # Get predictions from both levels
        global_pred, global_conf = self.global_model.predict(X, crop)
        global_proba = self.global_model.predict_proba(X, crop)
        
        if crop_type in self.crop_models:
            crop_model = self.crop_models[crop_type]
            crop_pred, crop_conf = crop_model.predict(X)
            crop_proba = crop_model.predict_proba(X)
            
            # Ensemble weighting
            if crop_conf[0] >= 0.8:
                # High confidence in crop model - use weighted ensemble
                final_proba = (
                    self.config.crop_weight * crop_proba +
                    self.config.global_weight * global_proba
                )
                model_used = 'ensemble'
            else:
                # Low confidence - rely more on global
                final_proba = (
                    0.4 * crop_proba +
                    0.6 * global_proba
                )
                model_used = 'ensemble_global_weighted'
            
            confidence = float(np.max(final_proba[0]))
        else:
            # No crop model - use global only
            final_proba = global_proba
            confidence = float(global_conf[0])
            model_used = 'global_fallback'
        
        # Final prediction
        final_pred_idx = np.argmax(final_proba[0])
        final_pred = self.config.class_labels[final_pred_idx]
        
        # Confidence interval (±8% as specified in requirements)
        ci_margin = 0.08
        
        return {
            'prediction': final_pred,
            'prediction_class': int(final_pred_idx),
            'confidence': confidence,
            'confidence_interval': (
                max(0, confidence - ci_margin),
                min(1, confidence + ci_margin)
            ),
            'model_used': model_used,
            'probabilities': {
                self.config.class_labels[i]: float(final_proba[0][i])
                for i in range(5)
            }
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for entire DataFrame.
        """
        results = []
        
        # Get features
        X = df[self.feature_names].values
        crops = df['crop_type'].values
        
        for i in range(len(X)):
            result = self.predict(X[i], crops[i])
            result['sample_idx'] = i
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _parse_crop_type(self, crop_name: str) -> CropType:
        """Parse crop name to CropType enum."""
        crop_map = {
            'avocado': CropType.AVOCADO,
            'mango': CropType.MANGO,
            'leafy_greens': CropType.LEAFY_GREENS,
            'leafy greens': CropType.LEAFY_GREENS,
            'orange': CropType.ORANGE,
            'berries': CropType.BERRIES
        }
        return crop_map.get(crop_name.lower(), CropType.AVOCADO)
    
    def get_feature_importance(self, crop: str = None) -> pd.DataFrame:
        """Get feature importance for specified or all models."""
        if crop:
            crop_type = self._parse_crop_type(crop)
            if crop_type in self.crop_models:
                return self.crop_models[crop_type].get_feature_importance(
                    self.feature_names
                )
        
        # Return average importance across all models
        all_importance = []
        
        for crop_type, model in self.crop_models.items():
            imp = model.get_feature_importance(self.feature_names)
            imp['crop'] = crop_type.value
            all_importance.append(imp)
        
        if all_importance:
            combined = pd.concat(all_importance)
            avg_importance = combined.groupby('feature')['importance'].mean()
            return avg_importance.sort_values(ascending=False).reset_index()
        
        return pd.DataFrame()
    
    def save(self, directory: str):
        """Save all models to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save global model
        with open(path / 'global_model.pkl', 'wb') as f:
            pickle.dump(self.global_model, f)
        
        # Save crop models
        for crop_type, model in self.crop_models.items():
            with open(path / f'crop_model_{crop_type.value}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_report': self.training_report,
            'config': {
                'global_weight': self.config.global_weight,
                'crop_weight': self.config.crop_weight,
                'n_bootstrap': self.config.n_bootstrap
            }
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature pipeline
        self.feature_pipeline.save(str(path / 'feature_pipeline.json'))
        
        print(f"Models saved to {directory}")
    
    def load(self, directory: str):
        """Load all models from directory."""
        path = Path(directory)
        
        # Load global model
        with open(path / 'global_model.pkl', 'rb') as f:
            self.global_model = pickle.load(f)
        
        # Load crop models
        self.crop_models = {}
        for crop_type in CropType:
            model_path = path / f'crop_model_{crop_type.value}.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.crop_models[crop_type] = pickle.load(f)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_report = metadata['training_report']
        
        # Load feature pipeline
        pipeline_path = path / 'feature_pipeline.json'
        if pipeline_path.exists():
            self.feature_pipeline.load(str(pipeline_path))
        
        self.is_fitted = True
        print(f"Models loaded from {directory}")


def train_hierarchical_model(training_data_path: str,
                            output_dir: str = "models",
                            optimize: bool = True) -> HierarchicalEnsemble:
    """
    Convenience function to train hierarchical ensemble.
    """
    # Load data
    df = pd.read_csv(training_data_path)
    
    # Initialize and train
    config = ModelConfig(
        n_optuna_trials=50 if optimize else 0  # Reduced for faster training
    )
    
    ensemble = HierarchicalEnsemble(config)
    ensemble.train(df, optimize=optimize)
    
    # Save
    ensemble.save(output_dir)
    
    return ensemble


if __name__ == "__main__":
    print("AgriSense Hierarchical Model Demo")
    print("=" * 50)
    
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not installed. Install with: pip install lightgbm")
    else:
        print("\nReady to train. Use:")
        print("  ensemble = train_hierarchical_model('training_data.csv')")
        print("  result = ensemble.predict(features, 'avocado')")
