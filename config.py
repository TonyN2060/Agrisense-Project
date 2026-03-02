"""
AgriSense ML Module - Configuration Management

Centralized configuration for all components.
Supports environment variables for production deployment.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# Environment Detection
# =============================================================================

class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Get current environment from ENV variable."""
    env_str = os.getenv("AGRISENSE_ENV", "development").lower()
    return Environment(env_str) if env_str in [e.value for e in Environment] else Environment.DEVELOPMENT


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """ML model configuration."""
    # Model architecture
    model_type: str = "lightgbm"  # lightgbm, xgboost, catboost
    use_hierarchical: bool = True
    use_ensemble: bool = True
    bootstrap_replicates: int = 10
    
    # Optuna optimization
    optuna_trials: int = 100
    optuna_timeout: int = 3600  # seconds
    
    # LightGBM hyperparameters (defaults)
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 300,
        'verbose': -1,
        'random_state': 42
    })
    
    # Feature engineering
    num_features: int = 39
    temporal_window_hours: int = 24
    
    # Performance targets
    target_accuracy: float = 0.92
    target_f1: float = 0.85
    confidence_interval: float = 0.08  # ±8%
    
    # Model persistence
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "models"))
    model_version: Optional[str] = None


# =============================================================================
# Physics Engine Configuration
# =============================================================================

@dataclass
class PhysicsConfig:
    """Physics-based model configuration."""
    # Q10 kinetics
    enable_q10: bool = True
    q10_base_temperature: float = 20.0
    
    # Microbial growth
    enable_gompertz: bool = True
    microbial_threshold: float = 1e7  # CFU/g
    
    # Transpiration
    enable_transpiration: bool = True
    max_water_loss_percent: float = 15.0
    
    # CO2 respiration
    enable_respiration: bool = True
    co2_anaerobic_threshold: float = 5000.0  # ppm


# =============================================================================
# Digital Twin Configuration
# =============================================================================

@dataclass
class DigitalTwinConfig:
    """Digital twin simulation configuration."""
    # Cold room dimensions
    room_volume_m3: float = 50.0
    max_capacity_kg: float = 1000.0
    
    # Infrastructure
    compressor_power_kw: float = 5.0
    insulation_r_value: float = 25.0
    
    # Scenarios
    num_temperature_scenarios: int = 8
    num_humidity_scenarios: int = 6
    num_co2_scenarios: int = 5
    simulation_hours: int = 168  # 1 week
    time_step_minutes: int = 15
    
    # Data generation
    samples_per_scenario: int = 4


# =============================================================================
# Synthetic Data Configuration
# =============================================================================

@dataclass
class SyntheticDataConfig:
    """Synthetic data generation configuration."""
    # Sample targets
    total_samples: int = 162500
    digital_twin_samples: int = 120000
    tvae_samples: int = 40000
    real_samples: int = 2500
    
    # Class distribution
    class_distribution: Dict[str, float] = field(default_factory=lambda: {
        'GOOD': 0.30,
        'MARGINAL': 0.25,
        'AT_RISK': 0.20,
        'CRITICAL': 0.15,
        'SPOILED': 0.10
    })
    
    # TVAE settings
    tvae_epochs: int = 300
    tvae_batch_size: int = 500
    tvae_embedding_dim: int = 128
    tvae_compress_dims: Tuple[int, ...] = (128, 128)
    tvae_decompress_dims: Tuple[int, ...] = (128, 128)
    
    # Validation thresholds
    ks_threshold: float = 0.15
    correlation_threshold: float = 0.8
    class_imbalance_threshold: float = 0.15


# =============================================================================
# API Configuration
# =============================================================================

@dataclass
class APIConfig:
    """FastAPI service configuration."""
    # Server
    host: str = field(default_factory=lambda: os.getenv("UVICORN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("UVICORN_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("UVICORN_WORKERS", "2")))
    
    # JWT
    jwt_secret_key: str = field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", "agrisense-secret-change-me")
    )
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    max_batch_size: int = 1000
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    prediction_cache_size: int = 1000
    
    # Performance targets
    target_latency_ms: int = 100
    target_p95_latency_ms: int = 200


# =============================================================================
# Database Configuration
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration."""
    # PostgreSQL
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", 
            "postgresql://agrisense:agrisense@localhost:5432/agrisense"
        )
    )
    pool_size: int = 10
    max_overflow: int = 20
    
    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    redis_max_connections: int = 10


# =============================================================================
# Monitoring Configuration
# =============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    # Drift detection
    drift_window_size: int = 500
    drift_f1_threshold: float = 0.05
    drift_kl_threshold: float = 0.1
    
    # Incremental learning
    enable_incremental_learning: bool = True
    min_samples_for_retrain: int = 200
    max_samples_for_retrain: int = 5000
    retrain_cooldown_hours: int = 6
    
    # Metrics
    enable_prometheus: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "json"  # json, plain


# =============================================================================
# Crop Configuration
# =============================================================================

@dataclass
class CropConfig:
    """Supported crop configuration."""
    supported_crops: List[str] = field(default_factory=lambda: [
        'avocado', 'mango', 'leafy_greens', 'orange', 'berries'
    ])
    
    # Quality status mapping
    quality_levels: List[str] = field(default_factory=lambda: [
        'GOOD', 'MARGINAL', 'AT_RISK', 'CRITICAL', 'SPOILED'
    ])
    
    # Legacy mapping (from original dataset)
    legacy_crop_mapping: Dict[str, str] = field(default_factory=lambda: {
        'tomato': 'leafy_greens',
        'banana': 'mango',
        'pineapple': 'mango',
        'grape': 'berries',
        'strawberry': 'berries'
    })


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class AgriSenseConfig:
    """Master configuration combining all components."""
    environment: Environment = field(default_factory=get_environment)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    digital_twin: DigitalTwinConfig = field(default_factory=DigitalTwinConfig)
    synthetic_data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    
    @classmethod
    def from_env(cls) -> 'AgriSenseConfig':
        """Create configuration from environment variables."""
        return cls()
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


# =============================================================================
# Global Configuration Instance
# =============================================================================

_config: Optional[AgriSenseConfig] = None


def get_config() -> AgriSenseConfig:
    """Get global configuration instance (singleton)."""
    global _config
    if _config is None:
        _config = AgriSenseConfig.from_env()
    return _config


def reload_config() -> AgriSenseConfig:
    """Reload configuration from environment."""
    global _config
    _config = AgriSenseConfig.from_env()
    return _config


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: AgriSenseConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Check JWT secret
    if config.api.jwt_secret_key == "agrisense-secret-change-me":
        warnings.append("WARNING: Using default JWT secret key. Set JWT_SECRET_KEY in production!")
    
    # Check model directory
    if not os.path.exists(config.model.model_dir):
        warnings.append(f"Model directory does not exist: {config.model.model_dir}")
    
    # Check environment
    if config.is_production():
        if config.api.workers < 2:
            warnings.append("WARNING: Production should use at least 2 workers")
    
    return warnings


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import json
    
    config = get_config()
    warnings = validate_config(config)
    
    print("AgriSense Configuration")
    print("=" * 50)
    print(f"Environment: {config.environment.value}")
    print(f"Model Directory: {config.model.model_dir}")
    print(f"API Host: {config.api.host}:{config.api.port}")
    print(f"Workers: {config.api.workers}")
    
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
