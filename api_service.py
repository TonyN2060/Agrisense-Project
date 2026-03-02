"""
AgriSense ML Prediction API Service

FastAPI-based REST API providing:
- Single prediction endpoint (/v1/predict)
- Batch prediction endpoint (/v1/predict_batch)
- SHAP explanation endpoint (/v1/explain)
- Model health/metrics endpoint (/v1/health)
- JWT-based multitenancy for coldroom/farm isolation

Performance Targets:
- Single prediction: <50ms
- Batch (1000 samples): <500ms
- P95 latency: <200ms
"""

import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
from pydantic import BaseModel, Field, validator

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")

# JWT handling
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("PyJWT not installed. Install with: pip install pyjwt")

# Internal imports
from physics_engine import CropType, CROP_PHYSICS_PARAMS, QualityStatus
from feature_engineering import FeaturePipeline, FeatureEngineer
from hierarchical_model import HierarchicalEnsemble, ModelConfig


# =============================================================================
# Configuration
# =============================================================================

class APIConfig:
    """API configuration settings."""
    # JWT Settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "agrisense-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Model paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    
    # Rate limiting
    MAX_BATCH_SIZE: int = 1000
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Caching
    PREDICTION_CACHE_SIZE: int = 1000
    CACHE_TTL_SECONDS: int = 300


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class SensorReading(BaseModel):
    """Single sensor reading input."""
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    co2_ppm: float = Field(default=400, ge=200, le=50000, description="CO2 in ppm")
    light_lux: float = Field(default=0, ge=0, le=100000, description="Light in lux")
    
    # Optional infrastructure sensors
    door_cycles_today: int = Field(default=0, ge=0, description="Door cycles today")
    compressor_duty_cycle: float = Field(default=50, ge=0, le=100, description="Compressor %")
    energy_anomaly_score: float = Field(default=0, ge=0, le=1, description="Energy anomaly")
    pressure_stability: float = Field(default=1.0, ge=0, le=1, description="Pressure stability")
    
    # Crop information
    crop_type: str = Field(..., description="Crop type (avocado, mango, leafy_greens, orange, berries)")
    
    # Optional batch/shipment ID
    batch_id: Optional[str] = None
    
    @validator('crop_type')
    def validate_crop_type(cls, v):
        valid_crops = ['avocado', 'mango', 'leafy_greens', 'leafy greens', 'orange', 'berries']
        if v.lower() not in valid_crops:
            raise ValueError(f"Invalid crop type. Must be one of: {valid_crops}")
        return v.lower().replace(' ', '_')


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str = Field(..., description="Quality status (GOOD, MARGINAL, AT_RISK, CRITICAL, SPOILED)")
    prediction_class: int = Field(..., description="Numeric class (0-4)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence bounds")
    
    # Remaining shelf life
    rsl_hours: Optional[float] = Field(None, description="Estimated remaining shelf life in hours")
    rsl_status: Optional[str] = Field(None, description="RSL status description")
    
    # Model metadata
    model_used: str = Field(..., description="Which model level made prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Request context
    crop_type: str
    batch_id: Optional[str] = None
    timestamp: str


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""
    prediction: str
    confidence: float
    
    top_contributors: List[Dict] = Field(..., description="Top contributing features")
    summary: str = Field(..., description="Natural language summary")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    
    # SHAP values (optional, can be large)
    shap_values: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    readings: List[SensorReading] = Field(..., max_items=1000, description="Sensor readings")
    include_explanations: bool = Field(default=False, description="Include SHAP explanations")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float
    summary: Dict[str, int]  # Count per status


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    drift_detected: bool
    recent_predictions: int
    uptime_seconds: float


class TokenRequest(BaseModel):
    """JWT token request."""
    tenant_id: str = Field(..., description="Tenant/coldroom identifier")
    api_key: str = Field(..., description="API key for authentication")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# JWT Authentication
# =============================================================================

class JWTHandler:
    """JWT token generation and validation."""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        
    def create_token(self, tenant_id: str, additional_claims: Dict = None) -> str:
        """Create JWT token for a tenant."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        payload = {
            'sub': tenant_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.config.JWT_EXPIRATION_HOURS),
            'tenant_id': tenant_id,
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(
            payload,
            self.config.JWT_SECRET_KEY,
            algorithm=self.config.JWT_ALGORITHM
        )
    
    def validate_token(self, token: str) -> Dict:
        """Validate JWT token and return payload."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed")
        
        try:
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")


# =============================================================================
# ML Service
# =============================================================================

class MLService:
    """
    Service wrapper for ML model operations.
    
    Handles:
    - Model loading and caching
    - Feature engineering
    - Prediction with confidence intervals
    - SHAP explanations
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.ensemble: Optional[HierarchicalEnsemble] = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        
        self.is_loaded = False
        self.model_version: Optional[str] = None
        self.load_timestamp: Optional[datetime] = None
        
        # Metrics
        self.prediction_count = 0
        self.total_latency_ms = 0.0
    
    def load_model(self):
        """Load model from disk."""
        try:
            self.ensemble = HierarchicalEnsemble()
            self.ensemble.load(self.model_dir)
            
            self.feature_engineer = FeatureEngineer()
            self.feature_pipeline = self.ensemble.feature_pipeline
            
            self.is_loaded = True
            self.load_timestamp = datetime.now()
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"Model loaded from {self.model_dir}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_loaded = False
    
    def predict(self, reading: SensorReading) -> Dict:
        """Make single prediction with timing."""
        start_time = time.time()
        
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to feature dict
        sensor_dict = reading.dict()
        
        # Extract features
        features = self.feature_engineer.extract_features(
            sensor_dict, 
            crop_type=reading.crop_type
        )
        
        # Convert to array
        feature_array = np.array([
            features.get(name, 0.0) 
            for name in self.feature_engineer.FEATURE_NAMES
        ]).reshape(1, -1)
        
        # Get prediction
        result = self.ensemble.predict(feature_array, reading.crop_type)
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        self.prediction_count += 1
        self.total_latency_ms += elapsed_ms
        
        # Estimate RSL based on prediction
        rsl_hours = self._estimate_rsl(result['prediction'], reading.crop_type)
        
        return {
            'prediction': result['prediction'],
            'prediction_class': result['prediction_class'],
            'confidence': result['confidence'],
            'confidence_interval': {
                'lower': result['confidence_interval'][0],
                'upper': result['confidence_interval'][1]
            },
            'rsl_hours': rsl_hours,
            'rsl_status': self._rsl_status_text(rsl_hours),
            'model_used': result['model_used'],
            'processing_time_ms': elapsed_ms,
            'crop_type': reading.crop_type,
            'batch_id': reading.batch_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, readings: List[SensorReading]) -> Dict:
        """Make batch predictions."""
        start_time = time.time()
        
        predictions = []
        status_counts = {
            'GOOD': 0, 'MARGINAL': 0, 'AT_RISK': 0, 
            'CRITICAL': 0, 'SPOILED': 0
        }
        
        for reading in readings:
            pred = self.predict(reading)
            predictions.append(pred)
            status_counts[pred['prediction']] += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'predictions': predictions,
            'total_samples': len(readings),
            'processing_time_ms': elapsed_ms,
            'summary': status_counts
        }
    
    def explain(self, reading: SensorReading) -> Dict:
        """Generate SHAP explanation for prediction."""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Get features
        sensor_dict = reading.dict()
        features = self.feature_engineer.extract_features(
            sensor_dict,
            crop_type=reading.crop_type
        )
        
        feature_array = np.array([
            features.get(name, 0.0)
            for name in self.feature_engineer.FEATURE_NAMES
        ]).reshape(1, -1)
        
        # Get prediction with explanation
        pred_result = self.ensemble.predict(feature_array, reading.crop_type)
        
        # Generate explanation (simplified without full SHAP)
        explanation = self._generate_explanation(
            features, 
            pred_result,
            reading.crop_type
        )
        
        return explanation
    
    def _generate_explanation(self, features: Dict, prediction: Dict, 
                             crop_type: str) -> Dict:
        """Generate feature-based explanation."""
        # Get crop parameters
        try:
            crop_enum = CropType(crop_type.lower())
            crop_params = CROP_PHYSICS_PARAMS[crop_enum]
        except (ValueError, KeyError):
            crop_params = None
        
        # Identify top contributors
        contributors = []
        
        # Temperature analysis
        temp = features.get('temperature', 20)
        if crop_params:
            T_opt = crop_params.T_optimal
            if temp < T_opt[0]:
                contributors.append({
                    'feature': 'temperature',
                    'value': temp,
                    'interpretation': f"Temperature {temp:.1f}°C below optimal ({T_opt[0]}-{T_opt[1]}°C)",
                    'direction': 'toward_spoilage'
                })
            elif temp > T_opt[1]:
                contributors.append({
                    'feature': 'temperature',
                    'value': temp,
                    'interpretation': f"Temperature {temp:.1f}°C above optimal ({T_opt[0]}-{T_opt[1]}°C)",
                    'direction': 'toward_spoilage'
                })
        
        # Humidity analysis
        humidity = features.get('humidity', 80)
        if crop_params:
            H_opt = crop_params.H_optimal
            if humidity < H_opt[0]:
                contributors.append({
                    'feature': 'humidity',
                    'value': humidity,
                    'interpretation': f"Humidity {humidity:.0f}% below optimal ({H_opt[0]}-{H_opt[1]}%)",
                    'direction': 'toward_spoilage'
                })
            elif humidity > H_opt[1]:
                contributors.append({
                    'feature': 'humidity',
                    'value': humidity,
                    'interpretation': f"Humidity {humidity:.0f}% above optimal (mold risk)",
                    'direction': 'toward_spoilage'
                })
        
        # Add stress index if high
        stress = features.get('stress_index', 0)
        if stress > 0.5:
            contributors.append({
                'feature': 'stress_index',
                'value': stress,
                'interpretation': f"Overall stress index elevated ({stress:.2f})",
                'direction': 'toward_spoilage'
            })
        
        # Generate recommendations
        recommendations = []
        if any(c['feature'] == 'temperature' for c in contributors):
            if temp < crop_params.T_optimal[0] if crop_params else 5:
                recommendations.append("Increase temperature to prevent chilling injury")
            else:
                recommendations.append("Reduce temperature to slow spoilage")
        
        if any(c['feature'] == 'humidity' for c in contributors):
            if humidity < 85:
                recommendations.append("Increase humidity to prevent desiccation")
            else:
                recommendations.append("Reduce humidity to prevent mold")
        
        if not recommendations:
            recommendations.append("Conditions within acceptable range - maintain settings")
        
        # Summary
        summary = (
            f"Prediction: {prediction['prediction']} "
            f"({prediction['confidence']*100:.0f}% confidence). "
            f"Key factors: {', '.join(c['feature'] for c in contributors[:2]) or 'within optimal ranges'}."
        )
        
        return {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'top_contributors': contributors[:3],
            'summary': summary,
            'recommendations': recommendations[:3],
            'feature_names': list(features.keys())[:10]
        }
    
    def _estimate_rsl(self, prediction: str, crop_type: str) -> float:
        """Estimate remaining shelf life based on prediction."""
        try:
            crop_enum = CropType(crop_type.lower())
            base_sl = CROP_PHYSICS_PARAMS[crop_enum].SL_ref
        except (ValueError, KeyError):
            base_sl = 200  # Default
        
        # Estimate RSL based on quality status
        rsl_factors = {
            'GOOD': 0.8,
            'MARGINAL': 0.5,
            'AT_RISK': 0.25,
            'CRITICAL': 0.1,
            'SPOILED': 0.0
        }
        
        factor = rsl_factors.get(prediction, 0.5)
        return base_sl * factor
    
    def _rsl_status_text(self, rsl_hours: float) -> str:
        """Generate RSL status text."""
        if rsl_hours <= 0:
            return "Expired - immediate action required"
        elif rsl_hours < 24:
            return f"Critical - {rsl_hours:.0f}h remaining"
        elif rsl_hours < 72:
            return f"Limited - {rsl_hours/24:.1f} days remaining"
        else:
            return f"Good - {rsl_hours/24:.1f} days remaining"
    
    def get_metrics(self) -> Dict:
        """Get service metrics."""
        avg_latency = (self.total_latency_ms / self.prediction_count 
                      if self.prediction_count > 0 else 0)
        
        return {
            'prediction_count': self.prediction_count,
            'average_latency_ms': avg_latency,
            'model_version': self.model_version,
            'uptime_seconds': (datetime.now() - self.load_timestamp).total_seconds() 
                             if self.load_timestamp else 0
        }


# =============================================================================
# FastAPI Application
# =============================================================================

# Global instances
ml_service = MLService()
jwt_handler = JWTHandler()
security = HTTPBearer() if FASTAPI_AVAILABLE else None
start_time = datetime.now()


@asynccontextmanager
async def lifespan(app):
    """Application lifespan handler."""
    # Startup
    print("Loading ML model...")
    ml_service.load_model()
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AgriSense ML Prediction API",
        description="Intelligent cold-chain quality prediction for agricultural produce",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None


# Dependency for JWT validation
async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """Validate JWT and extract tenant information."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization")
    
    token = credentials.credentials
    payload = jwt_handler.validate_token(token)
    
    return payload


# =============================================================================
# API Endpoints
# =============================================================================

if FASTAPI_AVAILABLE:
    
    @app.post("/v1/token", response_model=TokenResponse, tags=["Authentication"])
    async def create_token(request: TokenRequest):
        """
        Generate JWT access token.
        
        In production, validate api_key against database.
        """
        # TODO: Validate API key in production
        if len(request.api_key) < 10:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        token = jwt_handler.create_token(
            tenant_id=request.tenant_id,
            additional_claims={'permissions': ['predict', 'explain']}
        )
        
        return TokenResponse(
            access_token=token,
            expires_in=APIConfig.JWT_EXPIRATION_HOURS * 3600
        )
    
    
    @app.post("/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def predict(
        reading: SensorReading,
        tenant: Dict = Depends(get_current_tenant)
    ):
        """
        Make single quality prediction.
        
        Returns prediction with confidence interval and RSL estimate.
        Target latency: <50ms
        """
        result = ml_service.predict(reading)
        return PredictionResponse(**result)
    
    
    @app.post("/v1/predict_batch", response_model=BatchPredictionResponse, 
              tags=["Predictions"])
    async def predict_batch(
        request: BatchPredictionRequest,
        tenant: Dict = Depends(get_current_tenant)
    ):
        """
        Make batch predictions.
        
        Maximum 1000 samples per request.
        Target latency: <500ms for 1000 samples.
        """
        if len(request.readings) > APIConfig.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum batch size is {APIConfig.MAX_BATCH_SIZE}"
            )
        
        result = ml_service.predict_batch(request.readings)
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**p) for p in result['predictions']],
            total_samples=result['total_samples'],
            processing_time_ms=result['processing_time_ms'],
            summary=result['summary']
        )
    
    
    @app.post("/v1/explain", response_model=ExplanationResponse, tags=["Explainability"])
    async def explain(
        reading: SensorReading,
        tenant: Dict = Depends(get_current_tenant)
    ):
        """
        Generate SHAP-based explanation for prediction.
        
        Returns top contributing features and recommendations.
        """
        result = ml_service.explain(reading)
        return ExplanationResponse(**result)
    
    
    @app.get("/v1/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """
        Health check endpoint.
        
        Returns model status and metrics.
        """
        metrics = ml_service.get_metrics()
        uptime = (datetime.now() - start_time).total_seconds()
        
        return HealthResponse(
            status="healthy" if ml_service.is_loaded else "degraded",
            model_loaded=ml_service.is_loaded,
            model_version=ml_service.model_version,
            drift_detected=False,  # TODO: Connect to drift detector
            recent_predictions=metrics['prediction_count'],
            uptime_seconds=uptime
        )
    
    
    @app.get("/v1/crops", tags=["Reference"])
    async def list_crops():
        """
        List supported crop types with optimal conditions.
        """
        crops = []
        for crop_type in CropType:
            params = CROP_PHYSICS_PARAMS[crop_type]
            crops.append({
                'name': crop_type.value,
                'display_name': params.crop_name,
                'optimal_temperature': {
                    'min': params.T_optimal[0],
                    'max': params.T_optimal[1]
                },
                'optimal_humidity': {
                    'min': params.H_optimal[0],
                    'max': params.H_optimal[1]
                },
                'shelf_life_hours': params.SL_ref
            })
        
        return {'crops': crops}
    
    
    @app.get("/v1/metrics", tags=["System"])
    async def metrics(tenant: Dict = Depends(get_current_tenant)):
        """
        Get detailed service metrics.
        """
        return ml_service.get_metrics()


# =============================================================================
# Alternative prediction without auth (for testing)
# =============================================================================

if FASTAPI_AVAILABLE:
    
    @app.post("/v1/predict_noauth", response_model=PredictionResponse, 
              tags=["Testing"],
              include_in_schema=False)
    async def predict_noauth(reading: SensorReading):
        """Prediction endpoint without authentication (for testing only)."""
        result = ml_service.predict(reading)
        return PredictionResponse(**result)


# =============================================================================
# CLI Entry Point
# =============================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    import uvicorn
    
    print(f"Starting AgriSense ML API on {host}:{port}")
    uvicorn.run(
        "api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with:")
        print("  pip install fastapi uvicorn python-multipart")
    else:
        print("AgriSense ML API Service")
        print("=" * 50)
        print("\nStarting server...")
        print("API docs: http://localhost:8000/docs")
        print("Health: http://localhost:8000/v1/health")
        
        start_server(reload=True)
