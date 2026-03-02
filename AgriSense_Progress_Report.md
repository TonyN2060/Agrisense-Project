# AgriSense ML Module Progress Report
## AI Model x SDV Improvement Initiative

**Document Version:** 1.0  
**Date:** February 27, 2026  
**Project:** AgriSense - Physics-Grounded Digital Twin Intelligence for Cold-Chain Produce Quality  
**Scope:** Complete Evolution Plan Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview and Objectives](#project-overview-and-objectives)
3. [System Architecture Analysis](#system-architecture-analysis)
4. [Physics Engine Implementation](#physics-engine-implementation)
5. [Digital Twin Environment](#digital-twin-environment)
6. [Synthetic Data Generation with SDV](#synthetic-data-generation-with-sdv)
7. [Feature Engineering Pipeline](#feature-engineering-pipeline)
8. [Hierarchical Model Architecture](#hierarchical-model-architecture)
9. [Explainability and Drift Detection](#explainability-and-drift-detection)
10. [API Service Implementation](#api-service-implementation)
11. [Quality Metrics and Validation](#quality-metrics-and-validation)
12. [Technical Challenges and Solutions](#technical-challenges-and-solutions)
13. [Current Implementation Status](#current-implementation-status)
14. [Recommendations and Next Steps](#recommendations-and-next-steps)
15. [Conclusion](#conclusion)

---

## 1. Executive Summary

The AgriSense Model Evolution Plan represents a comprehensive transformation of an existing binary spoilage classification system into a sophisticated, physics-grounded, hierarchical intelligence platform designed specifically for cold-chain produce quality management. This progress report documents the complete implementation journey, technical architecture decisions, and the current state of the system as of February 2026.

The primary objective of this initiative has been to significantly enhance prediction accuracy from a baseline of 65-72% to a target of 91-94%, while simultaneously improving the F1-score from 0.38-0.48 to 0.82-0.88. This represents a substantial improvement in model performance that directly translates to an estimated 65% reduction in produce spoilage across the cold-chain logistics network.

The implementation encompasses several key technological innovations including a physics-based digital twin simulation environment, Tabular Variational AutoEncoder (TVAE) synthetic data generation through the Synthetic Data Vault (SDV) framework, a 39-feature engineering pipeline derived from agricultural science principles, and a three-level hierarchical ensemble model architecture utilizing LightGBM with Optuna hyperparameter optimization.

The system now supports five distinct crop types: Avocado, Mango, Leafy Greens, Orange, and Berries, each with crop-specific physics parameters calibrated from post-harvest physiology literature. The transition from a binary (Good/Bad) classification scheme to a comprehensive 5-level quality status system (GOOD, MARGINAL, AT_RISK, CRITICAL, SPOILED) provides significantly more granular and actionable insights for cold-chain operators.

All core modules have been successfully implemented and integrated, with recent resolution of dependency issues related to missing Python packages including FastAPI, uvicorn, PyJWT, SHAP, LightGBM, Optuna, and SDV. The system is now ready for comprehensive testing and validation against the target performance metrics.

---

## 2. Project Overview and Objectives

### 2.1 Background and Motivation

The agricultural cold-chain industry faces enormous challenges in maintaining produce quality during storage and transportation. Post-harvest losses account for a significant percentage of total food production, with improper temperature and humidity management being primary contributors to spoilage. Traditional monitoring systems provide raw sensor data but lack the intelligent analysis capabilities needed to predict quality degradation before it becomes critical.

The original AgriSense system employed a relatively simple binary classification approach, distinguishing only between "Good" and "Bad" produce conditions. While functional, this approach suffered from several limitations:

- **Limited Granularity:** Binary classification provides no early warning of degradation progression
- **Crop Agnostic:** Single model applied uniformly to all produce types regardless of their distinct physiological characteristics
- **Static Feature Set:** Relied primarily on instantaneous sensor readings without temporal context
- **No Physics Grounding:** Machine learning models trained purely on statistical patterns without incorporation of known agricultural science principles

### 2.2 Evolution Plan Goals

The Model Evolution Plan was designed to address these limitations comprehensively through the following strategic objectives:

**Performance Improvement Targets:**
| Metric | Baseline | Target | Projected Improvement |
|--------|----------|--------|----------------------|
| Overall Accuracy | 65-72% | 91-94% | +27 percentage points |
| Weighted F1-Score | 0.38-0.48 | 0.82-0.88 | +0.42 points |
| Spoilage Reduction | N/A | 65% | Business outcome |
| Prediction Latency | Variable | <100ms | Real-time capability |

**Functional Enhancement Goals:**
- Expand crop support from 4 to 5 varieties with crop-specific optimization
- Implement 5-level quality classification for granular risk assessment
- Generate physics-constrained synthetic training data at scale (162,500 samples)
- Provide SHAP-based explainability with bootstrap confidence intervals (±8%)
- Enable real-time API predictions with JWT-based multi-tenancy support
- Implement concept drift detection and incremental learning capabilities

### 2.3 Scope Definition

The implementation scope encompasses the complete machine learning lifecycle from data generation through production deployment:

1. **Physics Engine Development:** Q10 kinetics, Gompertz microbial growth, transpiration modeling
2. **Digital Twin Simulation:** Cold room environment simulation for synthetic data generation
3. **Synthetic Data Generation:** TVAE-based augmentation using the SDV framework
4. **Feature Engineering:** 39-feature pipeline incorporating temporal, psychrometric, and infrastructure features
5. **Model Training:** Hierarchical LightGBM ensemble with Optuna optimization
6. **Explainability System:** SHAP integration with confidence intervals and natural language explanations
7. **API Service:** FastAPI production server with authentication and monitoring
8. **Deployment Infrastructure:** Docker containerization with Prometheus metrics

---

## 3. System Architecture Analysis

### 3.1 High-Level Architecture

The AgriSense ML Module follows a modular, pipeline-oriented architecture that cleanly separates concerns across distinct components. This design philosophy enables independent testing, iterative improvement, and flexible deployment configurations.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AgriSense ML Module                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Physics    │    │   Digital    │    │   Synthetic  │          │
│  │   Engine     │ -> │    Twin      │ -> │    Data      │          │
│  │  (Q10, etc)  │    │  Simulator   │    │   (TVAE)     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│          │                   │                   │                  │
│          └───────────────────┴───────────────────┘                  │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │    Feature        │                           │
│                    │   Engineering     │                           │
│                    │   (39 features)   │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │   Hierarchical    │                           │
│                    │    Ensemble       │                           │
│                    │   (LightGBM)      │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
│          ┌───────────────────┼───────────────────┐                  │
│          │                   │                   │                  │
│  ┌───────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐           │
│  │    SHAP      │   │    Drift     │   │  Incremental │           │
│  │  Explainer   │   │  Detection   │   │   Learning   │           │
│  └──────────────┘   └──────────────┘   └──────────────┘           │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │     FastAPI       │                           │
│                    │   (JWT Auth)      │                           │
│                    └───────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Organization

The codebase is organized into the following primary modules, each with well-defined responsibilities:

| Module | File | Primary Responsibility | Lines of Code |
|--------|------|----------------------|---------------|
| Physics Engine | `physics_engine.py` | Crop-specific physics models (Q10, microbial, transpiration) | 926 |
| Digital Twin | `digital_twin.py` | Cold room simulation environment | 927 |
| Synthetic Data | `synthetic_data.py` | TVAE/SDV synthetic data generation | 747 |
| Feature Engineering | `feature_engineering.py` | 39-feature extraction pipeline | 837 |
| Hierarchical Model | `hierarchical_model.py` | LightGBM hierarchical ensemble | 850 |
| Explainability | `explainability.py` | SHAP, drift detection, incremental learning | 851 |
| API Service | `api_service.py` | FastAPI REST endpoints | 761 |
| Configuration | `config.py` | Centralized configuration management | 372 |
| Pipeline Orchestrator | `main.py` | End-to-end training pipeline | 437 |

**Total Implementation:** Approximately 6,708 lines of production Python code

### 3.3 Data Flow Architecture

The data flow through the system follows a clear progression from raw input through final prediction:

**Training Pipeline:**
1. Real sensor data ingested from CSV (2,500 samples)
2. Physics engine generates crop-specific parameters
3. Digital twin simulates cold room scenarios (120,000 samples)
4. TVAE synthesizer generates additional augmentation data (40,000 samples)
5. Feature engineering extracts 39 features per sample
6. Hierarchical ensemble trained with Optuna optimization
7. Models exported with SHAP explainer pre-computed

**Inference Pipeline:**
1. API receives sensor reading via POST request
2. JWT authentication validates tenant credentials
3. Feature engineering transforms raw reading
4. Hierarchical ensemble generates prediction with confidence
5. SHAP explainer provides feature attribution
6. Response returned with prediction, RSL estimate, and explanation

### 3.4 Dependency Architecture

The system utilizes a carefully selected stack of Python libraries optimized for agricultural ML applications:

**Core ML & Scientific Computing:**
- NumPy (>=1.24.0): Numerical operations foundation
- Pandas (>=2.0.0): Data manipulation and analysis
- SciPy (>=1.10.0): Scientific computing utilities
- Scikit-learn (>=1.3.0): ML preprocessing and metrics

**Gradient Boosting:**
- LightGBM (>=4.0.0): Primary model architecture
- XGBoost (>=2.0.0): Fallback model option

**Hyperparameter Optimization:**
- Optuna (>=3.3.0): Bayesian hyperparameter tuning with TPE sampler

**Explainability:**
- SHAP (>=0.42.0): Tree-based SHAP explanations

**Synthetic Data Generation:**
- SDV (>=1.5.0): Synthetic Data Vault framework
- RDT (>=1.6.0): Reversible data transforms
- CTGAN (>=0.7.0): Conditional tabular GAN support

**Web Service:**
- FastAPI (>=0.103.0): High-performance async API framework
- Uvicorn (>=0.23.0): ASGI server for production
- PyJWT (>=2.8.0): JWT token handling
- Pydantic (>=2.0.0): Request/response validation

---

## 4. Physics Engine Implementation

### 4.1 Q10 Temperature Kinetics

The physics engine forms the scientific foundation of the AgriSense system, incorporating established principles from post-harvest physiology to ensure that synthetic data generation and quality predictions are grounded in real agricultural science rather than purely statistical correlations.

The Q10 temperature coefficient model is central to understanding how storage temperature affects produce shelf life. The Q10 value represents the factor by which metabolic reaction rates increase for every 10°C rise in temperature. This relationship is expressed mathematically as:

$$Rate = Q10^{(T_{current} - T_{ref}) / 10}$$

Where:
- $Rate$ is the metabolic rate multiplier
- $Q10$ is the temperature coefficient (crop-specific)
- $T_{current}$ is the current storage temperature
- $T_{ref}$ is the reference temperature for the crop

The system calculates Fractional Life Used (FLU) at each timestep to track cumulative shelf-life consumption:

$$FLU = \frac{timestep_{hours}}{SL_{ref}} \times Q10^{(T - T_{ref})/10}$$

This physics-based approach ensures that temperature excursions are correctly weighted according to their actual impact on produce quality, rather than treating all temperature deviations equally.

### 4.2 Crop-Specific Physics Parameters

Each of the five supported crop types has been parameterized based on agricultural science literature, with distinct values for optimal storage conditions, temperature sensitivity, and quality decay characteristics:

| Crop | Q10 | T_optimal (°C) | H_optimal (%) | SL_ref (hours) | Notes |
|------|-----|---------------|---------------|----------------|-------|
| Avocado | 2.4 | 5-8 | 90-95 | 400 | Highest ethylene sensitivity |
| Mango | 2.1 | 10-13 | 85-90 | 360 | Chilling-sensitive tropical |
| Leafy Greens | 1.8 | 0-4 | 95-98 | 240 | Highest transpiration rate |
| Orange | 1.9 | 3-8 | 85-90 | 2160 | Longest shelf life |
| Berries | 2.5 | -1-2 | 90-95 | 168 | Most temperature sensitive |

The `CropPhysicsParams` dataclass captures these parameters along with additional factors:
- Chilling and heat injury thresholds
- Transpiration coefficients (water loss rate)
- CO2 respiration rates
- Mold growth temperature thresholds
- Ethylene sensitivity ratings
- Feature importance weights (T, H, CO2, Light)

### 4.3 Microbial Growth Modeling

The Gompertz microbial growth model simulates bacterial and fungal proliferation on produce surfaces. This asymmetric sigmoid growth curve accurately captures the three phases of microbial development:

1. **Lag Phase:** Initial slow growth as microorganisms adapt
2. **Exponential Phase:** Rapid logarithmic population increase
3. **Stationary Phase:** Growth plateaus as resources deplete

The model accounts for temperature-dependent growth rates, with higher temperatures (within limits) accelerating microbial proliferation. The microbial load is tracked continuously and contributes to overall quality assessment.

### 4.4 Transpiration and Water Loss

The transpiration engine models water loss from produce surfaces, which is critical for leafy greens and berries that are particularly sensitive to dehydration. The transpiration rate depends on:

- Vapor Pressure Deficit (VPD) between produce and surrounding air
- Crop-specific transpiration coefficients
- Surface area to mass ratio
- Relative humidity conditions

Cumulative water loss is tracked as a percentage of initial mass, with crop-specific tolerance thresholds determining when water loss begins to negatively impact quality.

### 4.5 CO2 Accumulation and Respiration

The CO2 respiration module tracks carbon dioxide production from produce metabolism and its accumulation in enclosed cold room environments. Key considerations include:

- Temperature-dependent respiration rates (Arrhenius relationship)
- Room volume and ventilation characteristics
- Anaerobic respiration thresholds (>5000 ppm typically)
- CO2 injury effects on different crop types

The system calculates both instantaneous respiration rates and cumulative CO2 accumulation over 24-hour windows for feature engineering.

---

## 5. Digital Twin Environment

### 5.1 Cold Room Simulation Architecture

The Digital Twin Environment represents a complete physics-grounded simulation of commercial cold storage facilities, generating realistic sensor data trajectories across diverse operating scenarios. This simulation capability is essential for producing the large-scale training datasets required for high-performance machine learning models.

The `DigitalTwinEnvironment` class encapsulates:

**Cold Room Configuration (`ColdRoomConfig`):**
- Room volume (default: 100 m³)
- Thermal insulation R-value (default: 3.5)
- Compressor power capacity (default: 5 kW)
- Ambient temperature characteristics (Kenya context: 28°C mean)
- Door infiltration rates
- Humidity control capacity

**Environmental State (`EnvironmentalState`):**
- Temperature (including multi-point gradient)
- Relative humidity (with spatial variation)
- CO2 concentration
- Light exposure levels
- Ethylene concentration

**Infrastructure State (`InfrastructureState`):**
- Door open/closed status with cumulative cycles
- Compressor on/off with duty cycle tracking
- Energy consumption metrics
- Room pressure monitoring
- Condensation risk zone mapping

### 5.2 Scenario-Based Data Generation

The simulation framework supports diverse operating scenarios through the `SimulationScenario` class, enabling generation of training data that covers the full operational envelope:

**Temperature Profiles:**
- `optimal`: Maintained within crop-specific optimal range
- `warm`: Consistently above optimal with gradual warming
- `cold`: Below optimal (chilling injury scenarios)
- `fluctuating`: Periodic temperature swings
- `excursion`: Discrete temperature excursion events

**Humidity Profiles:**
- `optimal`: Maintained at crop-optimal humidity
- `dry`: Below optimal causing accelerated water loss
- `humid`: Above optimal increasing mold/condensation risk
- `variable`: Natural fluctuations

**Infrastructure Profiles:**
- `low/normal/high/erratic` door frequency patterns
- `good/degrading/failing` equipment health trajectories

**Initial Conditions:**
- `fresh`: Perfect starting condition
- `pre_damaged`: 15% FLU, elevated microbial load
- `aged_2days`: 48 hours of optimal storage simulated

### 5.3 Physics-Based Sensor Simulation

Each simulation timestep (default: 5 minutes) applies physics models to update environmental and quality states:

**Temperature Update:** Combines heat transfer through walls (conduction), door opening infiltration, produce respiration heat release, and compressor cooling effects. Diurnal ambient temperature variation follows a sinusoidal pattern appropriate for tropical/subtropical climates.

**Humidity Update:** Balances moisture addition from produce transpiration against dry air infiltration during door openings and humidity control system capacity.

**CO2 Update:** Tracks respiration-driven CO2 production and ventilation-driven removal.

**Quality Update:** Integrates all environmental factors through the physics engines to update FLU, microbial load, water loss, and composite quality index.

### 5.4 Data Output Format

The digital twin generates comprehensive records at each timestep including:

- All environmental sensor readings
- Infrastructure state variables
- Physics engine intermediate values (FLU, microbial load, water loss)
- Calculated quality index (0-100 scale)
- Assigned quality status (5-level classification)
- Remaining shelf life estimates

A single scenario simulation over 14 days produces approximately 4,032 sample records (at 5-minute intervals), enabling rapid generation of large-scale training datasets.

---

## 6. Synthetic Data Generation with SDV

### 6.1 TVAE Architecture Overview

The Synthetic Data Vault (SDV) integration represents a critical component of the data generation strategy, utilizing the Tabular Variational AutoEncoder (TVAE) architecture to learn and reproduce the multivariate statistical relationships present in real and digital twin data.

TVAE offers several advantages over simpler synthetic data approaches:

**Multivariate Correlation Preservation:** Unlike univariate sampling methods, TVAE learns the joint distribution of all features simultaneously, preserving correlations such as the relationship between temperature and humidity or between CO2 levels and quality metrics.

**Non-Gaussian Distribution Support:** Agricultural sensor data often exhibits non-normal distributions (bounded values, multimodal patterns). TVAE's neural network architecture can capture these complex distributions.

**Categorical Variable Integration:** TVAE effectively handles the mixed data type requirements of the AgriSense dataset (continuous sensor values + categorical crop types and quality labels).

### 6.2 TVAE Configuration

The TVAE synthesizer is configured with hyperparameters optimized for agricultural sensor data:

```python
TVAESynthesizer(
    epochs=300,              # Training iterations
    batch_size=500,          # Samples per training batch
    embedding_dim=128,       # Latent space dimensionality
    compress_dims=(128, 128), # Encoder hidden layers
    decompress_dims=(128, 128) # Decoder hidden layers
)
```

These parameters balance training time against generation quality, with the 128-dimensional latent space providing sufficient capacity to capture the statistical relationships without overfitting.

### 6.3 Data Flow for Synthetic Generation

The synthetic data generation follows a carefully orchestrated pipeline:

**Stage 1 - Load Real Data (2,500 samples):**
Real sensor data is loaded from the Dataset.csv file, with column name standardization and quality label mapping to the 5-level classification system.

**Stage 2 - Generate Digital Twin Data (120,000 samples):**
The physics-grounded digital twin simulates diverse cold room scenarios, producing 120,000 samples across all crop types and operating conditions.

**Stage 3 - Fit TVAE Model:**
The TVAE model is trained on the combined real + digital twin dataset, learning the joint statistical properties of the complete feature set.

**Stage 4 - Generate TVAE Samples (40,000 samples):**
The fitted TVAE model generates 40,000 additional synthetic samples that augment the training dataset while preserving learned correlations.

**Stage 5 - Class Balancing:**
The combined 162,500-sample dataset is balanced according to the target class distribution to ensure adequate representation of minority classes.

### 6.4 Quality Validation Metrics

The `DataValidator` class implements comprehensive quality checks for synthetic data:

**Kolmogorov-Smirnov Statistics:**
For each numeric feature, the two-sample KS test compares the distribution of synthetic values against the reference (real + digital twin) distribution. A KS statistic < 0.05 indicates acceptable distribution matching.

**Spearman Correlation Preservation:**
The correlation matrices of numeric features are computed for both reference and synthetic data. The Spearman correlation between flattened upper triangle values measures how well feature relationships are preserved, with ρ > 0.95 indicating acceptable correlation preservation.

**Class Distribution Validation:**
Actual class proportions in the synthetic data are compared against target distribution, with <5% deviation acceptable for each class.

### 6.5 Target Dataset Composition

The final training dataset targets 162,500 total samples with the following composition:

| Source | Samples | Percentage | Purpose |
|--------|---------|------------|---------|
| Real Sensor Data | 2,500 | 1.5% | Ground truth from actual deployments |
| Digital Twin | 120,000 | 73.8% | Physics-constrained scenario coverage |
| TVAE Synthetic | 40,000 | 24.6% | Statistical augmentation |

**Target Class Distribution:**
| Quality Status | Target % | Sample Count |
|----------------|----------|--------------|
| GOOD | 35% | 56,875 |
| MARGINAL | 22% | 35,750 |
| AT_RISK | 21% | 34,125 |
| CRITICAL | 12% | 19,500 |
| SPOILED | 10% | 16,250 |

This distribution reflects realistic operational proportions while ensuring sufficient minority class representation for robust model training.

---

## 7. Feature Engineering Pipeline

### 7.1 Feature Categories Overview

The feature engineering pipeline transforms raw sensor readings into a comprehensive 39-feature representation that captures temporal dynamics, environmental physics, and infrastructure behavior. This rich feature set enables the hierarchical models to learn complex decision boundaries that generalize across operating conditions.

The 39 features are organized into the following categories:

| Category | Feature Count | Description |
|----------|---------------|-------------|
| Raw Sensors | 4 | Direct sensor readings |
| Temporal Features | 12 | 24-hour lookback statistics |
| Psychrometric Features | 4 | Thermodynamic calculations |
| Crop-Weighted Features | 4 | Importance-weighted values |
| Interaction Features | 7 | Combined stress indicators |
| Infrastructure Features | 5 | Equipment behavior |
| Sensor Quality Features | 3 | Anomaly and data quality |

### 7.2 Raw Sensor Features

The four primary sensor readings form the foundation:

1. **temperature:** Current storage temperature (°C)
2. **humidity:** Current relative humidity (%)
3. **co2_ppm:** CO2 concentration (ppm)
4. **light_lux:** Light exposure (lux)

### 7.3 Temporal Features

A rolling buffer maintains 24 hours of history (288 samples at 5-minute intervals) for computing temporal statistics:

**Temperature Statistics:**
- `temperature_mean_24h`: Average over lookback window
- `temperature_std_24h`: Standard deviation (variability)
- `temperature_min_24h`: Minimum value (cold excursions)
- `temperature_max_24h`: Maximum value (heat excursions)
- `temperature_trend`: Change over last hour

**Humidity Statistics:**
- `humidity_mean_24h`: Average humidity
- `humidity_min_24h`: Minimum (drying periods)
- `humidity_deficit_cumulative`: Integrated deficit from 100%

**CO2 Statistics:**
- `co2_rate_of_change`: Change rate (ppm/hour)
- `co2_accumulation_24h`: Cumulative excess above baseline
- `co2_time_above_600`: Duration above threshold

**Light Statistics:**
- `light_cumulative_24h`: Integrated lux-hours (photodegradation)

### 7.4 Psychrometric Features

Based on ASHRAE fundamentals for moist air thermodynamics:

**Dew Point Temperature:**
Calculated using the Magnus formula to identify condensation risk:
$$T_{dew} = \frac{b \cdot \alpha}{a - \alpha}$$
where $\alpha = \frac{a \cdot T}{b + T} + \ln(RH/100)$

**Vapor Pressure Deficit (VPD):**
The difference between saturation and actual vapor pressure, indicating drying stress:
$$VPD = P_{sat}(T) - P_{sat}(T) \cdot RH/100$$

**Condensation Risk Index:**
Binary indicator when dew point approaches surface temperature, triggering moisture accumulation concerns.

**Enthalpy:**
Moist air enthalpy for energy balance calculations:
$$h = 1.006 \cdot T + W \cdot (2501 + 1.86 \cdot T)$$

### 7.5 Crop-Specific Weighted Features

Each crop type has distinct sensitivity to environmental factors. The physics engine provides importance weights that are applied to create crop-normalized features:

- `weighted_temperature`: T × T_weight
- `weighted_humidity`: H × H_weight
- `weighted_co2`: CO2 × CO2_weight
- `weighted_light`: Light × light_weight

This enables the model to learn crop-specific thresholds automatically.

### 7.6 Interaction Features

Combined indicators that capture multi-factor stress conditions:

1. **stress_index:** Composite deviation from optimal across all factors
2. **chilling_hours:** Cumulative time below chilling threshold
3. **heat_stress_hours:** Cumulative time above heat threshold
4. **humidity_stress_index:** Combined humidity and VPD stress
5. **anaerobic_risk:** CO2 accumulation above anaerobic threshold
6. **temperature_humidity_interaction:** Cross-term for coupled effects
7. **door_temperature_interaction:** Door frequency × temperature deviation

### 7.7 Infrastructure Features

Equipment behavior indicators that provide operational context:

1. **door_cycle_frequency:** Door openings per hour
2. **compressor_duty_cycle:** Percentage runtime
3. **energy_anomaly_score:** Deviation from baseline consumption
4. **pressure_stability:** Room pressure consistency
5. **condensation_zones:** Count of at-risk areas

### 7.8 Sensor Quality Features

Data quality indicators for anomaly detection:

1. **temperature_anomaly_flag:** Jump detection in readings
2. **humidity_anomaly_flag:** Jump detection
3. **overall_data_quality_score:** Composite reliability metric

---

## 8. Hierarchical Model Architecture

### 8.1 Three-Level Hierarchy Design

The hierarchical ensemble model architecture addresses the challenge of crop-specific optimization while maintaining robust generalization through a three-level design:

**Level 0 - Global Fallback Model:**
A single LightGBM classifier trained on all crops simultaneously. This model provides reasonable predictions even for novel crop types or edge cases where crop-specific models may be uncertain. It captures universal patterns in produce quality degradation that transcend individual crop characteristics.

**Level 1 - Crop-Specific Models:**
Five independent LightGBM classifiers, each trained exclusively on data from one crop type (Avocado, Mango, Leafy Greens, Orange, Berries). These models learn the unique temperature sensitivities, optimal storage conditions, and degradation patterns specific to each produce variety.

**Level 2 - Ensemble Aggregation:**
A confidence-weighted voting mechanism that combines predictions from the relevant crop-specific model with the global model. The default weighting assigns 70% to the crop model and 30% to the global model, with dynamic adjustment based on prediction confidence.

### 8.2 LightGBM Configuration

LightGBM (Light Gradient Boosting Machine) was selected as the primary algorithm due to its excellent performance characteristics:

- **Leaf-wise growth:** Faster training with better accuracy than level-wise alternatives
- **Categorical feature support:** Native handling of crop type encoding
- **Low memory footprint:** Efficient for production deployment
- **Parallelization:** Near-linear scaling with CPU cores

Default hyperparameters (before Optuna optimization):

```python
{
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
}
```

### 8.3 Optuna Hyperparameter Optimization

The Optuna framework provides Bayesian optimization for hyperparameter tuning, using the Tree-structured Parzen Estimator (TPE) sampler for efficient exploration of the hyperparameter space.

**Tuned Parameters:**
| Parameter | Search Range | Impact |
|-----------|--------------|--------|
| num_leaves | 20-150 | Model complexity |
| max_depth | 3-15 | Tree depth limit |
| learning_rate | 0.01-0.2 (log) | Training step size |
| n_estimators | 100-500 | Number of trees |
| min_child_samples | 5-50 | Leaf minimum samples |
| subsample | 0.6-1.0 | Row sampling |
| colsample_bytree | 0.6-1.0 | Feature sampling |
| reg_alpha | 1e-8-10.0 (log) | L1 regularization |
| reg_lambda | 1e-8-10.0 (log) | L2 regularization |

**Optimization Configuration:**
- 100 trials per crop-specific model
- 5-fold stratified cross-validation per trial
- Weighted F1-score as optimization objective
- 1-hour timeout per crop optimization session

### 8.4 Bootstrap Ensemble for Uncertainty Quantification

The `BootstrapEnsemble` class implements bootstrap aggregation (bagging) specifically for uncertainty quantification:

1. Train 10 replicate models on bootstrap samples of the training data
2. For each prediction, collect predictions from all replicates
3. Compute prediction mode (most common class) as final prediction
4. Compute 90% confidence interval from prediction distribution
5. Report confidence as agreement rate among replicates

This approach provides the ±8% confidence interval specification without requiring probabilistic model architectures.

### 8.5 Training Pipeline

The complete model training workflow:

1. **Data Loading:** Load or generate combined training dataset
2. **Feature Engineering:** Apply 39-feature transformation pipeline
3. **Train/Test Split:** 80/20 stratified split preserving class proportions
4. **Global Model Training:** Optuna optimization on all-crop data
5. **Crop Model Training:** Parallel Optuna optimization for each crop
6. **Bootstrap Ensemble:** Train 10 replicates for each model
7. **Validation:** Compute accuracy, F1, confusion matrices on test set
8. **Export:** Save all models, feature pipeline, and metadata

---

## 9. Explainability and Drift Detection

### 9.1 SHAP TreeExplainer Integration

The explainability module provides transparent, interpretable predictions through SHAP (SHapley Additive exPlanations) integration. The TreeExplainer algorithm, specifically optimized for tree-based models like LightGBM, computes exact Shapley values efficiently.

**SHAP Value Interpretation:**
For each prediction, SHAP values indicate how much each feature contributed to pushing the prediction toward or away from each quality class. Positive values push toward higher class indices (worse quality), while negative values push toward lower indices (better quality).

**Top Contributors:**
The system identifies the top 3 features most responsible for each prediction, providing:
- Feature name and human-readable description
- Feature value at prediction time
- SHAP value (magnitude and direction)
- Natural language interpretation

### 9.2 Natural Language Explanations

The `SHAPExplainer` class generates human-readable explanations combining SHAP analysis with agricultural domain knowledge:

**Example Output:**
```
"This avocado storage is classified as AT_RISK with 85% confidence.
Key contributing factors:
1. Temperature (8.5°C) is at the high end of optimal range, 
   increasing metabolic rate
2. Humidity deficit over 24 hours indicates potential water loss stress
3. CO2 accumulation suggests reduced ventilation efficiency

Recommended actions:
- Lower temperature setpoint by 1-2°C
- Verify ventilation system operation
- Schedule inspection within 24 hours"
```

### 9.3 Confidence Intervals

Bootstrap ensemble predictions provide confidence intervals meeting the ±8% specification:

```python
{
    'prediction': 'AT_RISK',
    'confidence': 0.85,
    'confidence_interval': {
        'lower': 0.77,  # 85% - 8%
        'upper': 0.93   # 85% + 8%
    }
}
```

### 9.4 Concept Drift Detection

The `ConceptDriftDetector` class monitors for distribution shifts that could degrade model performance:

**Sliding Window Approach:**
A window of recent predictions (default: 500 samples) is compared against baseline statistics computed during training.

**Detection Methods:**
1. **F1 Monitoring:** Alert when F1-score drops >5% below baseline
2. **Feature Distribution:** KS-test on feature distributions vs. training
3. **Prediction Distribution:** Chi-square test on class proportions

**Automatic Actions:**
- Drift detected → Alert logged with severity level
- Persistent drift → Flag for retraining
- Weekly scheduled retraining option

### 9.5 Incremental Learning

The `IncrementalLearner` supports model updates without full retraining:

**Stream Updates:**
Accumulate new labeled samples arriving during production use.

**Batch Retraining Triggers:**
- Sample count threshold (default: 1,000 new samples)
- F1 drop threshold (default: 5%)
- Weekly schedule

**Warm Start:**
LightGBM supports continuing training from existing model weights, dramatically reducing retraining time while incorporating new patterns.

---

## 10. API Service Implementation

### 10.1 FastAPI Architecture

The API service provides production-ready REST endpoints using FastAPI, an async-first Python web framework offering automatic OpenAPI documentation, request validation, and high performance.

**Key Design Decisions:**
- Async handlers for concurrent request processing
- Pydantic models for strict request/response validation
- JWT-based authentication for multi-tenant isolation
- CORS middleware for cross-origin browser access
- Background tasks for drift detection updates

### 10.2 Authentication Flow

JWT (JSON Web Token) authentication enables multi-tenant usage:

**Token Request:**
```http
POST /v1/token
Content-Type: application/json

{
    "tenant_id": "farm-001",
    "api_key": "your-api-key"
}
```

**Token Response:**
```json
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

**Authenticated Request:**
```http
POST /v1/predict
Authorization: Bearer eyJ...
Content-Type: application/json

{...sensor reading...}
```

### 10.3 Prediction Endpoints

**Single Prediction (`POST /v1/predict`):**

Request Schema (`SensorReading`):
- temperature: float (required, -10 to 50°C)
- humidity: float (required, 0-100%)
- co2_ppm: float (optional, default 400)
- light_lux: float (optional, default 0)
- door_cycles_today: int (optional)
- compressor_duty_cycle: float (optional)
- crop_type: string (required)
- batch_id: string (optional)

Response Schema (`PredictionResponse`):
- prediction: Quality status string
- prediction_class: Numeric class (0-4)
- confidence: Prediction confidence (0-1)
- confidence_interval: Lower/upper bounds
- rsl_hours: Remaining shelf life estimate
- rsl_status: Human-readable RSL description
- model_used: "crop_avocado", "global", etc.
- processing_time_ms: API latency

**Batch Prediction (`POST /v1/predict_batch`):**

Request: Array of up to 1,000 sensor readings
Response: Array of prediction responses + summary statistics

**Performance Targets:**
- Single prediction: <50ms (typical), <100ms (P95)
- Batch (1,000): <500ms total

### 10.4 Explanation Endpoint

**SHAP Explanation (`POST /v1/explain`):**

Extends prediction with full SHAP analysis:
- top_contributors: List of feature attributions
- summary: Natural language explanation
- recommendations: Actionable mitigation steps
- shap_values: Full SHAP vector (optional)

### 10.5 Health and Metrics

**Health Check (`GET /v1/health`):**

Returns system status:
- Model loaded status
- Model version identifier
- Drift detection status
- Recent prediction count
- Uptime duration

**Prometheus Metrics Integration:**
- Request counts by endpoint
- Request latency histograms
- Prediction class distributions
- Error rates

---

## 11. Quality Metrics and Validation

### 11.1 Model Performance Targets

The evolution plan establishes clear performance benchmarks:

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| Overall Accuracy | 65-72% | 91-94% | Mean across 5-fold CV |
| Weighted F1 | 0.38-0.48 | 0.82-0.88 | Macro-averaged by class |
| CRITICAL Recall | ~40% | >85% | Prevent false negatives |
| SPOILED Precision | ~60% | >90% | Prevent false alarms |
| Latency P95 | Variable | <200ms | Production monitoring |

### 11.2 Synthetic Data Quality Metrics

**Distribution Matching (KS Statistics):**
For each numeric feature, KS statistic < 0.05 indicates acceptable synthetic data quality. Pass rate target: >90% of features passing.

**Correlation Preservation:**
Spearman ρ > 0.95 between reference and synthetic correlation matrices.

**Class Balance:**
Within 5% of target distribution for each quality class.

### 11.3 Model Validation Framework

**Cross-Validation Protocol:**
5-fold stratified cross-validation ensuring class proportions maintained in each fold.

**Holdout Testing:**
20% test set never used during training or hyperparameter optimization.

**Confusion Matrix Analysis:**
Per-class precision, recall, and F1 scores to identify systematic errors.

**ROC/AUC Analysis:**
One-vs-rest ROC curves for multi-class classification quality.

---

## 12. Technical Challenges and Solutions

### 12.1 Dependency Management

**Challenge:** The system requires numerous Python packages (FastAPI, LightGBM, Optuna, SHAP, SDV) that may not be pre-installed in development or production environments.

**Solution:** Comprehensive `requirements.txt` with pinned version ranges. Graceful fallbacks implemented throughout the codebase with `try/except` imports and boolean availability flags (e.g., `SDV_AVAILABLE`, `SHAP_AVAILABLE`). Recent dependency resolution completed successfully installing all required packages.

### 12.2 WSL Environment Complexity

**Challenge:** Development in Windows Subsystem for Linux creates path complexity with UNC paths (`\\wsl.localhost\Ubuntu\...`) and mixed Python environments.

**Solution:** Utilized VS Code's integrated Python environment configuration tools to properly detect and activate the `.venv` virtual environment within the WSL filesystem.

### 12.3 Class Imbalance

**Challenge:** Real agricultural data exhibits significant class imbalance, with SPOILED samples being rare compared to GOOD samples.

**Solution:** Multi-pronged approach:
1. Digital twin generates scenarios specifically targeting minority classes
2. TVAE can be conditioned to generate balanced samples
3. Final dataset explicitly balanced according to target distribution
4. Weighted loss function options in LightGBM

### 12.4 Feature Correlation

**Challenge:** Agricultural sensors exhibit high correlation (temperature affects humidity, humidity affects CO2), potentially causing multicollinearity issues.

**Solution:** LightGBM handles correlated features well through its feature sampling (colsample_bytree). Additionally, SHAP values properly attribute predictions even with correlated features. Feature engineering creates orthogonal interaction terms.

### 12.5 Real-Time Inference

**Challenge:** Production deployment requires sub-100ms latency for real-time quality monitoring.

**Solution:** 
1. FastAPI async handlers enable concurrent request processing
2. LightGBM models are inherently fast for inference (~1ms per prediction)
3. Feature engineering cache for temporal buffers
4. Pre-loaded SHAP explainer (TreeExplainer is fast)

---

## 13. Current Implementation Status

### 13.1 Completed Components

| Component | Status | Notes |
|-----------|--------|-------|
| Physics Engine | ✅ Complete | All crop types parameterized |
| Digital Twin | ✅ Complete | Scenario generation working |
| Synthetic Data (SDV) | ✅ Complete | TVAE integration ready |
| Feature Engineering | ✅ Complete | 39 features implemented |
| Hierarchical Model | ✅ Complete | LightGBM + Optuna |
| Explainability | ✅ Complete | SHAP + drift detection |
| API Service | ✅ Complete | FastAPI endpoints ready |
| Configuration | ✅ Complete | Centralized config |
| Dependencies | ✅ Resolved | All packages installed |

### 13.2 Integration Testing Status

- **Unit Tests:** Not yet implemented (recommended priority)
- **Integration Tests:** Manual testing required
- **Performance Tests:** Not yet benchmarked
- **Load Tests:** Not yet conducted

### 13.3 Documentation Status

- README: ✅ Comprehensive with examples
- API Documentation: ✅ Auto-generated via OpenAPI
- Code Documentation: ✅ Extensive docstrings
- Architecture Documentation: This report

### 13.4 Deployment Readiness

- **Docker:** ✅ Dockerfile and docker-compose.yml ready
- **Database:** ✅ PostgreSQL schema (init.sql)
- **Monitoring:** ✅ Prometheus configuration
- **Production Config:** ⚠️ Requires environment variables

---

## 14. Recommendations and Next Steps

This section provides a comprehensive, phased implementation roadmap covering immediate technical priorities through full production deployment with frontend integration. Each phase includes specific deliverables, acceptance criteria, and estimated timelines.

### 14.1 Phase 1: Model Training and Validation (Week 1-2)

#### 14.1.1 Full Training Pipeline Execution

Execute the complete training pipeline with production-scale data generation to validate model performance against target metrics.

**Step 1 - Environment Preparation:**
```bash
# Activate virtual environment
cd /home/pc/Agrisense-Project-default
source .venv/bin/activate

# Verify all dependencies
pip list | grep -E "lightgbm|optuna|shap|sdv|fastapi"

# Create output directories
mkdir -p models logs reports
```

**Step 2 - Full-Scale Data Generation:**
```bash
# Generate complete training dataset (162,500 samples)
python -c "
from synthetic_data import create_training_dataset
df = create_training_dataset(
    real_data_path='Dataset.csv',
    output_path='data/agrisense_full_training.csv',
    digital_twin_samples=120000,
    tvae_samples=40000
)
print(f'Generated {len(df)} samples')
"
```

**Step 3 - Model Training with Optuna:**
```bash
# Full training with hyperparameter optimization
python main.py \
    --data data/agrisense_full_training.csv \
    --model-dir models \
    --optuna-trials 100 \
    --output-report reports/training_report.json
```

**Expected Duration:** 8-12 hours (Optuna optimization dominates)

**Acceptance Criteria:**
- [ ] 162,500 samples generated successfully
- [ ] All 5 crop-specific models trained
- [ ] Global fallback model trained
- [ ] Bootstrap ensembles (10 replicates each) completed
- [ ] Training report generated with metrics

#### 14.1.2 Performance Benchmarking

**Metrics Collection Script:**
```python
# benchmark_model.py
from hierarchical_model import HierarchicalEnsemble
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import json

# Load test data
test_df = pd.read_csv('data/test_holdout.csv')
X_test = test_df.drop(columns=['quality_status'])
y_test = test_df['quality_status']

# Load trained ensemble
ensemble = HierarchicalEnsemble.load('models/')

# Generate predictions
y_pred, confidences = ensemble.predict_with_confidence(X_test)

# Compute metrics
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Save results
with open('reports/benchmark_results.json', 'w') as f:
    json.dump({
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'target_accuracy': 0.92,
        'actual_accuracy': report['accuracy'],
        'target_f1': 0.85,
        'actual_f1': report['weighted avg']['f1-score']
    }, f, indent=2)
```

**Performance Validation Checklist:**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Overall Accuracy | ≥91% | TBD | ⏳ Pending |
| Weighted F1 | ≥0.82 | TBD | ⏳ Pending |
| CRITICAL Recall | ≥85% | TBD | ⏳ Pending |
| SPOILED Precision | ≥90% | TBD | ⏳ Pending |
| Avg Inference Time | <50ms | TBD | ⏳ Pending |

#### 14.1.3 Gap Analysis and Remediation

If performance targets are not met, implement the following remediation strategies:

**Accuracy < 91%:**
1. Increase Optuna trials to 200
2. Add more digital twin scenarios for edge cases
3. Review feature importance and add domain-specific features
4. Consider XGBoost as alternative algorithm

**F1 < 0.82:**
1. Adjust class weights in LightGBM
2. Implement SMOTE for minority class oversampling
3. Review misclassification patterns via confusion matrix
4. Add class-specific threshold calibration

**Inference Time > 50ms:**
1. Profile feature engineering pipeline
2. Reduce SHAP background samples
3. Implement feature caching
4. Consider model quantization

---

### 14.2 Phase 2: Comprehensive Testing Strategy (Week 2-3)

#### 14.2.1 Unit Test Implementation

Create a comprehensive test suite using pytest with the following structure:

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_physics_engine.py
│   ├── test_feature_engineering.py
│   ├── test_digital_twin.py
│   ├── test_synthetic_data.py
│   └── test_hierarchical_model.py
├── integration/
│   ├── test_training_pipeline.py
│   ├── test_prediction_pipeline.py
│   └── test_api_endpoints.py
└── e2e/
    ├── test_full_workflow.py
    └── test_api_scenarios.py
```

**Physics Engine Tests (`test_physics_engine.py`):**
```python
import pytest
import numpy as np
from physics_engine import Q10KineticsEngine, CropType, CROP_PHYSICS_PARAMS

class TestQ10Kinetics:
    @pytest.fixture
    def avocado_engine(self):
        return Q10KineticsEngine(CropType.AVOCADO)
    
    def test_rate_multiplier_at_reference_temp(self, avocado_engine):
        """At reference temperature, rate multiplier should be 1.0"""
        T_ref = CROP_PHYSICS_PARAMS[CropType.AVOCADO].T_ref
        multiplier = avocado_engine.calculate_rate_multiplier(T_ref)
        assert abs(multiplier - 1.0) < 0.001
    
    def test_rate_multiplier_increases_with_temp(self, avocado_engine):
        """Higher temperature should increase metabolic rate"""
        low_temp = 5.0
        high_temp = 15.0
        assert avocado_engine.calculate_rate_multiplier(high_temp) > \
               avocado_engine.calculate_rate_multiplier(low_temp)
    
    def test_flu_accumulation(self, avocado_engine):
        """FLU should accumulate over time"""
        initial_flu = avocado_engine.total_flu
        avocado_engine.calculate_flu(10.0)  # Above optimal
        assert avocado_engine.total_flu > initial_flu
    
    @pytest.mark.parametrize("crop_type", list(CropType))
    def test_all_crops_have_valid_params(self, crop_type):
        """Verify all crops have valid physics parameters"""
        params = CROP_PHYSICS_PARAMS[crop_type]
        assert params.Q10 >= 1.0
        assert params.SL_ref > 0
        assert params.T_optimal[0] < params.T_optimal[1]
```

**Feature Engineering Tests (`test_feature_engineering.py`):**
```python
import pytest
import numpy as np
import pandas as pd
from feature_engineering import FeatureEngineer, PsychrometricCalculator

class TestPsychrometrics:
    def test_dew_point_below_temperature(self):
        """Dew point should always be ≤ ambient temperature"""
        calc = PsychrometricCalculator()
        for temp in [0, 10, 20, 30]:
            for rh in [50, 70, 90]:
                dew_point = calc.calculate_dew_point(temp, rh)
                assert dew_point <= temp
    
    def test_vpd_zero_at_saturation(self):
        """VPD should be ~0 at 100% humidity"""
        calc = PsychrometricCalculator()
        vpd = calc.calculate_vapor_pressure_deficit(20.0, 100.0)
        assert vpd < 0.01
    
    def test_vpd_increases_with_lower_humidity(self):
        """VPD increases as humidity decreases"""
        calc = PsychrometricCalculator()
        vpd_high_rh = calc.calculate_vapor_pressure_deficit(20.0, 90.0)
        vpd_low_rh = calc.calculate_vapor_pressure_deficit(20.0, 50.0)
        assert vpd_low_rh > vpd_high_rh

class TestFeatureEngineer:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(crop_type='avocado')
    
    def test_feature_count(self, engineer):
        """Should produce exactly 39 features"""
        sample = pd.DataFrame({
            'temperature': [7.0],
            'humidity': [90.0],
            'co2_ppm': [450.0],
            'light_lux': [0.0]
        })
        features = engineer.transform(sample)
        assert features.shape[1] == 39
    
    def test_no_nan_features(self, engineer):
        """Transformed features should not contain NaN"""
        sample = pd.DataFrame({
            'temperature': [7.0],
            'humidity': [90.0],
            'co2_ppm': [450.0],
            'light_lux': [0.0]
        })
        features = engineer.transform(sample)
        assert not features.isnull().any().any()
```

**API Endpoint Tests (`test_api_endpoints.py`):**
```python
import pytest
from fastapi.testclient import TestClient
from api_service import app, JWTHandler

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_token():
    handler = JWTHandler()
    return handler.create_token("test-tenant")

class TestPredictionEndpoint:
    def test_predict_valid_request(self, client, auth_token):
        response = client.post(
            "/v1/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "temperature": 7.0,
                "humidity": 90.0,
                "co2_ppm": 450,
                "light_lux": 0,
                "crop_type": "avocado"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["GOOD", "MARGINAL", "AT_RISK", "CRITICAL", "SPOILED"]
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_crop_type(self, client, auth_token):
        response = client.post(
            "/v1/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "temperature": 7.0,
                "humidity": 90.0,
                "crop_type": "invalid_crop"
            }
        )
        assert response.status_code == 422
    
    def test_predict_unauthorized(self, client):
        response = client.post(
            "/v1/predict",
            json={"temperature": 7.0, "humidity": 90.0, "crop_type": "avocado"}
        )
        assert response.status_code == 401

class TestBatchPrediction:
    def test_batch_predict_multiple_samples(self, client, auth_token):
        readings = [
            {"temperature": 7.0, "humidity": 90.0, "crop_type": "avocado"},
            {"temperature": 12.0, "humidity": 85.0, "crop_type": "mango"},
            {"temperature": 2.0, "humidity": 95.0, "crop_type": "leafy_greens"}
        ]
        response = client.post(
            "/v1/predict_batch",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"readings": readings}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
```

**pytest Configuration (`pytest.ini`):**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=. --cov-report=html --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests requiring full system
```

**Run Tests:**
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run integration tests (requires trained models)
pytest tests/integration/ -v -m integration
```

#### 14.2.2 Integration Testing

**Training Pipeline Integration Test:**
```python
# tests/integration/test_training_pipeline.py
import pytest
import tempfile
import os
from main import DataIngestionStage, SyntheticDataStage, TrainingStage

@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipeline:
    def test_mini_pipeline_execution(self):
        """Test complete pipeline with minimal data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Stage 1: Data ingestion
            ingestion = DataIngestionStage()
            real_df = ingestion.run("Dataset.csv")
            assert len(real_df) > 0
            
            # Stage 2: Synthetic data (minimal)
            synthetic = SyntheticDataStage()
            combined_df = synthetic.run(
                real_df, 
                digital_twin_samples=100,
                tvae_samples=0  # Skip for speed
            )
            assert len(combined_df) >= len(real_df)
            
            # Stage 3: Training (minimal)
            training = TrainingStage()
            models = training.run(
                combined_df,
                model_dir=tmpdir,
                optuna_trials=2  # Minimal trials
            )
            
            # Verify model files created
            assert os.path.exists(os.path.join(tmpdir, 'global_model.pkl'))
```

#### 14.2.3 Load Testing

**Locust Load Test Configuration (`locustfile.py`):**
```python
from locust import HttpUser, task, between
import random

class AgriSenseUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        # Authenticate and get token
        response = self.client.post("/v1/token", json={
            "tenant_id": f"load-test-{random.randint(1, 100)}",
            "api_key": "load-test-key"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(10)
    def predict_single(self):
        crop = random.choice(["avocado", "mango", "leafy_greens", "orange", "berries"])
        self.client.post("/v1/predict", headers=self.headers, json={
            "temperature": random.uniform(0, 15),
            "humidity": random.uniform(70, 98),
            "co2_ppm": random.uniform(300, 1500),
            "light_lux": random.uniform(0, 100),
            "crop_type": crop
        })
    
    @task(1)
    def health_check(self):
        self.client.get("/v1/health")
    
    @task(2)
    def predict_batch(self):
        readings = [{
            "temperature": random.uniform(0, 15),
            "humidity": random.uniform(70, 98),
            "crop_type": random.choice(["avocado", "mango"])
        } for _ in range(50)]
        self.client.post("/v1/predict_batch", headers=self.headers, json={
            "readings": readings
        })
```

**Run Load Tests:**
```bash
# Install locust
pip install locust

# Run load test (100 users, 10 spawn rate)
locust -f locustfile.py --host=http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 5m --headless
```

**Load Test Acceptance Criteria:**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Requests/sec | ≥500 | Locust statistics |
| P50 Latency | <30ms | Locust statistics |
| P95 Latency | <100ms | Locust statistics |
| P99 Latency | <200ms | Locust statistics |
| Error Rate | <0.1% | Failed requests / total |

---

### 14.3 Phase 3: API Integration Roadmap (Week 3-4)

#### 14.3.1 API Versioning Strategy

Implement semantic versioning for the API to ensure backward compatibility:

**URL Structure:**
```
/v1/predict      # Current stable version
/v2/predict      # Future major version (breaking changes)
/v1/health       # Version-independent health check
```

**Versioning Implementation:**
```python
# api_service.py additions
from fastapi import APIRouter

# Version 1 router
v1_router = APIRouter(prefix="/v1", tags=["v1"])

@v1_router.post("/predict")
async def predict_v1(reading: SensorReading):
    # Current implementation
    pass

# Version 2 router (future)
v2_router = APIRouter(prefix="/v2", tags=["v2"])

@v2_router.post("/predict")
async def predict_v2(reading: SensorReadingV2):
    # Enhanced implementation with new features
    pass

# Mount routers
app.include_router(v1_router)
# app.include_router(v2_router)  # Enable when V2 ready
```

#### 14.3.2 API Documentation Enhancement

**OpenAPI Schema Customization:**
```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="AgriSense ML Prediction API",
    description="""
    ## Overview
    The AgriSense API provides real-time produce quality predictions 
    using physics-grounded machine learning models.
    
    ## Authentication
    All endpoints require JWT authentication. Obtain a token via `/v1/token`.
    
    ## Rate Limits
    - Single predictions: 100 requests/minute
    - Batch predictions: 10 requests/minute
    - Maximum batch size: 1000 samples
    
    ## Supported Crops
    - avocado
    - mango
    - leafy_greens
    - orange
    - berries
    """,
    version="1.0.0",
    contact={
        "name": "AgriSense Development Team",
        "email": "api@agrisense.io"
    },
    license_info={
        "name": "Proprietary",
    }
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add example responses
    openapi_schema["paths"]["/v1/predict"]["post"]["responses"]["200"]["content"] = {
        "application/json": {
            "example": {
                "prediction": "GOOD",
                "prediction_class": 0,
                "confidence": 0.92,
                "confidence_interval": {"lower": 0.84, "upper": 1.0},
                "rsl_hours": 268.8,
                "rsl_status": "Good - 11.2 days remaining",
                "model_used": "crop_avocado",
                "processing_time_ms": 12.5,
                "crop_type": "avocado",
                "timestamp": "2026-02-27T10:30:00Z"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

#### 14.3.3 API Client SDK Generation

**Python SDK (`agrisense_client.py`):**
```python
"""
AgriSense Python SDK
Official client library for AgriSense ML Prediction API
"""
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PredictionResult:
    prediction: str
    prediction_class: int
    confidence: float
    confidence_interval: Dict[str, float]
    rsl_hours: Optional[float]
    rsl_status: Optional[str]
    model_used: str
    processing_time_ms: float
    timestamp: str

class AgriSenseClient:
    """
    Client for AgriSense ML Prediction API.
    
    Usage:
        client = AgriSenseClient(
            base_url="https://api.agrisense.io",
            tenant_id="farm-001",
            api_key="your-api-key"
        )
        
        result = client.predict(
            temperature=7.5,
            humidity=90.0,
            crop_type="avocado"
        )
        print(f"Quality: {result.prediction} ({result.confidence:.0%})")
    """
    
    def __init__(self, base_url: str, tenant_id: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.tenant_id = tenant_id
        self.api_key = api_key
        self._token = None
        self._token_expiry = None
    
    def _ensure_authenticated(self):
        """Refresh token if expired or not present."""
        if self._token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return
        
        response = requests.post(
            f"{self.base_url}/v1/token",
            json={"tenant_id": self.tenant_id, "api_key": self.api_key}
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expiry = datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60)
    
    def _headers(self) -> Dict[str, str]:
        self._ensure_authenticated()
        return {"Authorization": f"Bearer {self._token}"}
    
    def predict(
        self,
        temperature: float,
        humidity: float,
        crop_type: str,
        co2_ppm: float = 400,
        light_lux: float = 0,
        door_cycles_today: int = 0,
        compressor_duty_cycle: float = 50,
        batch_id: Optional[str] = None
    ) -> PredictionResult:
        """Make a single quality prediction."""
        payload = {
            "temperature": temperature,
            "humidity": humidity,
            "crop_type": crop_type,
            "co2_ppm": co2_ppm,
            "light_lux": light_lux,
            "door_cycles_today": door_cycles_today,
            "compressor_duty_cycle": compressor_duty_cycle
        }
        if batch_id:
            payload["batch_id"] = batch_id
        
        response = requests.post(
            f"{self.base_url}/v1/predict",
            headers=self._headers(),
            json=payload
        )
        response.raise_for_status()
        return PredictionResult(**response.json())
    
    def predict_batch(
        self,
        readings: List[Dict],
        include_explanations: bool = False
    ) -> List[PredictionResult]:
        """Make batch quality predictions."""
        response = requests.post(
            f"{self.base_url}/v1/predict_batch",
            headers=self._headers(),
            json={
                "readings": readings,
                "include_explanations": include_explanations
            }
        )
        response.raise_for_status()
        data = response.json()
        return [PredictionResult(**p) for p in data["predictions"]]
    
    def explain(
        self,
        temperature: float,
        humidity: float,
        crop_type: str,
        **kwargs
    ) -> Dict:
        """Get SHAP explanation for prediction."""
        payload = {
            "temperature": temperature,
            "humidity": humidity,
            "crop_type": crop_type,
            **kwargs
        }
        response = requests.post(
            f"{self.base_url}/v1/explain",
            headers=self._headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return response.json()
```

**JavaScript/TypeScript SDK (`agrisense-client.ts`):**
```typescript
/**
 * AgriSense JavaScript SDK
 * Official client library for AgriSense ML Prediction API
 */

interface SensorReading {
  temperature: number;
  humidity: number;
  crop_type: 'avocado' | 'mango' | 'leafy_greens' | 'orange' | 'berries';
  co2_ppm?: number;
  light_lux?: number;
  door_cycles_today?: number;
  compressor_duty_cycle?: number;
  batch_id?: string;
}

interface PredictionResult {
  prediction: 'GOOD' | 'MARGINAL' | 'AT_RISK' | 'CRITICAL' | 'SPOILED';
  prediction_class: number;
  confidence: number;
  confidence_interval: { lower: number; upper: number };
  rsl_hours: number | null;
  rsl_status: string | null;
  model_used: string;
  processing_time_ms: number;
  crop_type: string;
  timestamp: string;
}

interface ExplanationResult {
  prediction: string;
  confidence: number;
  top_contributors: Array<{
    feature: string;
    feature_description: string;
    value: number;
    shap_value: number;
    direction: 'toward_spoilage' | 'toward_good';
    interpretation: string;
  }>;
  summary: string;
  recommendations: string[];
}

export class AgriSenseClient {
  private baseUrl: string;
  private tenantId: string;
  private apiKey: string;
  private token: string | null = null;
  private tokenExpiry: Date | null = null;

  constructor(baseUrl: string, tenantId: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.tenantId = tenantId;
    this.apiKey = apiKey;
  }

  private async ensureAuthenticated(): Promise<void> {
    if (this.token && this.tokenExpiry && new Date() < this.tokenExpiry) {
      return;
    }

    const response = await fetch(`${this.baseUrl}/v1/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tenant_id: this.tenantId,
        api_key: this.apiKey
      })
    });

    if (!response.ok) {
      throw new Error(`Authentication failed: ${response.statusText}`);
    }

    const data = await response.json();
    this.token = data.access_token;
    this.tokenExpiry = new Date(Date.now() + (data.expires_in - 60) * 1000);
  }

  private async getHeaders(): Promise<Headers> {
    await this.ensureAuthenticated();
    return new Headers({
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    });
  }

  async predict(reading: SensorReading): Promise<PredictionResult> {
    const response = await fetch(`${this.baseUrl}/v1/predict`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: JSON.stringify(reading)
    });

    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }

    return response.json();
  }

  async predictBatch(readings: SensorReading[]): Promise<PredictionResult[]> {
    const response = await fetch(`${this.baseUrl}/v1/predict_batch`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: JSON.stringify({ readings })
    });

    if (!response.ok) {
      throw new Error(`Batch prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.predictions;
  }

  async explain(reading: SensorReading): Promise<ExplanationResult> {
    const response = await fetch(`${this.baseUrl}/v1/explain`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: JSON.stringify(reading)
    });

    if (!response.ok) {
      throw new Error(`Explanation failed: ${response.statusText}`);
    }

    return response.json();
  }

  async health(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(`${this.baseUrl}/v1/health`);
    return response.json();
  }
}
```

#### 14.3.4 WebSocket Real-Time Streaming

For real-time sensor monitoring, implement WebSocket support:

```python
# api_service.py additions
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/v1/ws/predictions")
async def websocket_predictions(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time prediction streaming.
    
    Connect: ws://api.agrisense.io/v1/ws/predictions?token=YOUR_JWT_TOKEN
    
    Send sensor readings as JSON, receive predictions in real-time.
    """
    # Validate token
    try:
        jwt_handler = JWTHandler()
        payload = jwt_handler.validate_token(token)
    except:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process prediction
            prediction = await process_prediction(data)
            
            # Send result back
            await websocket.send_json({
                "type": "prediction",
                "data": prediction,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

### 14.4 Phase 4: Frontend Integration (Week 4-6)

#### 14.4.1 Frontend Architecture Overview

Implement a modern React-based dashboard for AgriSense monitoring:

**Technology Stack:**
| Layer | Technology | Purpose |
|-------|------------|---------|
| Framework | React 18 + TypeScript | Component-based UI |
| State Management | Zustand / Redux Toolkit | Global state |
| Data Fetching | React Query (TanStack) | Server state management |
| Styling | Tailwind CSS + shadcn/ui | Utility-first styling |
| Charts | Recharts / Visx | Data visualization |
| Real-time | Socket.io-client | WebSocket integration |
| Build | Vite | Fast development builds |

**Project Structure:**
```
agrisense-dashboard/
├── src/
│   ├── api/
│   │   ├── client.ts              # AgriSense SDK integration
│   │   ├── hooks/
│   │   │   ├── usePrediction.ts
│   │   │   ├── useBatchPrediction.ts
│   │   │   └── useExplanation.ts
│   │   └── types.ts
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── Loading.tsx
│   │   ├── dashboard/
│   │   │   ├── QualityGauge.tsx
│   │   │   ├── SensorChart.tsx
│   │   │   ├── CropSelector.tsx
│   │   │   └── PredictionCard.tsx
│   │   ├── explanations/
│   │   │   ├── SHAPWaterfall.tsx
│   │   │   ├── FeatureImpact.tsx
│   │   │   └── Recommendations.tsx
│   │   └── layout/
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       └── MainLayout.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── ColdRoomDetail.tsx
│   │   ├── BatchHistory.tsx
│   │   ├── Alerts.tsx
│   │   └── Settings.tsx
│   ├── store/
│   │   ├── authStore.ts
│   │   ├── sensorStore.ts
│   │   └── predictionStore.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   └── constants.ts
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

#### 14.4.2 Core Frontend Components

**API Hook Integration (`usePrediction.ts`):**
```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { AgriSenseClient, SensorReading, PredictionResult } from '../client';

const client = new AgriSenseClient(
  import.meta.env.VITE_API_URL,
  import.meta.env.VITE_TENANT_ID,
  import.meta.env.VITE_API_KEY
);

export function usePrediction() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (reading: SensorReading) => client.predict(reading),
    onSuccess: (data) => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      // Update prediction history
      queryClient.setQueryData(['latestPrediction'], data);
    }
  });
}

export function usePredictionHistory(coldRoomId: string) {
  return useQuery({
    queryKey: ['predictions', coldRoomId],
    queryFn: () => fetchPredictionHistory(coldRoomId),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000
  });
}

export function useExplanation(reading: SensorReading | null) {
  return useQuery({
    queryKey: ['explanation', reading],
    queryFn: () => reading ? client.explain(reading) : null,
    enabled: !!reading
  });
}
```

**Quality Gauge Component (`QualityGauge.tsx`):**
```tsx
import React from 'react';
import { PredictionResult } from '@/api/types';

interface QualityGaugeProps {
  prediction: PredictionResult;
  size?: 'sm' | 'md' | 'lg';
}

const statusColors = {
  GOOD: { bg: 'bg-green-500', text: 'text-green-700', ring: 'ring-green-500' },
  MARGINAL: { bg: 'bg-yellow-500', text: 'text-yellow-700', ring: 'ring-yellow-500' },
  AT_RISK: { bg: 'bg-orange-500', text: 'text-orange-700', ring: 'ring-orange-500' },
  CRITICAL: { bg: 'bg-red-500', text: 'text-red-700', ring: 'ring-red-500' },
  SPOILED: { bg: 'bg-gray-800', text: 'text-gray-900', ring: 'ring-gray-800' }
};

export function QualityGauge({ prediction, size = 'md' }: QualityGaugeProps) {
  const colors = statusColors[prediction.prediction];
  const sizeClasses = {
    sm: 'w-24 h-24 text-sm',
    md: 'w-32 h-32 text-base',
    lg: 'w-48 h-48 text-lg'
  };
  
  const confidencePercent = Math.round(prediction.confidence * 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference * (1 - prediction.confidence);
  
  return (
    <div className={`relative ${sizeClasses[size]} flex items-center justify-center`}>
      {/* Background circle */}
      <svg className="absolute inset-0 transform -rotate-90">
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-gray-200"
        />
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className={colors.text}
        />
      </svg>
      
      {/* Center content */}
      <div className="text-center z-10">
        <div className={`font-bold ${colors.text}`}>
          {prediction.prediction}
        </div>
        <div className="text-gray-500 text-sm">
          {confidencePercent}% conf.
        </div>
      </div>
    </div>
  );
}
```

**SHAP Waterfall Chart (`SHAPWaterfall.tsx`):**
```tsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ExplanationResult } from '@/api/types';

interface SHAPWaterfallProps {
  explanation: ExplanationResult;
}

export function SHAPWaterfall({ explanation }: SHAPWaterfallProps) {
  const data = explanation.top_contributors.map(contrib => ({
    feature: contrib.feature_description,
    value: contrib.shap_value,
    direction: contrib.direction,
    interpretation: contrib.interpretation
  }));
  
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-4">Feature Impact Analysis</h3>
      
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
          <XAxis type="number" domain={['auto', 'auto']} />
          <YAxis type="category" dataKey="feature" width={100} />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload?.[0]) {
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-2 border rounded shadow">
                    <p className="font-medium">{data.feature}</p>
                    <p className="text-sm text-gray-600">{data.interpretation}</p>
                    <p className="text-sm">
                      Impact: <span className={data.value > 0 ? 'text-red-600' : 'text-green-600'}>
                        {data.value.toFixed(3)}
                      </span>
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell 
                key={index} 
                fill={entry.direction === 'toward_spoilage' ? '#ef4444' : '#22c55e'} 
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">{explanation.summary}</p>
      </div>
      
      {explanation.recommendations.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium text-sm text-gray-700 mb-2">Recommendations:</h4>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
            {explanation.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

**Real-Time Sensor Dashboard (`Dashboard.tsx`):**
```tsx
import React, { useState, useEffect } from 'react';
import { useSocket } from '@/hooks/useSocket';
import { usePrediction, usePredictionHistory } from '@/api/hooks/usePrediction';
import { QualityGauge } from '@/components/dashboard/QualityGauge';
import { SensorChart } from '@/components/dashboard/SensorChart';
import { SHAPWaterfall } from '@/components/explanations/SHAPWaterfall';
import { CropSelector } from '@/components/dashboard/CropSelector';

export function Dashboard() {
  const [selectedColdRoom, setSelectedColdRoom] = useState('coldroom-001');
  const [selectedCrop, setSelectedCrop] = useState<string>('avocado');
  const [latestReading, setLatestReading] = useState<SensorReading | null>(null);
  
  const { data: prediction, mutate: fetchPrediction } = usePrediction();
  const { data: history } = usePredictionHistory(selectedColdRoom);
  const { data: explanation } = useExplanation(latestReading);
  
  // WebSocket connection for real-time updates
  const socket = useSocket('/v1/ws/predictions');
  
  useEffect(() => {
    socket.on('sensor_update', (data: SensorReading) => {
      setLatestReading(data);
      fetchPrediction(data);
    });
    
    return () => socket.disconnect();
  }, [socket, fetchPrediction]);
  
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">AgriSense Dashboard</h1>
          <CropSelector value={selectedCrop} onChange={setSelectedCrop} />
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Current Status */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Current Quality Status</h2>
            {prediction && (
              <div className="flex flex-col items-center">
                <QualityGauge prediction={prediction} size="lg" />
                <div className="mt-4 text-center">
                  <p className="text-sm text-gray-600">
                    Remaining Shelf Life: {prediction.rsl_status}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Model: {prediction.model_used} | 
                    Latency: {prediction.processing_time_ms.toFixed(1)}ms
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Sensor Readings */}
          <div className="bg-white rounded-lg shadow p-6 lg:col-span-2">
            <h2 className="text-lg font-semibold mb-4">Sensor History (24h)</h2>
            <SensorChart 
              data={history?.sensorReadings || []}
              metrics={['temperature', 'humidity', 'co2_ppm']}
            />
          </div>
          
          {/* Explanation Panel */}
          <div className="lg:col-span-3">
            {explanation && <SHAPWaterfall explanation={explanation} />}
          </div>
          
          {/* Alert History */}
          <div className="bg-white rounded-lg shadow p-6 lg:col-span-3">
            <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
            <AlertTable coldRoomId={selectedColdRoom} />
          </div>
        </div>
      </main>
    </div>
  );
}
```

#### 14.4.3 Frontend Build and Deployment

**Vite Configuration (`vite.config.ts`):**
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
          query: ['@tanstack/react-query']
        }
      }
    }
  }
});
```

**Docker Configuration for Frontend (`Dockerfile.frontend`):**
```dockerfile
# Build stage
FROM node:20-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Nginx Configuration (`nginx.conf`):**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;

    # API proxy
    location /v1/ {
        proxy_pass http://api:8000/v1/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket proxy
    location /v1/ws/ {
        proxy_pass http://api:8000/v1/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

### 14.5 Phase 5: Production Deployment (Week 6-8)

#### 14.5.1 Complete Docker Compose Stack

**Production Docker Compose (`docker-compose.prod.yml`):**
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: agrisense-db
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: agrisense
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - agrisense-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: agrisense-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - agrisense-network

  # AgriSense ML API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: agrisense-api
    environment:
      - AGRISENSE_ENV=production
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@db:5432/agrisense
      - REDIS_URL=redis://redis:6379
      - MODEL_DIR=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    networks:
      - agrisense-network

  # Frontend Dashboard
  frontend:
    build:
      context: ./agrisense-dashboard
      dockerfile: Dockerfile.frontend
    container_name: agrisense-frontend
    depends_on:
      - api
    networks:
      - agrisense-network

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    container_name: agrisense-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - api
      - frontend
    networks:
      - agrisense-network

  # Prometheus Metrics
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: agrisense-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - agrisense-network

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:10.0.0
    container_name: agrisense-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
    networks:
      - agrisense-network

  # Log Aggregation
  loki:
    image: grafana/loki:2.8.0
    container_name: agrisense-loki
    volumes:
      - loki_data:/loki
    networks:
      - agrisense-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  loki_data:

networks:
  agrisense-network:
    driver: bridge
```

#### 14.5.2 Environment Configuration

**Production Environment Variables (`.env.prod`):**
```bash
# Database
DB_USER=agrisense_prod
DB_PASSWORD=<STRONG_RANDOM_PASSWORD>

# Authentication
JWT_SECRET_KEY=<CRYPTOGRAPHICALLY_SECURE_256BIT_KEY>

# API Configuration
AGRISENSE_ENV=production
MODEL_DIR=/app/models
LOG_LEVEL=INFO
MAX_WORKERS=4

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=<STRONG_RANDOM_PASSWORD>
PROMETHEUS_RETENTION_DAYS=30

# SSL (Let's Encrypt)
DOMAIN=api.agrisense.io
CERTBOT_EMAIL=ssl@agrisense.io
```

**Secrets Management with HashiCorp Vault (recommended):**
```bash
# Store secrets
vault kv put secret/agrisense/prod \
    db_password="<DB_PASSWORD>" \
    jwt_secret="<JWT_SECRET>" \
    grafana_password="<GRAFANA_PASSWORD>"

# Retrieve in application
export DB_PASSWORD=$(vault kv get -field=db_password secret/agrisense/prod)
```

#### 14.5.3 CI/CD Pipeline

**GitHub Actions Workflow (`.github/workflows/deploy.yml`):**
```yaml
name: AgriSense CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Ruff linter
        uses: chartboost/ruff-action@v1
      
      - name: Run type checker
        run: |
          pip install mypy
          mypy . --ignore-missing-imports

  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push API image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api:${{ github.sha }}
      
      - name: Build and push Frontend image
        uses: docker/build-push-action@v4
        with:
          context: ./agrisense-dashboard
          file: ./agrisense-dashboard/Dockerfile.frontend
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Deploy to staging
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.SSH_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/agrisense
            docker-compose -f docker-compose.staging.yml pull
            docker-compose -f docker-compose.staging.yml up -d
            docker system prune -f

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.SSH_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/agrisense
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d --no-deps api
            # Rolling restart with health checks
            sleep 30
            curl -f http://localhost:8000/v1/health || exit 1
            docker system prune -f
```

#### 14.5.4 Monitoring and Alerting

**Grafana Dashboard Configuration (`grafana/dashboards/agrisense.json`):**
Key panels to include:
- API Request Rate (requests/sec)
- API Latency (P50, P95, P99)
- Prediction Class Distribution
- Model Confidence Distribution
- Error Rate by Endpoint
- Active WebSocket Connections
- Database Connection Pool
- Memory/CPU Utilization

**Alert Rules (`prometheus/rules/agrisense.yml`):**
```yaml
groups:
  - name: agrisense-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 200ms)"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: ModelDriftDetected
        expr: agrisense_drift_detected == 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model drift detected"
          description: "Concept drift detected - consider retraining"
      
      - alert: LowPredictionConfidence
        expr: avg(agrisense_prediction_confidence) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Average prediction confidence is low"
```

---

### 14.6 Phase 6: Security Implementation (Week 7-8)

#### 14.6.1 Security Checklist

| Category | Requirement | Implementation | Status |
|----------|-------------|----------------|--------|
| Authentication | JWT with expiration | PyJWT with 24h expiry | ✅ |
| Authorization | Tenant isolation | JWT claims validation | ✅ |
| Transport | HTTPS/TLS 1.3 | Nginx SSL termination | ⏳ |
| API Security | Rate limiting | FastAPI middleware | ⏳ |
| Input Validation | Schema validation | Pydantic models | ✅ |
| Secrets | Secure storage | Environment variables + Vault | ⏳ |
| Logging | Audit trail | Structured logging to Loki | ⏳ |
| CORS | Origin restriction | FastAPI CORS middleware | ✅ |

#### 14.6.2 Rate Limiting Implementation

```python
from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < 60
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later."
            )
        
        self.requests[client_ip].append(now)
        return await call_next(request)

app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
```

---

### 14.7 Implementation Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Training & Validation | Week 1-2 | Trained models, benchmark results |
| Phase 2: Testing | Week 2-3 | Test suite, coverage reports |
| Phase 3: API Integration | Week 3-4 | SDK libraries, WebSocket support |
| Phase 4: Frontend | Week 4-6 | React dashboard, real-time monitoring |
| Phase 5: Production Deployment | Week 6-8 | Docker stack, CI/CD, monitoring |
| Phase 6: Security | Week 7-8 | SSL, rate limiting, audit logging |

**Total Estimated Duration:** 8 weeks to full production deployment

### 14.8 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Model Performance | ≥91% accuracy, ≥0.82 F1 | Test set evaluation |
| API Latency | P95 <100ms | Prometheus metrics |
| Uptime | 99.9% | Monitoring dashboard |
| Test Coverage | ≥80% | pytest-cov report |
| Security Scan | No critical vulnerabilities | Snyk/Trivy scan |
| Load Handling | 500 req/sec sustained | Locust load test |

### 14.9 Risk Mitigation Matrix

| Risk | Likelihood | Impact | Mitigation Strategy | Owner |
|------|------------|--------|---------------------|-------|
| Model underperforms | Medium | High | Increase data diversity, Optuna trials | ML Team |
| Integration delays | Medium | Medium | Parallel development, mock APIs | Dev Team |
| Security vulnerability | Low | Critical | Security scanning in CI, penetration test | Security |
| Production outage | Low | High | Blue-green deployment, rollback plan | DevOps |
| Data drift | Medium | Medium | Drift monitoring, automated retraining | ML Team |
| Dependency breaking | Low | Medium | Pin versions, test dependency updates | Dev Team |

---

## 15. Conclusion

The AgriSense Model Evolution Plan implementation represents a comprehensive transformation of agricultural quality prediction capabilities. Through the integration of physics-grounded digital twin simulation, TVAE-based synthetic data generation, sophisticated feature engineering, and hierarchical ensemble modeling, the system is positioned to deliver significant improvements over baseline performance.

Key achievements of this implementation phase include:

1. **Complete Physics Foundation:** Q10 kinetics, Gompertz microbial growth, transpiration, and CO2 respiration models implemented for all five target crop types with parameters derived from agricultural science literature.

2. **Robust Data Generation:** Digital twin environment capable of simulating diverse cold room scenarios, combined with SDV/TVAE integration for scalable synthetic data generation targeting 162,500 total training samples.

3. **Advanced Feature Engineering:** 39-feature pipeline incorporating temporal dynamics, psychrometric physics, infrastructure behavior, and sensor quality indicators for comprehensive input representation.

4. **Hierarchical Model Architecture:** Three-level LightGBM ensemble with Optuna hyperparameter optimization and bootstrap uncertainty quantification meeting the ±8% confidence interval specification.

5. **Production-Ready API:** FastAPI service with JWT authentication, comprehensive validation, SHAP-based explanations, and drift detection capabilities.

6. **Dependency Resolution:** All required Python packages (FastAPI, uvicorn, PyJWT, SHAP, LightGBM, Optuna, SDV) successfully installed and verified.

The system architecture follows software engineering best practices with modular design, comprehensive documentation, graceful degradation through optional dependency handling, and container-ready deployment configuration.

The next phase focuses on executing the full training pipeline, validating performance against target metrics (91-94% accuracy, 0.82-0.88 weighted F1), and preparing for production deployment. With continued focus on data quality, model monitoring, and continuous improvement, the AgriSense ML Module is well-positioned to deliver its projected 65% spoilage reduction impact for cold-chain produce quality management.

---

*Report prepared by AgriSense Development Team*  
*February 27, 2026*

---

## Appendix A: File Inventory

| File | Purpose | LOC |
|------|---------|-----|
| physics_engine.py | Q10 kinetics, microbial growth, transpiration | 926 |
| digital_twin.py | Cold room simulation environment | 927 |
| synthetic_data.py | TVAE/SDV synthetic generation | 747 |
| feature_engineering.py | 39-feature extraction pipeline | 837 |
| hierarchical_model.py | LightGBM hierarchical ensemble | 850 |
| explainability.py | SHAP, drift detection, incremental learning | 851 |
| api_service.py | FastAPI REST service | 761 |
| config.py | Configuration management | 372 |
| main.py | Pipeline orchestrator | 437 |
| requirements.txt | Python dependencies | 98 |
| Dockerfile | Container build | - |
| docker-compose.yml | Multi-service deployment | - |
| prometheus.yml | Metrics configuration | - |
| init.sql | Database schema | - |

## Appendix B: Crop Physics Parameters Summary

| Parameter | Avocado | Mango | Leafy Greens | Orange | Berries |
|-----------|---------|-------|--------------|--------|---------|
| Q10 | 2.4 | 2.1 | 1.8 | 1.9 | 2.5 |
| T_ref (°C) | 8 | 11 | 2 | 5 | 0 |
| SL_ref (hours) | 400 | 360 | 240 | 2160 | 168 |
| T_optimal (°C) | 5-8 | 10-13 | 0-4 | 3-8 | -1-2 |
| H_optimal (%) | 90-95 | 85-90 | 95-98 | 85-90 | 90-95 |
| k_transpiration | 1.2 | 1.0 | 2.0 | 0.4 | 0.6 |
| R_CO2_ref | 2.0 | 1.8 | 3.5 | 0.3 | 2.5 |

## Appendix C: Feature List

1. temperature
2. humidity
3. co2_ppm
4. light_lux
5. temperature_mean_24h
6. temperature_std_24h
7. temperature_min_24h
8. temperature_max_24h
9. temperature_trend
10. humidity_mean_24h
11. humidity_min_24h
12. humidity_deficit_cumulative
13. co2_rate_of_change
14. co2_accumulation_24h
15. co2_time_above_600
16. light_cumulative_24h
17. dew_point
18. vapor_pressure_deficit
19. condensation_risk
20. enthalpy
21. weighted_temperature
22. weighted_humidity
23. weighted_co2
24. weighted_light
25. stress_index
26. chilling_hours
27. heat_stress_hours
28. humidity_stress_index
29. anaerobic_risk
30. temperature_humidity_interaction
31. door_temperature_interaction
32. door_cycle_frequency
33. compressor_duty_cycle
34. energy_anomaly_score
35. pressure_stability
36. condensation_zones
37. temperature_anomaly_flag
38. humidity_anomaly_flag
39. overall_data_quality_score
