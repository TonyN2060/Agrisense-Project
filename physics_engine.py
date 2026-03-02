"""
AgriSense Physics-Grounded Digital Twin Engine

Implements comprehensive crop-specific physics models:
- Q10 metabolic kinetics for 5 target crops
- Humidity-respiration-transpiration cascade
- CO2 accumulation & anaerobic respiration modeling
- Microbial growth (Gompertz model)
- Quality index calculation
- Infrastructure sensor integration

Target Crops: Avocado, Mango, Leafy Greens, Orange, Berries
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class CropType(Enum):
    """Enumeration of supported crop types."""
    AVOCADO = "avocado"
    MANGO = "mango"
    LEAFY_GREENS = "leafy_greens"
    ORANGE = "orange"
    BERRIES = "berries"


class QualityStatus(Enum):
    """5-level quality classification system."""
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    AT_RISK = "AT_RISK"
    CRITICAL = "CRITICAL"
    SPOILED = "SPOILED"


@dataclass
class CropPhysicsParams:
    """Physics parameters for each crop type based on agricultural science literature."""
    crop_name: str
    T_ref: float                    # Reference temperature (°C)
    SL_ref: float                   # Reference shelf-life at T_ref (hours)
    Q10: float                      # Q10 temperature coefficient
    T_optimal: Tuple[float, float]  # Optimal temperature range (min, max)
    H_optimal: Tuple[float, float]  # Optimal humidity range (%, %)
    CO2_optimal: Tuple[float, float]  # Optimal CO2 range (ppm)
    chilling_threshold: float       # Temperature below which chilling injury occurs
    heat_threshold: float           # Temperature above which heat damage accelerates
    k_transpiration: float          # Transpiration coefficient (%/day at reference)
    R_CO2_ref: float               # CO2 respiration rate (mg CO2/kg/hr at reference)
    mold_growth_threshold: float    # Temperature threshold for mold activation
    ethylene_sensitivity: float     # Ethylene sensitivity (0-1 scale)
    water_loss_tolerance: float     # Acceptable water loss before quality impact (%)
    T_weight: float                 # Temperature importance weight for ML
    H_weight: float                 # Humidity importance weight for ML
    CO2_weight: float               # CO2 importance weight for ML
    light_weight: float             # Light importance weight for ML


# Comprehensive crop physics parameters based on post-harvest physiology literature
CROP_PHYSICS_PARAMS: Dict[CropType, CropPhysicsParams] = {
    CropType.AVOCADO: CropPhysicsParams(
        crop_name="Avocado",
        T_ref=8.0,
        SL_ref=400.0,  # ~16.7 days at optimal
        Q10=2.4,
        T_optimal=(5.0, 8.0),
        H_optimal=(90.0, 95.0),
        CO2_optimal=(300.0, 600.0),
        chilling_threshold=5.0,
        heat_threshold=13.0,
        k_transpiration=1.2,
        R_CO2_ref=2.0,
        mold_growth_threshold=15.0,
        ethylene_sensitivity=0.9,  # Highly sensitive
        water_loss_tolerance=8.0,
        T_weight=0.50,
        H_weight=0.25,
        CO2_weight=0.20,
        light_weight=0.05
    ),
    CropType.MANGO: CropPhysicsParams(
        crop_name="Mango",
        T_ref=11.0,
        SL_ref=360.0,  # ~15 days at optimal
        Q10=2.1,
        T_optimal=(10.0, 13.0),
        H_optimal=(85.0, 90.0),
        CO2_optimal=(300.0, 500.0),
        chilling_threshold=10.0,
        heat_threshold=15.0,
        k_transpiration=1.0,
        R_CO2_ref=1.8,
        mold_growth_threshold=15.0,
        ethylene_sensitivity=0.7,
        water_loss_tolerance=8.0,
        T_weight=0.55,
        H_weight=0.20,
        CO2_weight=0.15,
        light_weight=0.10
    ),
    CropType.LEAFY_GREENS: CropPhysicsParams(
        crop_name="Leafy Greens",
        T_ref=2.0,
        SL_ref=240.0,  # ~10 days at optimal
        Q10=1.8,
        T_optimal=(0.0, 4.0),
        H_optimal=(95.0, 98.0),
        CO2_optimal=(300.0, 800.0),
        chilling_threshold=-1.0,
        heat_threshold=8.0,
        k_transpiration=2.0,  # Highest transpiration rate
        R_CO2_ref=3.5,        # Highest respiration rate
        mold_growth_threshold=8.0,
        ethylene_sensitivity=0.5,
        water_loss_tolerance=5.0,  # Most sensitive to water loss
        T_weight=0.35,
        H_weight=0.40,  # Humidity most critical
        CO2_weight=0.15,
        light_weight=0.10
    ),
    CropType.ORANGE: CropPhysicsParams(
        crop_name="Orange",
        T_ref=5.0,
        SL_ref=2160.0,  # 90 days at optimal
        Q10=1.9,
        T_optimal=(3.0, 8.0),
        H_optimal=(85.0, 90.0),
        CO2_optimal=(300.0, 500.0),
        chilling_threshold=2.0,
        heat_threshold=15.0,
        k_transpiration=0.4,  # Low due to waxy skin
        R_CO2_ref=0.3,        # Lowest respiration
        mold_growth_threshold=10.0,
        ethylene_sensitivity=0.3,
        water_loss_tolerance=12.0,
        T_weight=0.40,
        H_weight=0.35,
        CO2_weight=0.15,
        light_weight=0.10
    ),
    CropType.BERRIES: CropPhysicsParams(
        crop_name="Berries",
        T_ref=0.0,
        SL_ref=168.0,  # 7 days at optimal
        Q10=2.5,       # Highest Q10 - most temperature sensitive
        T_optimal=(-1.0, 2.0),
        H_optimal=(90.0, 95.0),
        CO2_optimal=(300.0, 600.0),
        chilling_threshold=-2.0,
        heat_threshold=5.0,
        k_transpiration=0.6,
        R_CO2_ref=2.5,
        mold_growth_threshold=5.0,
        ethylene_sensitivity=0.4,
        water_loss_tolerance=8.0,
        T_weight=0.50,
        H_weight=0.20,
        CO2_weight=0.10,
        light_weight=0.00  # Light irrelevant for berries
    )
}


class Q10KineticsEngine:
    """
    Q10 temperature coefficient model for metabolic rate calculations.
    
    Q10 = factor by which reaction rate increases for every 10°C rise.
    Models shelf-life consumption based on storage temperature.
    """
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.params = CROP_PHYSICS_PARAMS[crop_type]
        self.total_flu = 0.0  # Fractional Life Used
        self.timestep_minutes = 5
        
    def calculate_rate_multiplier(self, T_current: float) -> float:
        """
        Calculate metabolic rate multiplier based on Q10 kinetics.
        
        Rate = Q10^((T_current - T_ref) / 10)
        """
        delta_T = T_current - self.params.T_ref
        return self.params.Q10 ** (delta_T / 10.0)
    
    def calculate_flu(self, T_current: float) -> float:
        """
        Calculate Fractional Life Used for current timestep.
        
        FLU = (timestep_hours / SL_ref) × Q10^((T - T_ref)/10)
        
        This measures how much shelf-life is consumed in this timestep,
        scaled by temperature-dependent metabolic acceleration.
        """
        rate_multiplier = self.calculate_rate_multiplier(T_current)
        timestep_hours = self.timestep_minutes / 60.0
        flu = (timestep_hours / self.params.SL_ref) * rate_multiplier
        return flu
    
    def calculate_chilling_injury_factor(self, T_current: float) -> float:
        """
        Calculate chilling injury acceleration factor.
        
        Returns a multiplier > 1 if temperature below chilling threshold.
        """
        if T_current < self.params.chilling_threshold:
            # Exponential damage below threshold
            delta = self.params.chilling_threshold - T_current
            return 1.0 + (0.5 * delta ** 1.5)  # Accelerating damage
        return 1.0
    
    def calculate_heat_stress_factor(self, T_current: float) -> float:
        """
        Calculate heat stress acceleration factor.
        
        Returns a multiplier > 1 if temperature above heat threshold.
        """
        if T_current > self.params.heat_threshold:
            delta = T_current - self.params.heat_threshold
            return 1.0 + (0.3 * delta ** 1.2)
        return 1.0
    
    def update_spoilage(self, T_current: float) -> Dict:
        """
        Update cumulative spoilage and return current status.
        
        Incorporates chilling injury and heat stress factors.
        """
        base_flu = self.calculate_flu(T_current)
        chilling_factor = self.calculate_chilling_injury_factor(T_current)
        heat_factor = self.calculate_heat_stress_factor(T_current)
        
        # Combined FLU with injury factors
        adjusted_flu = base_flu * chilling_factor * heat_factor
        self.total_flu += adjusted_flu
        
        # Calculate remaining shelf life
        rsl_percent = max(0.0, (1.0 - self.total_flu) * 100.0)
        rsl_hours = max(0.0, (1.0 - self.total_flu) * self.params.SL_ref)
        
        # Determine quality status
        status = self._determine_status(rsl_percent)
        
        return {
            'flu_current': adjusted_flu,
            'flu_total': self.total_flu,
            'rsl_percent': rsl_percent,
            'rsl_hours': rsl_hours,
            'status': status,
            'temperature': T_current,
            'in_optimal_range': self.params.T_optimal[0] <= T_current <= self.params.T_optimal[1],
            'chilling_injury_active': chilling_factor > 1.0,
            'heat_stress_active': heat_factor > 1.0,
            'rate_multiplier': self.calculate_rate_multiplier(T_current)
        }
    
    def _determine_status(self, rsl_percent: float) -> QualityStatus:
        """Map RSL percentage to 5-level quality status."""
        if rsl_percent <= 0:
            return QualityStatus.SPOILED
        elif rsl_percent <= 20:
            return QualityStatus.CRITICAL
        elif rsl_percent <= 40:
            return QualityStatus.AT_RISK
        elif rsl_percent <= 60:
            return QualityStatus.MARGINAL
        else:
            return QualityStatus.GOOD
    
    def reset(self):
        """Reset for new batch/shipment."""
        self.total_flu = 0.0


class TranspirationEngine:
    """
    Humidity-respiration-transpiration cascade model.
    
    Models water loss dynamics coupled to respiration rate:
    dW/dt = -k_w × (RH/100)^0.5 × Q10^((T-T_ref)/10)
    """
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.params = CROP_PHYSICS_PARAMS[crop_type]
        self.total_water_loss = 0.0  # Cumulative water loss (%)
        self.timestep_minutes = 5
        
    def calculate_transpiration_rate(self, T_current: float, RH_current: float) -> float:
        """
        Calculate instantaneous transpiration rate (%/hour).
        
        Transpiration is:
        - Decreased by high humidity (vapor pressure deficit lower)
        - Increased by higher temperature (metabolic rate higher)
        """
        # Vapor pressure deficit factor (higher RH = lower transpiration)
        vpd_factor = max(0.01, (100.0 - RH_current) / 100.0) ** 0.5
        
        # Q10 metabolic coupling
        delta_T = T_current - self.params.T_ref
        rate_multiplier = self.params.Q10 ** (delta_T / 10.0)
        
        # Base transpiration rate (convert from %/day to %/hour)
        base_rate = self.params.k_transpiration / 24.0
        
        return base_rate * vpd_factor * rate_multiplier
    
    def update_water_loss(self, T_current: float, RH_current: float) -> Dict:
        """
        Update cumulative water loss and return status.
        """
        rate = self.calculate_transpiration_rate(T_current, RH_current)
        timestep_hours = self.timestep_minutes / 60.0
        water_loss_step = rate * timestep_hours
        
        self.total_water_loss += water_loss_step
        
        # Quality impact calculation
        tolerance = self.params.water_loss_tolerance
        quality_impact = min(1.0, self.total_water_loss / tolerance) if tolerance > 0 else 0.0
        
        return {
            'transpiration_rate': rate,
            'water_loss_step': water_loss_step,
            'total_water_loss': self.total_water_loss,
            'quality_impact': quality_impact,
            'desiccation_warning': self.total_water_loss > (tolerance * 0.7),
            'desiccation_critical': self.total_water_loss > tolerance,
            'humidity_optimal': self.params.H_optimal[0] <= RH_current <= self.params.H_optimal[1]
        }
    
    def reset(self):
        """Reset for new batch."""
        self.total_water_loss = 0.0


class CO2RespirationEngine:
    """
    CO2 accumulation and anaerobic respiration trigger model.
    
    Models CO2 buildup from produce respiration in sealed/semi-sealed
    cold rooms, triggering anaerobic conditions at high concentrations.
    """
    
    AMBIENT_CO2 = 400.0  # ppm baseline
    ANAEROBIC_THRESHOLD = 5000.0  # ppm - triggers anaerobic respiration
    MOLD_THRESHOLD = 3000.0  # ppm - accelerates mold germination
    
    def __init__(self, crop_type: CropType, room_volume_m3: float = 100.0, 
                 produce_mass_kg: float = 5000.0):
        self.crop_type = crop_type
        self.params = CROP_PHYSICS_PARAMS[crop_type]
        self.room_volume = room_volume_m3
        self.produce_mass = produce_mass_kg
        self.current_co2 = self.AMBIENT_CO2
        self.timestep_minutes = 5
        
    def calculate_respiration_rate(self, T_current: float) -> float:
        """
        Calculate CO2 production rate (mg CO2/kg/hr) at current temperature.
        """
        delta_T = T_current - self.params.T_ref
        rate_multiplier = self.params.Q10 ** (delta_T / 10.0)
        return self.params.R_CO2_ref * rate_multiplier
    
    def update_co2(self, T_current: float, air_exchange_rate: float = 0.1,
                   door_open_minutes: float = 0.0) -> Dict:
        """
        Update CO2 concentration considering production and ventilation.
        
        Args:
            T_current: Current temperature (°C)
            air_exchange_rate: Air changes per hour (0-1)
            door_open_minutes: Minutes door was open in this timestep
        """
        timestep_hours = self.timestep_minutes / 60.0
        
        # CO2 production (mg/hr -> ppm conversion)
        respiration_rate = self.calculate_respiration_rate(T_current)
        co2_produced_mg = respiration_rate * self.produce_mass * timestep_hours
        
        # Convert mg CO2 to ppm (assuming standard conditions)
        # 1 ppm = 1.96 mg/m³ for CO2 at STP
        co2_produced_ppm = co2_produced_mg / (self.room_volume * 1.96)
        
        # Air exchange removes CO2
        exchange_factor = 1.0 - (air_exchange_rate * timestep_hours)
        
        # Door opening causes additional air exchange
        if door_open_minutes > 0:
            door_exchange = door_open_minutes / self.timestep_minutes * 0.3
            exchange_factor *= (1.0 - door_exchange)
        
        # Update CO2 level
        self.current_co2 = max(
            self.AMBIENT_CO2,
            (self.current_co2 - self.AMBIENT_CO2) * exchange_factor + self.AMBIENT_CO2 + co2_produced_ppm
        )
        
        # Assess risks
        anaerobic_risk = self.current_co2 > self.ANAEROBIC_THRESHOLD
        mold_risk_elevated = self.current_co2 > self.MOLD_THRESHOLD
        
        return {
            'current_co2': self.current_co2,
            'respiration_rate': respiration_rate,
            'co2_produced_ppm': co2_produced_ppm,
            'anaerobic_risk': anaerobic_risk,
            'mold_risk_elevated': mold_risk_elevated,
            'co2_optimal': self.params.CO2_optimal[0] <= self.current_co2 <= self.params.CO2_optimal[1],
            'ventilation_needed': self.current_co2 > self.params.CO2_optimal[1]
        }
    
    def reset(self):
        """Reset for new simulation."""
        self.current_co2 = self.AMBIENT_CO2


class MicrobialGrowthEngine:
    """
    Gompertz model for microbial growth dynamics.
    
    Models pathogen population growth as function of temperature,
    humidity, and time, enabling mold/bacterial outbreak prediction.
    
    Gompertz equation: N(t) = A × exp(-exp((μm × e / A) × (λ - t) + 1))
    Where:
        A = maximum population (asymptote)
        μm = maximum growth rate
        λ = lag phase duration
    """
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.params = CROP_PHYSICS_PARAMS[crop_type]
        self.microbial_load = 0.0  # Log CFU/g equivalent
        self.lag_phase_complete = False
        self.time_elapsed_hours = 0.0
        self.timestep_minutes = 5
        
        # Gompertz parameters (crop-specific tuning)
        self._set_gompertz_params()
        
    def _set_gompertz_params(self):
        """Set Gompertz model parameters based on crop type."""
        # Base parameters - modified by environmental conditions
        if self.crop_type == CropType.BERRIES:
            self.A = 8.0       # Max population (log CFU/g)
            self.mu_max = 0.8  # Max growth rate at optimal conditions
            self.lambda_base = 12.0  # Lag phase at optimal (hours)
        elif self.crop_type == CropType.LEAFY_GREENS:
            self.A = 7.5
            self.mu_max = 0.6
            self.lambda_base = 18.0
        elif self.crop_type == CropType.MANGO:
            self.A = 7.0
            self.mu_max = 0.5
            self.lambda_base = 24.0
        elif self.crop_type == CropType.AVOCADO:
            self.A = 7.0
            self.mu_max = 0.4
            self.lambda_base = 30.0
        else:  # Orange
            self.A = 6.0
            self.mu_max = 0.3
            self.lambda_base = 48.0
            
    def calculate_growth_rate(self, T_current: float, RH_current: float) -> float:
        """
        Calculate actual growth rate based on environmental conditions.
        
        Growth is:
        - Accelerated by temperature above mold threshold
        - Accelerated by high humidity (>90%)
        - Inhibited by low temperature
        """
        T_threshold = self.params.mold_growth_threshold
        
        # Temperature effect
        if T_current < T_threshold - 5:
            temp_factor = 0.1  # Minimal growth at low temps
        elif T_current < T_threshold:
            temp_factor = 0.3 + 0.14 * (T_current - (T_threshold - 5))
        else:
            temp_factor = 1.0 + 0.15 * (T_current - T_threshold)
        
        # Humidity effect  
        if RH_current < 75:
            humidity_factor = 0.2
        elif RH_current < 85:
            humidity_factor = 0.5
        elif RH_current < 95:
            humidity_factor = 0.8
        else:
            humidity_factor = 1.0 + 0.1 * (RH_current - 95)
            
        return min(2.0, self.mu_max * temp_factor * humidity_factor)
    
    def calculate_lag_phase(self, T_current: float) -> float:
        """Calculate adjusted lag phase based on temperature."""
        T_threshold = self.params.mold_growth_threshold
        
        if T_current < T_threshold - 5:
            return self.lambda_base * 3.0  # Extended lag at low temps
        elif T_current < T_threshold:
            return self.lambda_base * (1.5 - 0.1 * (T_current - (T_threshold - 5)))
        else:
            return max(2.0, self.lambda_base * (1.0 - 0.05 * (T_current - T_threshold)))
    
    def update_microbial_growth(self, T_current: float, RH_current: float) -> Dict:
        """
        Update microbial population using modified Gompertz model.
        """
        timestep_hours = self.timestep_minutes / 60.0
        self.time_elapsed_hours += timestep_hours
        
        growth_rate = self.calculate_growth_rate(T_current, RH_current)
        lag_phase = self.calculate_lag_phase(T_current)
        
        # Check if still in lag phase
        if self.time_elapsed_hours < lag_phase:
            self.lag_phase_complete = False
            growth_increment = 0.01  # Minimal growth during lag
        else:
            self.lag_phase_complete = True
            # Exponential-like growth phase
            growth_increment = growth_rate * timestep_hours * (1.0 - self.microbial_load / self.A)
        
        self.microbial_load = min(self.A, self.microbial_load + max(0, growth_increment))
        
        # Risk assessment
        spoilage_threshold = 0.6 * self.A  # 60% of max = visible spoilage
        
        return {
            'microbial_load': self.microbial_load,
            'growth_rate': growth_rate,
            'lag_phase_complete': self.lag_phase_complete,
            'time_to_spoilage_hours': self._estimate_time_to_spoilage(growth_rate),
            'mold_risk_score': self.microbial_load / self.A,
            'visible_spoilage': self.microbial_load >= spoilage_threshold,
            'mold_outbreak': self.microbial_load >= 0.8 * self.A
        }
    
    def _estimate_time_to_spoilage(self, current_growth_rate: float) -> float:
        """Estimate hours until visible spoilage at current rate."""
        spoilage_threshold = 0.6 * self.A
        remaining = spoilage_threshold - self.microbial_load
        
        if remaining <= 0:
            return 0.0
        if current_growth_rate <= 0.01:
            return float('inf')
            
        return remaining / current_growth_rate
    
    def reset(self):
        """Reset for new batch."""
        self.microbial_load = 0.0
        self.lag_phase_complete = False
        self.time_elapsed_hours = 0.0


class QualityIndexEngine:
    """
    Comprehensive quality index calculation combining all physics models.
    
    Quality_Index = Base × Decay_T × Decay_H × Decay_CO2 × Decay_Microbial × Infrastructure_Factor
    
    Returns a 0-100 scale quality index with multi-dimensional decay factors.
    """
    
    def __init__(self, crop_type: CropType, room_volume_m3: float = 100.0,
                 produce_mass_kg: float = 5000.0):
        self.crop_type = crop_type
        self.params = CROP_PHYSICS_PARAMS[crop_type]
        
        # Initialize component engines
        self.q10_engine = Q10KineticsEngine(crop_type)
        self.transpiration_engine = TranspirationEngine(crop_type)
        self.co2_engine = CO2RespirationEngine(crop_type, room_volume_m3, produce_mass_kg)
        self.microbial_engine = MicrobialGrowthEngine(crop_type)
        
        # Quality tracking
        self.base_quality = 100.0
        self.current_quality_index = 100.0
        self.quality_history: List[float] = []
        
    def calculate_temperature_decay_factor(self, T_current: float) -> float:
        """Calculate quality decay factor from temperature deviation."""
        T_opt_min, T_opt_max = self.params.T_optimal
        
        if T_opt_min <= T_current <= T_opt_max:
            return 1.0  # Optimal - no decay
        elif T_current < T_opt_min:
            # Chilling damage
            deviation = T_opt_min - T_current
            return max(0.5, 1.0 - 0.05 * deviation ** 1.3)
        else:
            # Heat damage
            deviation = T_current - T_opt_max
            return max(0.4, 1.0 - 0.08 * deviation ** 1.2)
    
    def calculate_humidity_decay_factor(self, RH_current: float) -> float:
        """Calculate quality decay factor from humidity deviation."""
        H_opt_min, H_opt_max = self.params.H_optimal
        
        if H_opt_min <= RH_current <= H_opt_max:
            return 1.0
        elif RH_current < H_opt_min:
            # Desiccation stress
            deviation = H_opt_min - RH_current
            return max(0.6, 1.0 - 0.03 * deviation)
        else:
            # Mold risk from excess humidity
            deviation = RH_current - H_opt_max
            return max(0.7, 1.0 - 0.02 * deviation)
    
    def calculate_co2_decay_factor(self, CO2_current: float) -> float:
        """Calculate quality decay factor from CO2 levels."""
        CO2_opt_min, CO2_opt_max = self.params.CO2_optimal
        
        if CO2_opt_min <= CO2_current <= CO2_opt_max:
            return 1.0
        elif CO2_current > CO2_opt_max:
            excess = CO2_current - CO2_opt_max
            # Exponential decay above optimal
            return max(0.5, 1.0 - 0.0001 * excess ** 1.1)
        return 1.0  # Low CO2 is generally not harmful
    
    def calculate_microbial_decay_factor(self, mold_risk_score: float) -> float:
        """Calculate quality decay from microbial activity."""
        if mold_risk_score < 0.2:
            return 1.0
        elif mold_risk_score < 0.5:
            return 1.0 - 0.2 * (mold_risk_score - 0.2) / 0.3
        else:
            return max(0.3, 0.8 - 0.5 * (mold_risk_score - 0.5) / 0.5)
    
    def calculate_infrastructure_factor(self, door_cycles: int = 0,
                                        energy_anomaly_score: float = 0.0,
                                        compressor_runtime_pct: float = 50.0) -> float:
        """
        Calculate infrastructure impact on quality.
        
        Args:
            door_cycles: Number of door open events in period
            energy_anomaly_score: Deviation from expected energy (0-1)
            compressor_runtime_pct: Compressor duty cycle (0-100)
        """
        # Door cycle stress (baseline 5 cycles/day acceptable)
        door_factor = max(0.85, 1.0 - 0.01 * max(0, door_cycles - 5))
        
        # Equipment health factor
        equipment_factor = 1.0 - energy_anomaly_score * 0.2
        
        # Cooling efficiency factor
        if compressor_runtime_pct > 80:
            cooling_factor = 0.9  # High load = potential issues
        elif compressor_runtime_pct < 20:
            cooling_factor = 0.95  # Very low may indicate problems
        else:
            cooling_factor = 1.0
            
        return door_factor * equipment_factor * cooling_factor
    
    def update_quality(self, T_current: float, RH_current: float,
                       air_exchange_rate: float = 0.1,
                       door_open_minutes: float = 0.0,
                       door_cycles: int = 0,
                       energy_anomaly_score: float = 0.0,
                       compressor_runtime_pct: float = 50.0) -> Dict:
        """
        Comprehensive quality update combining all physics models.
        """
        # Update component models
        q10_result = self.q10_engine.update_spoilage(T_current)
        transp_result = self.transpiration_engine.update_water_loss(T_current, RH_current)
        co2_result = self.co2_engine.update_co2(T_current, air_exchange_rate, door_open_minutes)
        microbial_result = self.microbial_engine.update_microbial_growth(T_current, RH_current)
        
        # Calculate decay factors
        temp_decay = self.calculate_temperature_decay_factor(T_current)
        humidity_decay = self.calculate_humidity_decay_factor(RH_current)
        co2_decay = self.calculate_co2_decay_factor(co2_result['current_co2'])
        microbial_decay = self.calculate_microbial_decay_factor(microbial_result['mold_risk_score'])
        infra_factor = self.calculate_infrastructure_factor(
            door_cycles, energy_anomaly_score, compressor_runtime_pct
        )
        
        # Combine with RSL-based quality
        rsl_quality = q10_result['rsl_percent']
        
        # Final quality index (weighted combination)
        self.current_quality_index = (
            rsl_quality * 
            temp_decay * 
            humidity_decay * 
            co2_decay * 
            microbial_decay * 
            infra_factor *
            (1.0 - transp_result['quality_impact'] * 0.3)  # Water loss impact
        )
        
        self.quality_history.append(self.current_quality_index)
        
        # Determine final status
        status = self._determine_status()
        
        return {
            'quality_index': self.current_quality_index,
            'status': status,
            'rsl_hours': q10_result['rsl_hours'],
            'rsl_percent': q10_result['rsl_percent'],
            'decay_factors': {
                'temperature': temp_decay,
                'humidity': humidity_decay,
                'co2': co2_decay,
                'microbial': microbial_decay,
                'infrastructure': infra_factor,
                'water_loss': 1.0 - transp_result['quality_impact'] * 0.3
            },
            'component_results': {
                'q10': q10_result,
                'transpiration': transp_result,
                'co2': co2_result,
                'microbial': microbial_result
            },
            'warnings': self._generate_warnings(
                q10_result, transp_result, co2_result, microbial_result,
                T_current, RH_current
            ),
            'recommendations': self._generate_recommendations(
                T_current, RH_current, co2_result['current_co2'],
                microbial_result['mold_risk_score']
            )
        }
    
    def _determine_status(self) -> QualityStatus:
        """Map quality index to status."""
        qi = self.current_quality_index
        if qi <= 0 or qi <= 10:
            return QualityStatus.SPOILED
        elif qi <= 30:
            return QualityStatus.CRITICAL
        elif qi <= 50:
            return QualityStatus.AT_RISK
        elif qi <= 70:
            return QualityStatus.MARGINAL
        else:
            return QualityStatus.GOOD
    
    def _generate_warnings(self, q10_result: Dict, transp_result: Dict,
                          co2_result: Dict, microbial_result: Dict,
                          T_current: float, RH_current: float) -> List[str]:
        """Generate warning messages based on current state."""
        warnings = []
        
        if q10_result['chilling_injury_active']:
            warnings.append(f"CHILLING INJURY: Temperature {T_current:.1f}°C below threshold")
        if q10_result['heat_stress_active']:
            warnings.append(f"HEAT STRESS: Temperature {T_current:.1f}°C above threshold")
        if transp_result['desiccation_warning']:
            warnings.append(f"DESICCATION WARNING: {transp_result['total_water_loss']:.1f}% water loss")
        if transp_result['desiccation_critical']:
            warnings.append(f"CRITICAL WATER LOSS: {transp_result['total_water_loss']:.1f}% - quality impact")
        if co2_result['anaerobic_risk']:
            warnings.append(f"ANAEROBIC RISK: CO2 at {co2_result['current_co2']:.0f} ppm")
        if co2_result['ventilation_needed']:
            warnings.append("VENTILATION NEEDED: CO2 above optimal range")
        if microbial_result['visible_spoilage']:
            warnings.append("MOLD VISIBLE: Microbial load at spoilage threshold")
        if microbial_result['mold_outbreak']:
            warnings.append("MOLD OUTBREAK: Critical microbial contamination")
            
        return warnings
    
    def _generate_recommendations(self, T_current: float, RH_current: float,
                                  CO2_current: float, mold_risk: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        T_opt_min, T_opt_max = self.params.T_optimal
        H_opt_min, H_opt_max = self.params.H_optimal
        
        if T_current < T_opt_min:
            recommendations.append(f"Increase temperature to {T_opt_min:.1f}°C")
        elif T_current > T_opt_max:
            recommendations.append(f"Reduce temperature to {T_opt_max:.1f}°C")
            
        if RH_current < H_opt_min:
            recommendations.append(f"Increase humidity to {H_opt_min:.0f}%")
        elif RH_current > H_opt_max:
            recommendations.append(f"Reduce humidity to {H_opt_max:.0f}%")
            
        if CO2_current > self.params.CO2_optimal[1]:
            recommendations.append("Increase ventilation to reduce CO2")
            
        if mold_risk > 0.5:
            recommendations.append("Consider expedited dispatch - mold risk elevated")
            
        if not recommendations:
            recommendations.append("Conditions optimal - maintain current settings")
            
        return recommendations
    
    def get_quality_trend(self, lookback_hours: int = 24) -> Dict:
        """Analyze quality trend over specified period."""
        timesteps = int(lookback_hours * 60 / 5)  # 5-minute timesteps
        
        if len(self.quality_history) < 2:
            return {'trend': 'stable', 'rate': 0.0}
            
        recent = self.quality_history[-timesteps:] if len(self.quality_history) >= timesteps else self.quality_history
        
        if len(recent) < 2:
            return {'trend': 'stable', 'rate': 0.0}
            
        # Linear regression for trend
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]
        
        # Slope in quality points per hour
        rate_per_hour = slope * (60 / 5)
        
        if rate_per_hour < -2.0:
            trend = 'rapidly_declining'
        elif rate_per_hour < -0.5:
            trend = 'declining'
        elif rate_per_hour > 0.5:
            trend = 'improving'  # Rare but possible with better conditions
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'rate_per_hour': rate_per_hour,
            'projected_hours_to_critical': (recent[-1] - 30) / abs(rate_per_hour) if rate_per_hour < -0.1 else float('inf')
        }
    
    def reset(self):
        """Reset all engines for new batch."""
        self.q10_engine.reset()
        self.transpiration_engine.reset()
        self.co2_engine.reset()
        self.microbial_engine.reset()
        self.current_quality_index = 100.0
        self.quality_history = []


def get_crop_params(crop_name: str) -> CropPhysicsParams:
    """Get physics parameters for a crop by name."""
    crop_map = {
        'avocado': CropType.AVOCADO,
        'mango': CropType.MANGO,
        'leafy_greens': CropType.LEAFY_GREENS,
        'leafy greens': CropType.LEAFY_GREENS,
        'kale': CropType.LEAFY_GREENS,
        'spinach': CropType.LEAFY_GREENS,
        'amaranth': CropType.LEAFY_GREENS,
        'orange': CropType.ORANGE,
        'oranges': CropType.ORANGE,
        'citrus': CropType.ORANGE,
        'berries': CropType.BERRIES,
        'strawberry': CropType.BERRIES,
        'blueberry': CropType.BERRIES,
        'berry': CropType.BERRIES
    }
    
    crop_key = crop_name.lower().strip()
    if crop_key not in crop_map:
        raise ValueError(f"Unknown crop: {crop_name}. Supported: {list(crop_map.keys())}")
    
    return CROP_PHYSICS_PARAMS[crop_map[crop_key]]


def create_quality_engine(crop_name: str, room_volume_m3: float = 100.0,
                         produce_mass_kg: float = 5000.0) -> QualityIndexEngine:
    """Factory function to create quality engine for a crop."""
    crop_map = {
        'avocado': CropType.AVOCADO,
        'mango': CropType.MANGO,
        'leafy_greens': CropType.LEAFY_GREENS,
        'leafy greens': CropType.LEAFY_GREENS,
        'kale': CropType.LEAFY_GREENS,
        'spinach': CropType.LEAFY_GREENS,
        'orange': CropType.ORANGE,
        'berries': CropType.BERRIES,
        'strawberry': CropType.BERRIES,
        'blueberry': CropType.BERRIES
    }
    
    crop_key = crop_name.lower().strip()
    if crop_key not in crop_map:
        raise ValueError(f"Unknown crop: {crop_name}")
    
    return QualityIndexEngine(crop_map[crop_key], room_volume_m3, produce_mass_kg)


if __name__ == "__main__":
    # Demo usage
    print("AgriSense Physics Engine Demo")
    print("=" * 50)
    
    for crop_type in CropType:
        engine = QualityIndexEngine(crop_type)
        print(f"\n{crop_type.value.upper()}:")
        print(f"  Optimal Temp: {engine.params.T_optimal}")
        print(f"  Optimal RH: {engine.params.H_optimal}")
        print(f"  Q10: {engine.params.Q10}")
        print(f"  Reference Shelf-Life: {engine.params.SL_ref} hours")
        
        # Simulate 24 hours at optimal conditions
        for _ in range(288):  # 5-min timesteps for 24 hours
            T = (engine.params.T_optimal[0] + engine.params.T_optimal[1]) / 2
            RH = (engine.params.H_optimal[0] + engine.params.H_optimal[1]) / 2
            result = engine.update_quality(T, RH)
        
        print(f"  After 24h optimal: Quality={result['quality_index']:.1f}, Status={result['status'].value}")
