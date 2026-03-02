"""
AgriSense Digital Twin Environment

Extended cold room simulation with:
- Crop-specific physics integration
- Infrastructure sensors (door cycles, energy, pressure, compressor runtime)
- Environmental sensor simulation (temperature, humidity, CO2, light)
- Scenario-based data generation for synthetic dataset creation
- Support for multi-day simulations with realistic environmental variations

Generates training data for hierarchical ML models.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Generator
from enum import Enum
import random
from datetime import datetime, timedelta

from physics_engine import (
    CropType, QualityStatus, QualityIndexEngine, 
    CROP_PHYSICS_PARAMS, CropPhysicsParams
)


@dataclass
class ColdRoomConfig:
    """Configuration for cold room simulation."""
    room_volume_m3: float = 100.0          # Room volume
    insulation_R_value: float = 3.5         # Thermal insulation R-value
    compressor_power_kw: float = 5.0        # Compressor cooling capacity
    ambient_temp_mean: float = 28.0         # Mean ambient temperature (Kenya context)
    ambient_temp_daily_range: float = 8.0   # Daily temperature swing
    produce_mass_kg: float = 5000.0         # Total produce mass
    
    # Infrastructure characteristics
    door_area_m2: float = 4.0               # Door opening area
    infiltration_rate: float = 0.15         # Air changes per door open event
    compressor_efficiency: float = 0.85     # Compressor COP efficiency factor
    humidity_control_capacity: float = 0.8  # Humidifier/dehumidifier capacity
    
    # Simulation settings
    timestep_minutes: int = 5               # Simulation timestep
    

@dataclass
class InfrastructureState:
    """Current state of cold room infrastructure."""
    # Door state
    door_open: bool = False
    door_open_duration_minutes: float = 0.0
    door_cycles_today: int = 0
    door_cycles_cumulative: int = 0
    
    # Compressor state
    compressor_on: bool = True
    compressor_runtime_current_hour: float = 0.0  # minutes
    compressor_duty_cycle: float = 50.0  # percentage
    compressor_power_draw: float = 0.0   # kW
    
    # Energy tracking
    energy_consumed_kwh: float = 0.0
    energy_anomaly_score: float = 0.0
    
    # Pressure monitoring
    room_pressure_pa: float = -2.0  # Slight negative pressure (normal)
    pressure_stability: float = 1.0  # 1.0 = stable, lower = unstable
    
    # Humidity gradient (multi-point)
    humidity_gradient_variance: float = 0.0
    condensation_risk_zones: int = 0


@dataclass
class EnvironmentalState:
    """Current environmental conditions in cold room."""
    temperature: float = 5.0
    humidity: float = 90.0
    co2_ppm: float = 400.0
    light_lux: float = 0.0
    ethylene_ppm: float = 0.5
    
    # Multi-point readings (for gradient detection)
    temperature_gradient: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    humidity_points: List[float] = field(default_factory=lambda: [90.0, 90.0, 90.0])


@dataclass
class SimulationScenario:
    """Defines a simulation scenario for data generation."""
    scenario_id: str
    crop_type: CropType
    duration_hours: int = 336  # 14 days default
    
    # Environmental profiles
    temperature_profile: str = "optimal"  # optimal, warm, cold, fluctuating, excursion
    humidity_profile: str = "optimal"     # optimal, dry, humid, variable
    co2_profile: str = "normal"           # normal, poor_ventilation, sealed
    light_profile: str = "dark"           # dark, periodic, constant
    
    # Infrastructure profiles
    door_frequency_profile: str = "normal"  # low, normal, high, erratic
    equipment_health_profile: str = "good"  # good, degrading, failing
    
    # Initial conditions
    initial_quality: str = "fresh"  # fresh, pre_damaged, aged_2days
    initial_temperature: Optional[float] = None
    initial_humidity: Optional[float] = None
    
    # Disturbance events
    cold_chain_breaks: List[Tuple[int, int]] = field(default_factory=list)  # (hour_start, duration_hours)
    door_events: List[Tuple[int, int]] = field(default_factory=list)  # (hour, duration_minutes)


class DigitalTwinEnvironment:
    """
    Physics-grounded digital twin for cold room simulation.
    
    Generates realistic sensor data combining:
    - Thermodynamic heat transfer models
    - Crop-specific spoilage physics
    - Equipment behavior modeling
    - Realistic noise and sensor artifacts
    """
    
    def __init__(self, crop_type: CropType, config: Optional[ColdRoomConfig] = None):
        self.crop_type = crop_type
        self.config = config or ColdRoomConfig()
        self.crop_params = CROP_PHYSICS_PARAMS[crop_type]
        
        # Initialize quality engine
        self.quality_engine = QualityIndexEngine(
            crop_type, 
            self.config.room_volume_m3,
            self.config.produce_mass_kg
        )
        
        # State tracking
        self.env_state = EnvironmentalState()
        self.infra_state = InfrastructureState()
        
        # Initialize to optimal conditions for crop
        self._initialize_optimal()
        
        # Time tracking
        self.time_elapsed_minutes = 0
        self.current_hour = 0
        
        # Data collection
        self.sensor_history: List[Dict] = []
        
    def _initialize_optimal(self):
        """Initialize environment to crop-optimal conditions."""
        T_opt = (self.crop_params.T_optimal[0] + self.crop_params.T_optimal[1]) / 2
        H_opt = (self.crop_params.H_optimal[0] + self.crop_params.H_optimal[1]) / 2
        
        self.env_state.temperature = T_opt
        self.env_state.humidity = H_opt
        self.env_state.co2_ppm = 450.0
        self.env_state.light_lux = 0.0
        
        self.setpoint_temperature = T_opt
        self.setpoint_humidity = H_opt
        
    def reset(self, scenario: Optional[SimulationScenario] = None):
        """Reset environment for new simulation."""
        self.time_elapsed_minutes = 0
        self.current_hour = 0
        self.sensor_history = []
        
        self.quality_engine.reset()
        self.infra_state = InfrastructureState()
        
        if scenario:
            self._apply_scenario_initial_conditions(scenario)
        else:
            self._initialize_optimal()
    
    def _apply_scenario_initial_conditions(self, scenario: SimulationScenario):
        """Apply scenario-specific initial conditions."""
        if scenario.initial_temperature is not None:
            self.env_state.temperature = scenario.initial_temperature
            self.setpoint_temperature = scenario.initial_temperature
        else:
            self._initialize_optimal()
            
        if scenario.initial_humidity is not None:
            self.env_state.humidity = scenario.initial_humidity
            self.setpoint_humidity = scenario.initial_humidity
            
        # Pre-damage simulation
        if scenario.initial_quality == "pre_damaged":
            self.quality_engine.q10_engine.total_flu = 0.15
            self.quality_engine.microbial_engine.microbial_load = 1.5
        elif scenario.initial_quality == "aged_2days":
            # Simulate 48 hours of optimal storage
            for _ in range(576):  # 48 hours at 5-min intervals
                self.quality_engine.update_quality(
                    self.env_state.temperature,
                    self.env_state.humidity
                )
    
    def _get_ambient_temperature(self, hour: int) -> float:
        """Calculate ambient temperature with diurnal variation."""
        # Sinusoidal daily pattern (peak at 14:00, min at 06:00)
        hour_of_day = hour % 24
        phase = (hour_of_day - 6) / 24 * 2 * np.pi
        variation = np.sin(phase) * (self.config.ambient_temp_daily_range / 2)
        
        # Add small random variation
        noise = np.random.normal(0, 0.5)
        
        return self.config.ambient_temp_mean + variation + noise
    
    def _update_temperature_physics(self, ambient_temp: float):
        """Update temperature using heat transfer physics."""
        current_temp = self.env_state.temperature
        
        # Heat gain through walls (conduction)
        delta_T_walls = ambient_temp - current_temp
        heat_gain_walls = delta_T_walls / self.config.insulation_R_value
        
        # Heat gain from door opening
        if self.infra_state.door_open:
            # Rapid infiltration while door open
            infiltration_rate = self.config.infiltration_rate * (
                self.config.door_area_m2 / self.config.room_volume_m3
            )
            heat_gain_door = (ambient_temp - current_temp) * infiltration_rate * 3.0
        else:
            heat_gain_door = 0.0
        
        # Heat from produce respiration (small but measurable)
        respiration_heat = self.quality_engine.co2_engine.calculate_respiration_rate(
            current_temp
        ) * self.config.produce_mass_kg * 0.0001  # Convert to temperature change
        
        # Cooling from compressor
        if self.infra_state.compressor_on:
            cooling_power = self.config.compressor_power_kw * self.config.compressor_efficiency
            # Cooling proportional to temperature difference from setpoint
            if current_temp > self.setpoint_temperature:
                cooling_effect = -cooling_power * 0.5  # degrees per timestep
            else:
                cooling_effect = -cooling_power * 0.1  # Maintenance cooling
        else:
            cooling_effect = 0.0
        
        # Combine and update
        timestep_hours = self.config.timestep_minutes / 60.0
        temp_change = (
            heat_gain_walls + 
            heat_gain_door + 
            respiration_heat + 
            cooling_effect
        ) * timestep_hours
        
        # Add sensor noise
        noise = np.random.normal(0, 0.05)
        
        self.env_state.temperature = current_temp + temp_change + noise
        
        # Update temperature gradient (multi-point)
        base_gradient = [
            self.env_state.temperature - 0.3,  # Floor level
            self.env_state.temperature,         # Mid level
            self.env_state.temperature + 0.5   # Ceiling level
        ]
        self.env_state.temperature_gradient = [
            t + np.random.normal(0, 0.1) for t in base_gradient
        ]
    
    def _update_humidity_physics(self, ambient_humidity: float = 65.0):
        """Update humidity considering transpiration and infiltration."""
        current_rh = self.env_state.humidity
        
        # Moisture from produce transpiration (increases humidity)
        transpiration_rate = self.quality_engine.transpiration_engine.calculate_transpiration_rate(
            self.env_state.temperature,
            current_rh
        )
        moisture_gain = transpiration_rate * 0.1  # Scale to RH change
        
        # Moisture loss from infiltration (door open events)
        if self.infra_state.door_open:
            infiltration_loss = (current_rh - ambient_humidity) * 0.05
        else:
            infiltration_loss = 0.0
        
        # Humidity control system
        setpoint_diff = self.setpoint_humidity - current_rh
        if abs(setpoint_diff) > 2.0:
            humidity_correction = np.sign(setpoint_diff) * min(
                abs(setpoint_diff) * 0.1,
                self.config.humidity_control_capacity
            )
        else:
            humidity_correction = 0.0
        
        # Update with noise
        rh_change = moisture_gain - infiltration_loss + humidity_correction
        noise = np.random.normal(0, 0.3)
        
        self.env_state.humidity = np.clip(
            current_rh + rh_change + noise,
            20.0, 99.9
        )
        
        # Update multi-point humidity
        self.env_state.humidity_points = [
            self.env_state.humidity + np.random.normal(0, 1.0),
            self.env_state.humidity + np.random.normal(0, 0.5),
            self.env_state.humidity + np.random.normal(0, 1.5)
        ]
        
        # Calculate humidity gradient variance
        self.infra_state.humidity_gradient_variance = np.var(self.env_state.humidity_points)
        
        # Condensation risk zones (where humidity exceeds dew point locally)
        self.infra_state.condensation_risk_zones = sum(
            1 for h in self.env_state.humidity_points if h > 97
        )
    
    def _update_co2_physics(self):
        """Update CO2 levels from produce respiration."""
        co2_result = self.quality_engine.co2_engine.update_co2(
            self.env_state.temperature,
            air_exchange_rate=0.05 if not self.infra_state.door_open else 0.3,
            door_open_minutes=self.infra_state.door_open_duration_minutes if self.infra_state.door_open else 0.0
        )
        
        self.env_state.co2_ppm = co2_result['current_co2'] + np.random.normal(0, 10)
    
    def _update_light(self, hour: int):
        """Update light levels based on time and door activity."""
        hour_of_day = hour % 24
        
        # Base light level (cold rooms typically dark)
        base_light = 5.0
        
        # Light from door opening (during daytime)
        if self.infra_state.door_open and 6 <= hour_of_day <= 18:
            door_light = np.random.uniform(500, 2000)
        elif self.infra_state.door_open:
            door_light = np.random.uniform(50, 200)  # Night lighting
        else:
            door_light = 0.0
        
        self.env_state.light_lux = base_light + door_light + np.random.uniform(0, 10)
    
    def _update_compressor_state(self):
        """Update compressor operation based on temperature control needs."""
        temp_error = self.env_state.temperature - self.setpoint_temperature
        
        # Hysteresis control (typical refrigeration control)
        if temp_error > 0.5:
            self.infra_state.compressor_on = True
        elif temp_error < -0.3:
            self.infra_state.compressor_on = False
        
        # Update runtime tracking
        if self.infra_state.compressor_on:
            self.infra_state.compressor_runtime_current_hour += self.config.timestep_minutes
            self.infra_state.compressor_power_draw = (
                self.config.compressor_power_kw * 
                np.random.uniform(0.9, 1.1)  # Small variation
            )
        else:
            self.infra_state.compressor_power_draw = 0.05  # Standby power
        
        # Calculate duty cycle every hour
        if self.time_elapsed_minutes % 60 == 0:
            self.infra_state.compressor_duty_cycle = (
                self.infra_state.compressor_runtime_current_hour / 60.0 * 100.0
            )
            self.infra_state.compressor_runtime_current_hour = 0.0
        
        # Energy consumption
        timestep_hours = self.config.timestep_minutes / 60.0
        self.infra_state.energy_consumed_kwh += (
            self.infra_state.compressor_power_draw * timestep_hours
        )
        
        # Calculate energy anomaly (deviation from expected)
        expected_energy_rate = self.config.compressor_power_kw * 0.5  # Expect 50% duty
        actual_rate = self.infra_state.compressor_power_draw
        self.infra_state.energy_anomaly_score = abs(
            (actual_rate - expected_energy_rate) / expected_energy_rate
        )
    
    def _update_door_state(self, scenario_door_event: bool = False,
                           event_duration: int = 0):
        """Update door state and simulate door events."""
        # Scheduled door event from scenario
        if scenario_door_event:
            self.infra_state.door_open = True
            self.infra_state.door_open_duration_minutes = event_duration
            self.infra_state.door_cycles_today += 1
            self.infra_state.door_cycles_cumulative += 1
            return
        
        # Random door events based on time of day
        hour_of_day = self.current_hour % 24
        
        # Higher door frequency during working hours
        if 6 <= hour_of_day <= 18:
            door_probability = 0.03  # 3% per timestep during day
        else:
            door_probability = 0.005  # 0.5% at night
        
        if self.infra_state.door_open:
            # Door was open, check if closing
            self.infra_state.door_open_duration_minutes += self.config.timestep_minutes
            if self.infra_state.door_open_duration_minutes >= np.random.uniform(2, 10):
                self.infra_state.door_open = False
                self.infra_state.door_open_duration_minutes = 0.0
        else:
            # Door was closed, check if opening
            if np.random.random() < door_probability:
                self.infra_state.door_open = True
                self.infra_state.door_cycles_today += 1
                self.infra_state.door_cycles_cumulative += 1
        
        # Reset daily counter at midnight
        if hour_of_day == 0 and self.time_elapsed_minutes % (24 * 60) < self.config.timestep_minutes:
            self.infra_state.door_cycles_today = 0
    
    def _update_pressure(self):
        """Update room pressure based on compressor operation."""
        # Slight negative pressure when compressor running (drawing air through coils)
        if self.infra_state.compressor_on:
            target_pressure = -3.0
        else:
            target_pressure = -1.0
        
        # Door open causes pressure equalization
        if self.infra_state.door_open:
            target_pressure = 0.0
        
        # Gradual pressure change
        current = self.infra_state.room_pressure_pa
        self.infra_state.room_pressure_pa = current + (target_pressure - current) * 0.1
        
        # Add noise
        self.infra_state.room_pressure_pa += np.random.normal(0, 0.3)
        
        # Calculate stability (variance over recent readings)
        self.infra_state.pressure_stability = max(
            0.5,
            1.0 - abs(self.infra_state.room_pressure_pa) * 0.05
        )
    
    def step(self, scenario: Optional[SimulationScenario] = None) -> Dict:
        """
        Execute one simulation timestep and return sensor readings.
        """
        # Update time
        self.time_elapsed_minutes += self.config.timestep_minutes
        self.current_hour = self.time_elapsed_minutes // 60
        
        # Check for scenario events
        scenario_door_event = False
        door_duration = 0
        cold_chain_break = False
        
        if scenario:
            # Check for door events
            for event_hour, duration in scenario.door_events:
                if event_hour == self.current_hour:
                    scenario_door_event = True
                    door_duration = duration
                    break
            
            # Check for cold chain breaks
            for break_start, break_duration in scenario.cold_chain_breaks:
                if break_start <= self.current_hour < break_start + break_duration:
                    cold_chain_break = True
                    break
        
        # Modify setpoints during cold chain break
        original_setpoint = self.setpoint_temperature
        if cold_chain_break:
            self.setpoint_temperature = self._get_ambient_temperature(self.current_hour) - 5
        
        # Update infrastructure
        self._update_door_state(scenario_door_event, door_duration)
        self._update_compressor_state()
        self._update_pressure()
        
        # Update environment
        ambient_temp = self._get_ambient_temperature(self.current_hour)
        self._update_temperature_physics(ambient_temp)
        self._update_humidity_physics()
        self._update_co2_physics()
        self._update_light(self.current_hour)
        
        # Restore setpoint after cold chain break handled
        if cold_chain_break:
            self.setpoint_temperature = original_setpoint
        
        # Update quality model
        quality_result = self.quality_engine.update_quality(
            T_current=self.env_state.temperature,
            RH_current=self.env_state.humidity,
            air_exchange_rate=0.3 if self.infra_state.door_open else 0.05,
            door_open_minutes=self.infra_state.door_open_duration_minutes,
            door_cycles=self.infra_state.door_cycles_today,
            energy_anomaly_score=self.infra_state.energy_anomaly_score,
            compressor_runtime_pct=self.infra_state.compressor_duty_cycle
        )
        
        # Compile sensor reading
        sensor_reading = self._compile_sensor_reading(quality_result)
        self.sensor_history.append(sensor_reading)
        
        return sensor_reading
    
    def _compile_sensor_reading(self, quality_result: Dict) -> Dict:
        """Compile all sensor data into a single reading."""
        return {
            # Timestamp
            'timestamp_minutes': self.time_elapsed_minutes,
            'hour': self.current_hour,
            
            # Crop info
            'crop_type': self.crop_type.value,
            
            # Environmental sensors
            'temperature': round(self.env_state.temperature, 2),
            'humidity': round(self.env_state.humidity, 2),
            'co2_ppm': round(self.env_state.co2_ppm, 1),
            'light_lux': round(self.env_state.light_lux, 1),
            'ethylene_ppm': round(self.env_state.ethylene_ppm + np.random.uniform(0, 0.2), 2),
            
            # Temperature gradient
            'temperature_floor': round(self.env_state.temperature_gradient[0], 2),
            'temperature_mid': round(self.env_state.temperature_gradient[1], 2),
            'temperature_ceiling': round(self.env_state.temperature_gradient[2], 2),
            
            # Humidity gradient
            'humidity_point_1': round(self.env_state.humidity_points[0], 2),
            'humidity_point_2': round(self.env_state.humidity_points[1], 2),
            'humidity_point_3': round(self.env_state.humidity_points[2], 2),
            'humidity_gradient_variance': round(self.infra_state.humidity_gradient_variance, 3),
            
            # Infrastructure sensors
            'door_open': int(self.infra_state.door_open),
            'door_cycles_today': self.infra_state.door_cycles_today,
            'door_cycles_cumulative': self.infra_state.door_cycles_cumulative,
            'compressor_on': int(self.infra_state.compressor_on),
            'compressor_duty_cycle': round(self.infra_state.compressor_duty_cycle, 2),
            'compressor_power_kw': round(self.infra_state.compressor_power_draw, 3),
            'energy_consumed_kwh': round(self.infra_state.energy_consumed_kwh, 3),
            'energy_anomaly_score': round(self.infra_state.energy_anomaly_score, 3),
            'room_pressure_pa': round(self.infra_state.room_pressure_pa, 2),
            'pressure_stability': round(self.infra_state.pressure_stability, 3),
            'condensation_risk_zones': self.infra_state.condensation_risk_zones,
            
            # Quality metrics
            'quality_index': round(quality_result['quality_index'], 2),
            'quality_status': quality_result['status'].value,
            'rsl_hours': round(quality_result['rsl_hours'], 1),
            'rsl_percent': round(quality_result['rsl_percent'], 2),
            
            # Decay factors
            'decay_temperature': round(quality_result['decay_factors']['temperature'], 4),
            'decay_humidity': round(quality_result['decay_factors']['humidity'], 4),
            'decay_co2': round(quality_result['decay_factors']['co2'], 4),
            'decay_microbial': round(quality_result['decay_factors']['microbial'], 4),
            'decay_infrastructure': round(quality_result['decay_factors']['infrastructure'], 4),
            
            # Component quality indicators
            'flu_total': round(quality_result['component_results']['q10']['flu_total'], 4),
            'water_loss_percent': round(quality_result['component_results']['transpiration']['total_water_loss'], 2),
            'microbial_load': round(quality_result['component_results']['microbial']['microbial_load'], 3),
            'mold_risk_score': round(quality_result['component_results']['microbial']['mold_risk_score'], 3),
            
            # Binary classification labels
            'is_good': int(quality_result['status'] == QualityStatus.GOOD),
            'is_spoiled': int(quality_result['status'] == QualityStatus.SPOILED),
            
            # Multi-class label (0-4)
            'quality_class': {
                QualityStatus.GOOD: 0,
                QualityStatus.MARGINAL: 1,
                QualityStatus.AT_RISK: 2,
                QualityStatus.CRITICAL: 3,
                QualityStatus.SPOILED: 4
            }[quality_result['status']]
        }
    
    def run_simulation(self, duration_hours: int,
                       scenario: Optional[SimulationScenario] = None) -> pd.DataFrame:
        """
        Run complete simulation and return sensor data DataFrame.
        """
        self.reset(scenario)
        
        timesteps = int(duration_hours * 60 / self.config.timestep_minutes)
        
        for _ in range(timesteps):
            self.step(scenario)
        
        return pd.DataFrame(self.sensor_history)
    
    def get_current_state(self) -> Dict:
        """Get current environment state summary."""
        return {
            'time_elapsed_hours': self.time_elapsed_minutes / 60,
            'temperature': self.env_state.temperature,
            'humidity': self.env_state.humidity,
            'co2': self.env_state.co2_ppm,
            'quality_index': self.quality_engine.current_quality_index,
            'status': self.quality_engine._determine_status().value,
            'compressor_duty_cycle': self.infra_state.compressor_duty_cycle,
            'energy_consumed': self.infra_state.energy_consumed_kwh,
            'door_cycles': self.infra_state.door_cycles_cumulative
        }


class ScenarioGenerator:
    """
    Generates diverse simulation scenarios for synthetic data creation.
    
    Creates 480 environmental combinations × 5 crops × 3 initial conditions
    × 2 infrastructure profiles = 14,400 base scenarios.
    """
    
    # Temperature profiles relative to optimal
    TEMP_PROFILES = {
        'optimal': (0, 0),           # At optimal
        'cold': (-3, -1),            # Below optimal
        'warm': (2, 5),              # Above optimal
        'fluctuating': (-2, 4),      # Wide swings
        'cold_excursion': (0, 0),    # Normal with cold events
        'warm_excursion': (0, 0),    # Normal with warm events
        'extreme_cold': (-8, -3),    # Severe chilling
        'extreme_warm': (8, 15)      # Severe heat
    }
    
    HUMIDITY_PROFILES = {
        'optimal': (0, 0),
        'dry': (-15, -5),
        'humid': (3, 8),
        'variable': (-10, 10),
        'very_dry': (-25, -15)
    }
    
    CO2_PROFILES = {
        'normal': 'normal',
        'poor_ventilation': 'high',
        'sealed': 'very_high',
        'well_ventilated': 'low'
    }
    
    LIGHT_PROFILES = {
        'dark': 'minimal',
        'periodic': 'daytime',
        'constant': 'continuous',
        'variable': 'erratic'
    }
    
    DOOR_PROFILES = {
        'low': (1, 3),           # 1-3 door cycles per day
        'normal': (4, 8),
        'high': (10, 20),
        'erratic': (5, 30)
    }
    
    EQUIPMENT_PROFILES = {
        'good': 0.0,            # No anomaly
        'degrading': 0.3,       # Moderate anomaly
        'failing': 0.6          # High anomaly
    }
    
    INITIAL_CONDITIONS = ['fresh', 'pre_damaged', 'aged_2days']
    
    @classmethod
    def generate_scenario_matrix(cls, 
                                 crops: List[CropType] = None,
                                 scenarios_per_crop: int = 100) -> List[SimulationScenario]:
        """
        Generate comprehensive scenario matrix for training data.
        """
        if crops is None:
            crops = list(CropType)
        
        scenarios = []
        scenario_id = 0
        
        for crop in crops:
            crop_params = CROP_PHYSICS_PARAMS[crop]
            T_opt = (crop_params.T_optimal[0] + crop_params.T_optimal[1]) / 2
            H_opt = (crop_params.H_optimal[0] + crop_params.H_optimal[1]) / 2
            
            for temp_profile_name, temp_offset in cls.TEMP_PROFILES.items():
                for humidity_profile_name, humidity_offset in cls.HUMIDITY_PROFILES.items():
                    for initial_condition in cls.INITIAL_CONDITIONS:
                        for door_profile in ['normal', 'high']:
                            for equipment_profile in ['good', 'degrading']:
                                
                                # Calculate actual starting conditions
                                if isinstance(temp_offset, tuple):
                                    init_temp = T_opt + np.random.uniform(*temp_offset)
                                else:
                                    init_temp = T_opt
                                
                                if isinstance(humidity_offset, tuple):
                                    init_humidity = H_opt + np.random.uniform(*humidity_offset)
                                else:
                                    init_humidity = H_opt
                                
                                # Add cold chain break events for excursion profiles
                                cold_chain_breaks = []
                                if 'excursion' in temp_profile_name:
                                    # Random break between hours 24-72
                                    break_start = np.random.randint(24, 72)
                                    break_duration = np.random.randint(2, 8)
                                    cold_chain_breaks.append((break_start, break_duration))
                                
                                scenario = SimulationScenario(
                                    scenario_id=f"{crop.value}_{scenario_id:05d}",
                                    crop_type=crop,
                                    duration_hours=336,  # 14 days
                                    temperature_profile=temp_profile_name,
                                    humidity_profile=humidity_profile_name,
                                    door_frequency_profile=door_profile,
                                    equipment_health_profile=equipment_profile,
                                    initial_quality=initial_condition,
                                    initial_temperature=init_temp,
                                    initial_humidity=init_humidity,
                                    cold_chain_breaks=cold_chain_breaks
                                )
                                
                                scenarios.append(scenario)
                                scenario_id += 1
                                
                                # Limit scenarios per crop for manageable generation
                                if scenario_id % (len(crops)) >= scenarios_per_crop:
                                    break
        
        return scenarios
    
    @classmethod
    def generate_single_scenario(cls, crop_type: CropType,
                                 profile: str = 'random') -> SimulationScenario:
        """Generate a single scenario with specified or random profile."""
        crop_params = CROP_PHYSICS_PARAMS[crop_type]
        T_opt = (crop_params.T_optimal[0] + crop_params.T_optimal[1]) / 2
        H_opt = (crop_params.H_optimal[0] + crop_params.H_optimal[1]) / 2
        
        if profile == 'random':
            temp_profile = random.choice(list(cls.TEMP_PROFILES.keys()))
            humidity_profile = random.choice(list(cls.HUMIDITY_PROFILES.keys()))
            door_profile = random.choice(list(cls.DOOR_PROFILES.keys()))
            initial_condition = random.choice(cls.INITIAL_CONDITIONS)
        else:
            temp_profile = 'optimal'
            humidity_profile = 'optimal'
            door_profile = 'normal'
            initial_condition = 'fresh'
        
        temp_offset = cls.TEMP_PROFILES[temp_profile]
        humidity_offset = cls.HUMIDITY_PROFILES[humidity_profile]
        
        init_temp = T_opt + np.random.uniform(*temp_offset) if isinstance(temp_offset, tuple) else T_opt
        init_humidity = H_opt + np.random.uniform(*humidity_offset) if isinstance(humidity_offset, tuple) else H_opt
        
        return SimulationScenario(
            scenario_id=f"{crop_type.value}_{random.randint(0, 99999):05d}",
            crop_type=crop_type,
            duration_hours=336,
            temperature_profile=temp_profile,
            humidity_profile=humidity_profile,
            door_frequency_profile=door_profile,
            initial_quality=initial_condition,
            initial_temperature=init_temp,
            initial_humidity=init_humidity
        )


class SyntheticDataGenerator:
    """
    Generates large-scale synthetic datasets from digital twin simulations.
    
    Target: 120,000 digital twin samples across 5 crops.
    """
    
    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = output_dir
        self.generated_samples = 0
        
    def generate_dataset(self, 
                        target_samples: int = 120000,
                        crops: List[CropType] = None,
                        simulation_duration_hours: int = 336) -> pd.DataFrame:
        """
        Generate complete synthetic dataset from digital twin simulations.
        """
        if crops is None:
            crops = list(CropType)
        
        samples_per_crop = target_samples // len(crops)
        timesteps_per_sim = int(simulation_duration_hours * 60 / 5)  # 5-min timesteps
        simulations_per_crop = max(1, samples_per_crop // timesteps_per_sim)
        
        all_data = []
        
        print(f"Generating {target_samples} samples across {len(crops)} crops...")
        print(f"  {simulations_per_crop} simulations per crop")
        print(f"  {timesteps_per_sim} timesteps per simulation")
        
        for crop_idx, crop_type in enumerate(crops):
            print(f"\nProcessing {crop_type.value}...")
            
            crop_data = []
            scenarios = ScenarioGenerator.generate_scenario_matrix(
                crops=[crop_type],
                scenarios_per_crop=simulations_per_crop
            )
            
            for sim_idx, scenario in enumerate(scenarios[:simulations_per_crop]):
                twin = DigitalTwinEnvironment(crop_type)
                sim_df = twin.run_simulation(simulation_duration_hours, scenario)
                
                # Add scenario metadata
                sim_df['scenario_id'] = scenario.scenario_id
                sim_df['temp_profile'] = scenario.temperature_profile
                sim_df['humidity_profile'] = scenario.humidity_profile
                sim_df['initial_condition'] = scenario.initial_quality
                
                crop_data.append(sim_df)
                
                if (sim_idx + 1) % 10 == 0:
                    print(f"  Completed {sim_idx + 1}/{simulations_per_crop} simulations")
            
            crop_df = pd.concat(crop_data, ignore_index=True)
            all_data.append(crop_df)
            
            print(f"  Generated {len(crop_df)} samples for {crop_type.value}")
        
        # Combine all crops
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Shuffle for training
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.generated_samples = len(full_df)
        print(f"\nTotal samples generated: {self.generated_samples}")
        
        return full_df
    
    def generate_balanced_dataset(self, 
                                  total_samples: int = 162500,
                                  class_distribution: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate class-balanced dataset for training.
        
        Target distribution:
        - GOOD: 35%
        - NOT_GOOD (MARGINAL + AT_RISK): 43%  
        - SEVERELY_SPOILED (CRITICAL + SPOILED): 22%
        """
        if class_distribution is None:
            class_distribution = {
                'GOOD': 0.35,
                'MARGINAL': 0.22,
                'AT_RISK': 0.21,
                'CRITICAL': 0.12,
                'SPOILED': 0.10
            }
        
        # Generate excess data then downsample
        excess_factor = 1.5
        raw_df = self.generate_dataset(int(total_samples * excess_factor))
        
        # Balance classes
        balanced_samples = []
        for status, proportion in class_distribution.items():
            target_count = int(total_samples * proportion)
            status_df = raw_df[raw_df['quality_status'] == status]
            
            if len(status_df) >= target_count:
                sampled = status_df.sample(n=target_count, random_state=42)
            else:
                # Oversample if insufficient
                sampled = status_df.sample(n=target_count, replace=True, random_state=42)
            
            balanced_samples.append(sampled)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nBalanced dataset distribution:")
        print(balanced_df['quality_status'].value_counts(normalize=True))
        
        return balanced_df


if __name__ == "__main__":
    # Demo: Generate sample data for one crop
    print("AgriSense Digital Twin Demo")
    print("=" * 50)
    
    # Create environment for avocado
    twin = DigitalTwinEnvironment(CropType.AVOCADO)
    
    # Generate scenario
    scenario = ScenarioGenerator.generate_single_scenario(
        CropType.AVOCADO, 
        profile='optimal'
    )
    
    # Run 24-hour simulation
    print(f"\nRunning 24-hour simulation for Avocado...")
    data = twin.run_simulation(24, scenario)
    
    print(f"Generated {len(data)} sensor readings")
    print(f"\nSample columns: {list(data.columns[:15])}")
    print(f"\nQuality progression:")
    print(data[['hour', 'temperature', 'humidity', 'quality_index', 'quality_status']].iloc[::50])
    
    print(f"\nFinal state:")
    print(twin.get_current_state())
