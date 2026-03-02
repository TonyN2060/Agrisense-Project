import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ColdRoomEnv(gym.Env):
    """
    Gymnasium environment simulating cold room physics.
    Used for RL training and data generation.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.room_volume = config.get('room_volume', 100)  # m³
        self.insulation_R = config.get('insulation_R', 3.5)  # R-value
        self.ambient_temp = config.get('ambient_temp', 25.0)  # °C
        self.target_temp = config.get('target_temp', 2.0)  # °C
        self.compressor_power = config.get('compressor_power', 5.0)  # kW
        self.timestep_minutes = 5
        
        # State variables
        self.current_temp = self.target_temp
        self.compressor_on = False
        self.door_open = False
        self.time_elapsed = 0
        
        # Action space: 0=OFF, 1=ON, 2=ADJUST_SETPOINT_UP, 3=ADJUST_SETPOINT_DOWN
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [temp, target_temp, time_of_day, door_status]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, 0.0, 0.0]),
            high=np.array([30.0, 30.0, 24.0, 1.0]),
            dtype=np.float32
        )
        
        # Energy tracking
        self.energy_consumed = 0.0
        self.temperature_violations = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_temp = self.target_temp + np.random.uniform(-1, 1)
        self.compressor_on = False
        self.door_open = False
        self.time_elapsed = 0
        self.energy_consumed = 0.0
        self.temperature_violations = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Parse action
        if action == 0:
            self.compressor_on = False
        elif action == 1:
            self.compressor_on = True
        elif action == 2:
            self.target_temp = min(self.target_temp + 0.5, 5.0)
        elif action == 3:
            self.target_temp = max(self.target_temp - 0.5, -2.0)
        
        # Simulate door openings (random events)
        if np.random.random() < 0.05:  # 5% chance per timestep
            self.door_open = True
            door_duration = np.random.randint(1, 4)  # 1-3 minutes
        else:
            self.door_open = False
            door_duration = 0
        
        # Physics simulation
        self._update_temperature(door_duration)
        
        # Track energy
        if self.compressor_on:
            energy_timestep = (self.compressor_power * self.timestep_minutes) / 60.0
            self.energy_consumed += energy_timestep
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination (24 hours = 288 timesteps)
        self.time_elapsed += self.timestep_minutes
        terminated = self.time_elapsed >= 1440  # 24 hours
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _update_temperature(self, door_duration_minutes):
        """Simulate temperature dynamics using heat transfer principles."""
        
        # Heat gain from ambient (through walls)
        heat_gain_walls = (self.ambient_temp - self.current_temp) / self.insulation_R
        
        # Heat gain from door opening
        if self.door_open:
            heat_gain_door = door_duration_minutes * 2.0  # Simplified
        else:
            heat_gain_door = 0.0
        
        # Cooling from compressor
        if self.compressor_on:
            cooling_effect = -3.0  # °C per timestep when running
        else:
            cooling_effect = 0.0
        
        # Update temperature
        temp_change = (heat_gain_walls + heat_gain_door + cooling_effect) * (self.timestep_minutes / 60.0)
        self.current_temp += temp_change
        
        # Add noise
        self.current_temp += np.random.normal(0, 0.1)
    
    def _calculate_reward(self):
        """Reward function balancing temperature control and energy efficiency."""
        
        # Temperature deviation penalty
        temp_error = abs(self.current_temp - self.target_temp)
        if temp_error > 2.0:
            temp_penalty = -10.0
            self.temperature_violations += 1
        elif temp_error > 1.0:
            temp_penalty = -5.0
        else:
            temp_penalty = -temp_error
        
        # Energy penalty
        if self.compressor_on:
            energy_penalty = -0.5
        else:
            energy_penalty = 0.0
        
        # Bonus for stability
        stability_bonus = 2.0 if temp_error < 0.5 else 0.0
        
        return temp_penalty + energy_penalty + stability_bonus
    
    def _get_obs(self):
        hour_of_day = (self.time_elapsed % 1440) / 60.0
        return np.array([
            self.current_temp,
            self.target_temp,
            hour_of_day,
            float(self.door_open)
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            'energy_consumed': self.energy_consumed,
            'temperature_violations': self.temperature_violations,
            'compressor_on': self.compressor_on
        }