import numpy as np
import pandas as pd

class Biosensor:
    def __init__(self, sensor_type="glucose"):
        self.sensor_type = sensor_type
        self.calibration = {
            "glucose": {"sensitivity": 0.05, "baseline": 0.1},
            "oxygen": {"sensitivity": 0.03, "baseline": 0.05},
            "lactate": {"sensitivity": 0.04, "baseline": 0.08}
        }
    
    def generate_response(self, true_concentration, time_hours=6, noise_level=0.02):
        """Generate realistic sensor response with noise and drift"""
        time_points = np.linspace(0, time_hours, len(true_concentration))
        
        # Get sensor parameters
        params = self.calibration.get(self.sensor_type, self.calibration["glucose"])
        
        # Ideal response (linear with saturation)
        ideal_response = params["sensitivity"] * true_concentration + params["baseline"]
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, len(ideal_response))
        
        # Add sensor drift (linear)
        drift = 0.001 * time_points
        
        # Add time delay (sensor response time)
        delay_samples = min(10, len(ideal_response) // 10)
        ideal_response = np.roll(ideal_response, delay_samples)
        ideal_response[:delay_samples] = params["baseline"]
        
        # Combine effects
        sensor_output = ideal_response + noise + drift
        sensor_output = np.maximum(sensor_output, 0)  # Non-negative
        
        return {
            'time': time_points,
            'true_concentration': true_concentration,
            'sensor_output': sensor_output,
            'ideal_response': ideal_response
        }
    
    def create_sample_data(self, time_points=500):
        """Create sample concentration data"""
        time = np.linspace(0, 6, time_points)  # 6 hours
        
        # Simulate realistic concentration variations
        base = 5.0  # Baseline concentration
        circadian = 2.0 * np.sin(2 * np.pi * time / 3)  # Circadian rhythm
        meals = 3.0 * np.exp(-((time - 1.5) / 0.5) ** 2)  # Meal response
        meals += 2.5 * np.exp(-((time - 4.0) / 0.5) ** 2)  # Second meal
        noise = 0.5 * np.random.normal(0, 1, time_points)  # Random variations
        
        concentration = base + circadian + meals + noise
        concentration = np.maximum(concentration, 0)  # Ensure non-negative
        
        return concentration
