import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SensorCalibrator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_training_data(self, sensor_readings, true_concentrations, window_size=10):
        """Prepare data for machine learning training"""
        X, y = [], []
        
        for i in range(len(sensor_readings) - window_size):
            # Use window of sensor readings as features
            features = sensor_readings[i:i + window_size]
            # Predict the true concentration at the end of the window
            target = true_concentrations[i + window_size]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_model(self, sensor_readings, true_concentrations, model_type='random_forest'):
        """Train calibration model"""
        # Prepare data
        X, y = self.prepare_training_data(sensor_readings, true_concentrations)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose and train model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'model_type': model_type
        }
    
    def calibrate_readings(self, sensor_readings):
        """Apply calibration to new sensor readings"""
        if not self.is_trained:
            raise ValueError("Model must be trained before calibration")
        
        window_size = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 10
        calibrated = []
        
        for i in range(len(sensor_readings)):
            if i < window_size:
                # Not enough data for prediction, use raw reading
                calibrated.append(sensor_readings[i])
            else:
                # Use window for prediction
                features = sensor_readings[i - window_size:i]
                features_scaled = self.scaler.transform([features])
                prediction = self.model.predict(features_scaled)[0]
                calibrated.append(prediction)
        
        return np.array(calibrated)

class BiomarkerClassifier:
    def __init__(self):
        self.thresholds = {
            'glucose': {'normal': (70, 100), 'high': (100, 125), 'critical': (125, 300)},
            'oxygen': {'normal': (95, 100), 'low': (90, 95), 'critical': (0, 90)}
        }
    
    def classify(self, concentrations, biomarker_type='glucose'):
        """Classify biomarker levels"""
        if biomarker_type not in self.thresholds:
            biomarker_type = 'glucose'
        
        thresholds = self.thresholds[biomarker_type]
        classifications = []
        
        for conc in concentrations:
            if thresholds['normal'][0] <= conc <= thresholds['normal'][1]:
                classifications.append('Normal')
            elif thresholds['high'][0] <= conc <= thresholds['high'][1]:
                classifications.append('High')
            else:
                classifications.append('Critical')
        
        return classifications
