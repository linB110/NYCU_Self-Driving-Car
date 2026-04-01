import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1, 0, 0], [0, 1, 0]])
        
        # State transition error with 0 mean 1 variance
        self.R = np.identity(3) * 0.001
        
        # Measurement error with # 0 mean 3 variance
        self.Q = np.identity(2) * 1000
        
    def predict(self, u):
        #state predice
        self.state = self.A @ self.state + self.B @ u
        
        # covariance predict
        self.S = self.A @ self.S + self.R
        

    def update(self, z):
        # Kalman Gain
        K = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
    
        # state update
        self.state = self.state + K @ (z - self.C @ self.state)
    
        # covariance update
        I = np.identity(len(self.state))
        self.S = (I - K @ self.C) @ self.S
    
        return self.state, self.S
