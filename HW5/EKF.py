import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self, x0=0, y0=0, yaw0=0):
        # State vector: [x, y, yaw] in GLOBAL frame
        self.pose = np.array([x0, y0, yaw0], dtype=float)
        
        # State covariance matrix (3x3)
        self.S = np.identity(3) * 1.0
        
        # Observation matrix Jacobian (2x3): GPS measures [x, y]
        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)
        
        # Process noise (3x3) - will be updated from radar covariance
        self.R = np.identity(3) * 0.1
        
        # Measurement noise (2x2) - will be updated from GPS covariance
        self.Q = np.identity(2) * 1.0
        
        print("Initialize EKF")
    
    def motion_model(self, pose, u):
        """
        Nonlinear Motion Model
        
        input:
            u = [u_x, u_y, u_yaw]  <- displacement in LOCAL frame
        
        Global frame update：
            x_new   = x   + u_x * cos(yaw) - u_y * sin(yaw)
            y_new   = y   + u_x * sin(yaw) + u_y * cos(yaw)
            yaw_new = yaw + u_yaw
        """
        x, y, yaw = pose
        u_x, u_y, u_yaw = u
        
        x_new   = x   + u_x * cos(yaw) - u_y * sin(yaw)
        y_new   = y   + u_x * sin(yaw) + u_y * cos(yaw)
        yaw_new = yaw + u_yaw
        
        return np.array([x_new, y_new, yaw_new])
    
    def compute_jacobian_A(self, pose, u):
        """
        Jacobian to Motion Model with state [x, y, yaw]  
        
        ∂f/∂x =
        [1,  0,  -u_x*sin(yaw) - u_y*cos(yaw)]
        [0,  1,   u_x*cos(yaw) - u_y*sin(yaw)]
        [0,  0,   1                            ]
        """
        yaw = pose[2]
        u_x, u_y, _ = u
        
        A = np.array([
            [1, 0, -u_x * sin(yaw) - u_y * cos(yaw)],
            [0, 1,  u_x * cos(yaw) - u_y * sin(yaw)],
            [0, 0,  1]
        ], dtype=float)
        
        return A
    
    def predict(self, u):
        """
        EKF Prediction Step
        
        1. update state from nonlinear motion model
        2. update covariance with Jacobian
        
        Args:
            u: [u_x, u_y, u_yaw] in LOCAL frame
        """
        # ---- Step 1: Compute Jacobian A at current state ----
        A = self.compute_jacobian_A(self.pose, u)
        
        # ---- Step 2: Propagate state through nonlinear model ----
        self.pose = self.motion_model(self.pose, u)
        
        # ---- Step 3: Normalize yaw ----
        self.pose[2] = atan2(sin(self.pose[2]), cos(self.pose[2]))
        
        # ---- Step 4: Update covariance with Jacobian ----
        # S = A * S * A^T + R
        self.S = A @ self.S @ A.T + self.R
        
        return self.pose, self.S
    
    def observation_model(self, pose):
        """
        observation：GPS -> global x, y
        h(x) = [x, y]
        """
        return self.C @ pose
    
    def update(self, z):
        """
        EKF Update Step (GPS)
        
            h(x) = C * x = [x, y]
            Jacobian C = [[1, 0, 0],
                          [0, 1, 0]]
        
        Args:
            z: [gps_x, gps_y]
        """
        # ---- Step 1: Predicted measurement ----
        z_pred = self.observation_model(self.pose)   # shape (2,)
        
        # ---- Step 2: Innovation ----
        innov = z - z_pred                           # shape (2,)
        
        # ---- Step 3: Innovation covariance ----
        # S_z = C * S * C^T + Q
        S_z = self.C @ self.S @ self.C.T + self.Q   # shape (2,2)
        
        # ---- Step 4: Kalman Gain ----
        # K = S * C^T * S_z^{-1}
        K = self.S @ self.C.T @ np.linalg.inv(S_z)  # shape (3,2)
        
        # ---- Step 5: Update state ----
        self.pose = self.pose + K @ innov            # shape (3,)
        self.pose[2] = atan2(sin(self.pose[2]), cos(self.pose[2]))
        
        # ---- Step 6: Update covariance (Joseph form for stability) ----
        I = np.identity(3)
        IKC = I - K @ self.C
        # Joseph form: (I-KC)*S*(I-KC)^T + K*Q*K^T
        self.S = IKC @ self.S @ IKC.T + K @ self.Q @ K.T
        
        return self.pose, self.S
