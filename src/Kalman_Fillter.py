import numpy as np
import matplotlib.pyplot as plt

class KF():
    def __init__(self, x0: float, y0: float, dot_x0: float, dot_y0: float, a_variance: float) -> None:
        
        #Mean of state 
        self.X = np.array([x0, y0, dot_x0, dot_y0]).reshape((4,1))
        
        #Covariance of state
        self.P = np.eye(4) 
        
        #Acceleation Variance
        self.Var_a = a_variance

    def Prediction(self, dt: float) -> None:
        """
            X = F * X \n
            P = F * P * F^T + G * G^T * a
        """
        F = np.array([[1, 0, dt, 0],[0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])
        G = np.array([[1/2 * dt**2, 0],[0, 1/2 * dt**2], [dt, 0], [0, dt]]).reshape((4,2))
        
        new_X = np.dot(F, self.X) 
        new_P = np.dot( np.dot(F,self.P), F.T ) + np.dot(G, G.T) * self.Var_a
        
        self.X = new_X
        self.P = new_P
    
    def Update(self, m_val: float, m_var: float):
        """
            y = z - H X         \n
            S = H * P * F^T + R \n
            K = P * H^T S^(-1)  \n
            X = X + K * y       \n
            P = (I - K * H) * P
        """
        self.X = self.X.reshape((4,1))
    
        z = np.array([m_val]).reshape((2,1))   #Measurement value
        R = np.array([[m_var[0],0],[0,m_var[1]]])   #Measurement variance
        
        H = np.array([[1,0,0,0],[0,1,0,0]])

        y = z - np.dot(H, self.X)
        S = np.dot( np.dot(H,self.P), H.T ) + R
        K = np.dot( np.dot(self.P, H.T), np.linalg.pinv(S))
        
        new_X = self.X + np.dot(K, y)
        new_P = np.dot(np.eye(4) - np.dot(K, H),self.P)
        
        self.X = new_X
        self.P = new_P
    
    @property
    def pos(self) -> float:
        return self.X[:2]

    @property
    def vel(self) -> float:
        return self.X[2:4]
    
    @property
    def cov(self) -> np.array:
        return self.P

    @property
    def mean(self) -> np.array:
        return self.X
    
        
    