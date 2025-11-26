import numpy as np


class StandardScaler:
    def __init__(self):
        self._mean = None
        self._scale = None

        
    def fit(self, X):
        X = np.asarray(X)
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0, ddof=0)
        # Replace zeros to avoid division-by zero
        self._scale[self._scale==0] = 1.0
        return self

    def transform(self, X):
        """
        Apply standardization: (X - mean) / std
        """
        if self._mean is None or self._scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X = np.asarray(X)
        return (X-self._mean) / self._scale

    
    def inverse_transform(self, X_scaled):
        """
        Reconstruct original values: X_scaled * std + mean
        """
        if not self._mean or not self._scale:
            raise ValueError("Scaler has not been fitted.")
        X_scaled = np.asarray(X_scaled)
        return X_scaled * self._scale + self._mean


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)