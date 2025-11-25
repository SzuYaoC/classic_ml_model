import numpy as np

def sigmoid(z):
    return 1 / (1+np.exp(-z))    


class LogisticRegression:
    def __init__(self, n_iters=100, lr=0.01, verbose=False):
        self.n_iters = n_iters
        self.lr = lr
        self.verbose= verbose


        self._weights = None
        self._bias = None
        self._costs = []


    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias
    

    @property
    def costs(self):
        return self._costs

    
    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,) or (n_samples, 1)
        """

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Ensure y is shape (n_samples,)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        
        # 1. Initialize parameters
        self._weights = np.zeros(n_features)
        self._bias = 0.0
        self._costs = []


        ## 2. Compute cost (with clipping for numerical stability)
        # eps = 1e-15
        # y_pred_clipped = np.clip(y_pred, eps, 1-eps)
        # cost = -1.0 / n_samples * np.sum(y * np.log(y_pred_clipped) + (1-y) * np.sum(1-y_pred_clipped))
        # dw = (1.0/n_samples) * np.dot(X.T, (y_pred_clipped-y))
        # db = (1.0/n_samples) * np.sum(y_pred_clipped-y)
        # self._weights -= self.lr * dw
        # self._bias -= self.lr * db



        # 2. Gradient Descent
        for ep in range(self.n_iters):
            z = np.dot(X, self._weights) + self._bias
            y_pred = sigmoid(z)

            ## calculate Cost (Log loss)
            cost = -1 / n_samples * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
            self._costs.append(cost)  

            ## derivative of cost with respect to weights (dw)
            dw = (1 / n_samples) * X.T @ (y_pred- y)

            ## derivative of cost with respect to bias (db)
            db = (1 / n_samples) * np.sum(y_pred-y)

            ## update parameters 
            self._weights -= self.lr * dw
            self._bias -= self.lr * db


            if ep % 100 == 0 :
                if self.verbose == True:
                    print(f"Cost after iteration {ep}: {cost}")



    def predict_proba(self, X):
        X = np.array(X)

        ## Compute the predicted probability
        z = np.dot(X, self._weights) + self._bias
        y_pred = sigmoid(z)
        return y_pred


    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        return (y_pred >= threshold).astype(int)
    


