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


        ## 2. Compute cost (with clipping for numerical stability)
        # eps = 1e-15
        # y_pred_clipped = np.clip(y_pred, eps, 1-eps)
        # cost = -1.0 / n_samples * np.sum(y * np.log(y_pred_clipped) + (1-y) * np.sum(1-y_pred_clipped))
        # dw = (1.0/n_samples) * np.dot(X.T, (y_pred_clipped-y))
        # db = (1.0/n_samples) * np.sum(y_pred_clipped-y)
        # self._weights -= self.lr * dw
        # self._bias -= self.lr * db



        # 2. Gradient Descent
        costs = []
        for ep in range(self.n_iters):
            z = np.dot(X, self._weights) + self._bias
            y_pred = sigmoid(z)

            ## calculate Cost (Log loss)
            cost = -1 / n_samples * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

            ## Gradient Descent helper
            self.loss_function(X , y, y_pred)

            if ep % 100 == 0 :
                costs.append(cost)  
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
    

    ## helper
    def loss_function(self, X, y, pred_y):
        """
        Output
        ---- 
        cost (int)
        """ 

        assert pred_y.shape == y.shape, "Predicted Label and Truth Label have different sizes."

        n_samples = len(y)
        
        ##  derivative of cost with respect to weights (dw)
        dw = (1 / n_samples) * X.T @ (pred_y- y)

        ## derivative of cost with respect to bias (db)
        db = (1 / n_samples) * np.sum(pred_y-y)

        ## update parameters 
        self._weights -= self.lr * dw
        self._bias -= self.lr * db



