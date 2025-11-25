import numpy as np

class LinearRegression:
    def __init__(self, n_iters=100, lr=0.01, verbose=False):
        self.n_iters = n_iters
        self.lr = lr
        self.verbose = verbose
    
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
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        assert n_samples == len(y), "Input and output have different sample size."

        ## 1. Initialize w and b
        self._weights = np.zeros(n_features)
        self._bias = 0
        self._costs = []
        
        # Iterative learning
        for ep in range(self.n_iters):
            ## 2. Compute the predicted value
            pred_y = X @ self._weights + self._bias

            ## 3. Gradient Descent for update w, b
            error = pred_y-y
            cost = (1 / (2 * n_samples)) * np.sum(error**2)
            self._costs.append(cost)

            ## derivatives of w
            dw = (2 / n_samples) * np.sum(np.dot(X.T, error)) 
            ## derivatives of b
            db = (2 / n_samples) * np.sum(error)


            ## 4. Update W and B
            self._weights -= self.lr * dw
            self._bias -= self.lr * db

            ## 5. Print Cost per n epochs
            if ep % 100  ==0:
                if self.verbose == True:
                    print(f"Cost after iteration {ep}: {cost}")



    def predict(self, X):
        X = np.array(X)
        pred_y = np.dot(X, self._weights) + self._bias
        return pred_y