import numpy as np


class KMeans:
    def __init__(self, n_clusters:int, max_iters:int=100, tol: float = 1e-4, random_state:int =101):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.tol = tol
        self._centroids = None
        self._labels = None


    @property
    def centroids(self):
        return self._centroids
    
    @property
    def labels(self):
        return self._labels
    

    def fit(self, X):
        
        # set random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # 1. Initialize centroid
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self._centroids = X[random_indices]


        # 2. Assign and Update loop
        for i in range(self.max_iters):
            ## store current centroids to check for convergence
            old_centroids = self._centroids.copy()

            # Assignment Step: distance from each centroid to each point
            # shape:
            #   self._centroids[:, np.newaxis] -> (k, 1, d)
            #   X       -> (n, d)
            #   result:  (k, n, d)
            distance = np.sqrt(((X - self._centroids[:, np.newaxis])**2).sum(axis=2))

            ## # Assign each point to the closest centroid
            self._labels = np.argmin(distance, axis=0)

            ## Update Step
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                # points assigned to cluster k
                cluster_points = X[self._labels == k]

                # Check if the cluster is non-empty before calculate the mean
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = self._centroids[k]
            
            self._centroids = new_centroids

            ## 3. check for convergence
            ## np.allclose(A, B) checks whether all elements in arrays A and B are numerically close within a small tolerance.
            ## If the centroids did not move anymore, the algorithm has converged.
            shift = np.linalg.norm(self._centroids - old_centroids)
            if shift < self.tol:
                print(f"Converged after {i+1} iteration.")
                break

        return self ## retrun the fitted object
    


    def predict(self, X):
        X = np.asarray(X)
        distance = np.sqrt(((X - self._centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distance, axis=0)
        return labels
    