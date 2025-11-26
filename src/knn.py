import numpy as np
from collections import Counter


def euclidean_distance(self, p1, p2):
    """
    sqrt of sum (p1 - p2) **2
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    return np.sqrt((np.sum(p1-p2)**2))

class KNN:
    def __init__(self, k: int =5):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        

        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples")

        print(f"Training KNN with k={self.k}: Storing {len(X)} samples.")
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        X = np.asarray(X)
        # If a single 1D sample is passed, reshape to (1, n_features)
        if x.ndim ==1:
            x = x.reshape(1, -1)

        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)


    def _predict_one(self, x):
        distances = []

        ## 1. Calculate distance to all training points
        for i, x_train in enumerate(self.X_train):
            dist = euclidean_distance(x_train, x)
            distances.append((dist, self.y_train[i]))
        

        ## 2. Sort by distance and get k nearest neighbors
        ## Sort based on the first element of the tuple (the distance)
        distances.sort(key=lambda t: t[0])
        k_nearest = distances[:self.k]
        

        ## 3. Perform majority vote
        # Extract the labels of the k neighbors
        k_nei_labels = [label for _, label in k_nearest]
        most_comm = Counter(k_nei_labels).most_common(1)
        predicted_class = most_comm[0][0]
        return predicted_class



class ANN(KNN):
    """
    Approximate Nearest Neighbors (ANN):
    Instead of scanning all points, we quickly find a small candidate set that probably contains the true nearest neighbors, and only compute distances within that set.
    
    LSH-style random hyperplane hashing
    1. During fit:
        - Draw L random hyperplanes in R^d -> m matrix W ∈ ℝ^{L×d}
        - For each training point x:
            Compute W @ x
            Take the sign of each component -> bit string like "101010..." -> this is its hash bucket
        - Store indices of points per bucket: bucket -> [indices]
    2. During prediction for query x*:
        - Compute its hash bucket
        - Only consider points in that bucket as KNN candidates
            -> massive speedup if there are many buckets vs N
        - If bucket is empty or too small, we can fallback to full search so it's safe.
    """
    def __init__(self, k:int =5, n_planes: int =10, min_candidates: int= None, random_state: int=43):
        self.k = k
        self.n_planes = n_planes
        self.min_candidates = min_candidates
        self.random_state = random_state


        self.X_train = None
        self.y_train = None
        self.W = None
        self.buckets = {}


    def _init_random_planes(self, n_features: int):
        rng = np.random.default_rng(self.random_state)
        # Gaussian random hyperplanes
        self.W = rng.normal(size=(self.n_planes, n_features))

    def _hash_vector(self, x):
        """
        Compute hash bitstring for a single vector x using the sign of W @ x.
        """
        x = np.asarray(x)
        proj = self.W @ x  # shape: (n_planes,)
        bits = (proj >= 0).astype(int)
        # Convert to string like '101010'
        return ''.join(bits.astype(str))

    def _build_buckets(self):
        """
        Build bucket dict from X_train using W.
        """
        self.buckets = {}
        for idx, x in enumerate(self.X_train):
            h = self._hash_vector(x)
            if h not in self.buckets:
                self.buckets[h] = []
            self.buckets[h].append(idx)

    def fit(self, X, y):
        """
        Store data, initialize random hyperplanes, and build hash buckets.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}"
            )

        self.X_train = X
        self.y_train = y

        n_samples, n_features = X.shape
        print(f"[ANN_KNN] Storing {n_samples} samples, dim={n_features}, k={self.k}, n_planes={self.n_planes}.")

        self._init_random_planes(n_features)
        self._build_buckets()



    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = [self._predict_one(x) for x in X]
        return np.array(preds)
    

    def _predict_one(self, x):
        # 1. Hash query to a bucket
        h = self._hash_vector(x)
        candidate_indices = self.buckets.get(h, [])

        # 2. If too few candidates, fall back to brute force over all points
        if len(candidate_indices) < self.min_candidates:
            candidate_indices = range(len(self.X_train))

        # 3. Compute distances only to candidates
        distances = []
        for idx in candidate_indices:
            x_train = self.X_train[idx]
            dist = euclidean_distance(x_train, x)
            distances.append((dist, self.y_train[idx]))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]

        labels = [label for _, label in k_nearest]
        most_common = Counter(labels).most_common(1)
        return most_common[0][0]