import numpy as np
from collections import Counter
from src.decision_tree import DecisionTree


class RandomForest:
    """
    To turn that single tree into a Random Forest, we need to implement two specific mechanism:
    1. Bootstrapping
    2. Aggregation (Voting)
    """
    def __init__(self, n_trees:int =10, max_depth: int=10, min_samples_split: int = 2, n_features:int=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.trees = []

    def fit(self, X, y):
        ## Reset trees
        self.tress = []
        for _ in range(self.n_trees):
            # A. Create a new tree model
            model = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )

            # B. Bootstrap sampling (bagging)
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # C. Train the tree on this random subset
            model.fit(X_sample, y_sample)
            self.trees.append(model)

    
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # Random indices with replacement
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    

    def predict(self, X):
        # A. Get predictions from every single tree
        # Result shape: [n_tree, n_samples]
        tree_preds = np.array([tree.predict(X) for tree in self.trees]) 

        #B. Swap axes to group by sample -> [n_samp;es. n_trees]
        # So tree_pred[0] = [vote_tree1, vote_tree2,....]
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # C. Majority Vote (hard voting)
        y_pred = [self._most_common_label(pred) for pred in tree_preds]
        return np.array(y_pred)
    

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
