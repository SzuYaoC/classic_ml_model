import numpy as np

class PCA:
    def __init__(self, n_components: int):
        """
        n_components: number of principal components to keep
        """
        self.n_components = n_components
        self.mean_ = None           # feature-wise mean (for centering)
        self.components_ = None     # principal axes (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Compute the principal components from the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        # 1. Convert to numpy array
        X = np.asarray(X)

        # 2. Compute mean and center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 3. Compute SVD on centered data
        # X_centered = U * S * Vt
        # Rows of Vt are principal directions
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 4. Select the top n_components principal axes
        self.components_ = Vt[:self.n_components]

        # 5. Compute explained variance (eigenvalues)
        # eigenvalues for PCA from SVD: (S^2) / (n_samples - 1)
        n_samples = X.shape[0]
        eigenvalues = (S ** 2) / (n_samples - 1)

        # Take the top n_components eigenvalues
        self.explained_variance_ = eigenvalues[:self.n_components]

        # 6. Explained variance ratio
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_pca : array, shape (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit(X) first.")

        X = np.asarray(X)
        X_centered = X - self.mean_
        # Projection: X_centered · components^T
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA on X, then return the transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate original data from its PCA representation.

        Parameters
        ----------
        X_pca : array-like, shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : array, shape (n_samples, n_features)
        """
        X_pca = np.asarray(X_pca)
        # X_reconstructed_centered = X_pca · components
        X_reconstructed_centered = X_pca @ self.components_
        # Add the mean back
        return X_reconstructed_centered + self.mean_