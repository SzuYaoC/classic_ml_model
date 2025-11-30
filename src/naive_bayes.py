import numpy as np


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        """
        alpha: Smoothing parameter (Laplace Smoothing).
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None        # shape: (n_classes,)
        self.feature_log_prob_ = None       # shape: (n_classes, n_features)
    
    
    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes model according to X, y.

        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize storage for log probabilities
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))


        # Calculate stats for each class
        for idx, c in enumerate(self.classes_):
            # 1. Filter data for this specific class
            X_c = X[y == c]

            # 2. Calculate Class Prior (Log Space)
            # P(y) = Count(y) / Total Samples
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)


            ## 3. Calculate Feature Counts
            # Sum the word counts for this class across all documents
            # Total_count_per_word: vector of size (n_features, )
            total_count_per_word = np.sum(X_c, axis=0)

            ## 4. Apply Laplace Smoothing to Likelihoods
            # Numerator: Word Count + Alpha
            numerator = total_count_per_word + self.alpha

            # Denominator: Total Words in Class + (Alpha * Vocabulary Size)
            # We sum(total_count_per_word) to get total words in class
            denominator = np.sum(total_count_per_word) + (self.alpha * n_features)

            ## 5. Store as Log Probabilities (to prevent underflow)
            self.feature_log_prob_[idx] = np.log(numerator / denominator)

        ## We return self, so that we can enable "method chaining" in the pipeline
        ## e.g. model.fit(X,y).predict(X)
        return self 
    


    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return elf.classes_[np.argmax(log_probs, axis=1)]


    def predict_proba(self, X):
       """
        Calculates log posterior probability for each class.
        Formula: log P(y) + sum(log P(x_i | y))
        """
       # X shape: (n_samples, n_features)
       # feature_log_prob_ shape: (n_classes, n_features)
       # Matrix Multiplication automatically performs the "Sum of log-likelihoods"
       # The dot product effectively sums the weights for the features present in X
       # Shape Result: (n_samples, n_features)

       log_likelihood = np.dot(X, self.feature_log_prob_.T)
       return log_likelihood + self.class_log_prior_