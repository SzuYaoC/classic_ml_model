## A. Conceptual & Intuition

#### 1.	What is linear regression and what assumptions does it make?

Linear Regression is a supervised learning method that models a continuous target as a linear combination of input features, finding the best-fit line (or hyperplane) that minimizes the sum of squared errors between predicted and actual values.

The key assumptions are: (1) linear relationship between predictors and target, (2) independent errors, (3) constant variance of errors, (4) no multicollinearity, and (5) normally distributed residuals for inference. Violations of these assumptions affect coefficient stability, interpretability, and statistical validity.”


#### 2.	What does the model actually learn?
Linear regression learns the weight vector and bias that define the best-fitting linear relationship between features and the target. The weights represent how much each feature influences the output, and they are chosen to minimize the squared error. 

#### 3.	Why do we use least squares? Is it the only choice?
Least squares is used because it is mathematically convenient, computationally efficient, statistically optimal under reasonable assumptions, and has a closed-form solution. It is not the only choice—different loss functions like L1, Huber, or quantile loss are used when we want robustness to outliers, different noise assumptions, or different prediction objectives.

** A closed-form solution is an explicit formula that computes the answer directly, without requiring iteration or numerical optimization.**

#### 4.	What does the coefficient $w_i$ mean in a linear regression model?
The coefficien represents how much the predicted output changes when feature $x_i$ increases by 1 unit, holding all other features constant. It’s the slope of the model along that feature’s direction. Positive weights increase the target, negative weights decrease it, and zero weights mean no effect. This interpretation breaks down if features are highly correlated.

#### 5.	How do you interpret the intercept?
The intercept is the model’s prediction when all features are zero. It represents the baseline level of the target before considering the effect of any feature. Its interpretability depends on whether ‘zero’ is meaningful for the features, and after standardization, the intercept represents the prediction at the mean of the data.



## B. Math, Optimization, and Theory
#### 1.	Derive the normal equation for linear regression.

* Object Function: $\hat{Y} = w_1X_1+ w_2X_2 + ....+ w_nX_n + b$
* Loss Function: $J(w) = \frac{1}{2n}(y-Xw)^T(y-Xw)$
    Take the gradient w.r.t $w$:
        $$\nabla_w J(w) = \frac{1}{2n} \cdot 2 X^T(Xw - y) = \frac{1}{n} X^T(Xw - y)$$
    
    Set gradient to zero at the optimum:
        $$X^T(Xw-y)=0$$

        $$X^{T}Xw = X^Ty$$

* Normal Equation: $w = (X^{-T}X)^{-1}X^{T}y


#### 2.	Why does the normal equation require X^TX to be invertible?
The normal equation is:

$$X^T X w = X^T y$$

To solve for w, we write:

$$w = (X^T X)^{-1} X^T y$$


#### 3.	What happens if X^TX is not invertible?
* The columns of X are linearly dependent
→ one feature is a linear combination of others (multicollinearity).
* The model parameters are not identifiable
→ infinitely many solutions for w.
* The projection subspace is not uniquely defined
→ you cannot uniquely project y onto the column space.

When $X^TX$ is invertible, the least-squares solution is unique and well-defined. If it’s not, we use a pseudoinverse or regularization (e.g., Ridge).


#### 4.	Explain the relationship between linear regression and MLE.
Linear regression becomes Maximum Likelihood Estimation when we assume the errors are Gaussian.
Start with the model:

$$y = Xw + \epsilon$$

Assume the noise terms are i.i.d.:

$$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Then each $y_i$ is normally distributed around $X_i w$. The likelihood of the data is:

$$L(w) = \prod_{i=1}^n \mathcal{N}(y_i \mid X_i w, \sigma^2)$$

Taking the log (for simplicity):

$$\log L(w) = -\frac{1}{2\sigma^2} \sum (y_i - X_i w)^2 + \text{const}$$

Maximizing this log-likelihood is equivalent to minimizing the sum of squared errors:

$$\min_w \sum (y - Xw)^2$$

which is exactly the ordinary least squares objective.


#### 5.	What is multicollinearity? How does it affect the model?
Multicollinearity means that two or more features in X are highly linearly correlated or one feature is an exact linear combination of others. In matrix terms, the columns of X are not independent, making $X^TX$ close to singular.

How it affects the model:
* Unstable coefficients: small changes in data cause large changes in estimated w_i.
* Inflated variances: coefficient estimates become noisy, leading to wide confidence intervals and unreliable p-values.
* Interpretation breaks: it’s hard to interpret “holding all else constant” when features move together.
* Normal equation may fail: if multicollinearity is perfect, X^TX is non-invertible and the OLS solution is not unique.



#### 6.	What’s the difference between OLS and Weighted Least Squares?
OLS treats all observations equally, minimizing:

$$\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

This assumes all errors have equal variance (homoscedasticity).


Weighted Least Squares (WLS) assigns a weight $w_i$ to each observation:

$$\sum_{i=1}^n w_i\, (y_i - \hat{y}_i)^2$$

This is used when errors have different variances (heteroscedasticity). Observations with lower variance (more reliable data) are given higher weight, and noisy points get lower weight.

In short:
	•	OLS: assumes constant error variance
	•	WLS: allows varying error variance and gives more trust to high-quality observations



## C. Practical & ML Engineering
	1.	Why do we need feature scaling for linear regression?
	2.	How do outliers affect linear regression? How do you fix it?
	3.	When would you choose gradient descent over the normal equation?
	4.	How do you regularize linear regression?
(L1, L2, ElasticNet — and when to use each)
	5.	How do you detect overfitting in linear regression?
	6.	How do you handle categorical variables?
	7.	Suppose your model is underperforming — what are the first 3 things you check?


D. Code / Implementation
	1.	Implement linear regression using gradient descent.
	2.	Why might gradient descent fail to converge?
	3.	How do you vectorize the gradient descent update?
	4.	Compare your implementation to scikit-learn’s.


🔥 Advanced Linear Regression Questions
	1.	Why is Ridge Regression equivalent to MAP with a Gaussian prior?
	2.	Why does Lasso lead to sparse solutions? (Geometry + L1 norm)
	3.	Explain the bias–variance trade-off in linear regression.
	4.	Why does linear regression fail in high dimensions?
	5.	What is the difference between R² and Adjusted R²?
	6.	How does linear regression behave with correlated errors?