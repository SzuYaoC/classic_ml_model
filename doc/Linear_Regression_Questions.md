## A. Conceptual & Intuition

* 1.	What is linear regression and what assumptions does it make?
    Linear Regression is a supervised learning method that models a continuous target as a linear combination of input features, finding the best-fit line (or hyperplane) that minimizes the sum of squared errors between predicted and actual values.

    The key assumptions are: (1) linear relationship between predictors and target, (2) independent errors, (3) constant variance of errors, (4) no multicollinearity, and (5) normally distributed residuals for inference. Violations of these assumptions affect coefficient stability, interpretability, and statistical validity.”


* 2.	What does the model actually learn?
    Linear regression learns the weight vector and bias that define the best-fitting linear relationship between features and the target. The weights represent how much each feature influences the output, and they are chosen to minimize the squared error. 

* 3.	Why do we use least squares? Is it the only choice?
    Least squares is used because it is mathematically convenient, computationally efficient, statistically optimal under reasonable assumptions, and has a closed-form solution. It is not the only choice—different loss functions like L1, Huber, or quantile loss are used when we want robustness to outliers, different noise assumptions, or different prediction objectives.
    ** A closed-form solution is an explicit formula that computes the answer directly, without requiring iteration or numerical optimization.**

* 4.	What does the coefficient $w_i$ mean in a linear regression model?
    The coefficien represents how much the predicted output changes when feature $x_i$ increases by 1 unit, holding all other features constant. It’s the slope of the model along that feature’s direction. Positive weights increase the target, negative weights decrease it, and zero weights mean no effect. This interpretation breaks down if features are highly correlated.

* 5.	How do you interpret the intercept?
    The intercept is the model’s prediction when all features are zero. It represents the baseline level of the target before considering the effect of any feature. Its interpretability depends on whether ‘zero’ is meaningful for the features, and after standardization, the intercept represents the prediction at the mean of the data.



## B. Math, Optimization, and Theory
	1.	Derive the normal equation for linear regression.
	2.	Why does the normal equation require X^TX to be invertible?
	3.	What happens if X^TX is not invertible?
	4.	Explain the relationship between linear regression and MLE.
	5.	Why is the least squares loss convex?
	6.	What is heteroscedasticity?
	7.	What is multicollinearity? How does it affect the model?
	8.	What’s the difference between OLS and Weighted Least Squares?
	9.	What is the Gauss-Markov theorem?


C. Practical & ML Engineering
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