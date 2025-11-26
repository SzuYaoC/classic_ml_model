## A. Conceptual & Intuition
#### 1.	What is logistic regression?
Logistic regression is a simple and effective classification model that predicts the probability of a binary outcome. It takes a linear combination of the features, passes it through a sigmoid function, and outputs a probability between 0 and 1. The model essentially learns how the inputs affect the likelihood of the positive class, and it’s trained by maximizing the likelihood or minimizing cross-entropy loss. It’s fast, interpretable, and widely used as a baseline classifier.

#### 2.	Why do we model log-odds instead of probability directly?
We use log-odds because probabilities are bounded between 0 and 1, but a linear function can output any real number. By converting probabilities into log-odds, which range from minus infinity to plus infinity, we can safely model them with a linear function. This keeps the math clean, the optimization convex, and the predictions consistent with probability constraints. The log-odds relationship is applied before the sigmoid.

#### 3.	Why do we need the sigmoid function?
The sigmoid function takes the model’s linear output and squashes it into a valid probability between 0 and 1. It gives us a smooth, differentiable mapping that’s perfect for gradient-based optimization, and it ties directly into the log-odds formulation used in maximum likelihood. Without the sigmoid, the model couldn’t produce probabilities in a stable, interpretable way.


#### 4.	Can logistic regression model non-linear boundaries?
On its own, logistic regression can only create linear decision boundaries. But if we add non-linear features—like polynomial terms or interactions—then the model can represent non-linear relationships. So the core model is linear, but with the right feature engineering, it can effectively handle non-linear patterns.



##  B. Probability, Loss Function, MLE
#### 1.	Derive the log-likelihood function of logistic regression.
In logistic regression, we model $p_i = P(y_i=1|x_i) = \sigmoid(w^Tx_i)$. Each label $y_i$ is assumed Bermoulli, so the likelihood for one point is $P_i^{y_i}(1-p_i)^(1-y_i)$. For all data points, the likelihood is the product over $i$ and if we take the log, that product becomes a sum. So the log0likelihood is 
$$ \ell(w) = \sum_i \big[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \big].$$

That's the function we maximize in logistic regression, and minimizing the negative of that gives us the usual cross-entropy loss.


#### 2.	Why don’t we use squared error for logistic regression?
We avoid squared error in logistic regression for a couple of reasons. First, with a sigmoid output, squared error leads to a non-convex loss surface, which makes optimization harder and less stable. Second, logistic regression is a probabilistic model with Bernoulli outputs, and the ‘correct’ loss from a likelihood perspective is the negative log-likelihood, which gives us cross-entropy. Using squared error breaks that probabilistic interpretation and usually gives worse gradients, especially when predictions are near 0 or 1.



#### 3.	Why is logistic loss convex?
The logistic loss is convex because when you write it as the negative log-likelihood of a Bermoulli with a sigmoid link, each term in the loss is a convex function of the linear score $z = W^TX$. If you take the second derivative or look at the Hessian, you get something like $x^TSX$, where $S$ is a diagonal matrix with positive entries $p_i(1 - p_i)$. That matrix is positive semidefinite, which means the loss has no local minima—only a single global minimum—so the optimization problem is convex.”

#### 4.	Why does logistic regression have no closed-form solution?
Logistic regression has no closed-form solution because the log-likelihood involves the sigmoid applied to a linear function of the parameters and then wrapped in a log. When you take the gradient and set it to zero, you end up with nonlinear equations in $w$ that don’t simplify to something like $(X^T X)^{-1} X^T y$. There’s no algebraic way to isolate $w$, so we rely on iterative numerical methods like gradient descent or Newton’s method to find the optimal parameters.”

## C. Optimization
#### 1.	How do you derive the gradient update rule?
We start from the log-likelihood of logistic regression, take the derivative with respect to the weights, and use the fact that the derivative of the sigmoid is $p(1-p)$. The gradient of the negative log-likelihood simplifies to $X^T(p - y)$. That gives the update rule:
$$w \leftarrow w - \eta X^T(p - y)$$
and similarly for the bias. It’s elegant because the gradient looks exactly like the difference between predicted probabilities and true labels.
#### 2.	Why use gradient descent instead of solving directly?
There’s no closed-form solution for logistic regression because the sigmoid makes the equations nonlinear. You can’t isolate w algebraically like in linear regression. Gradient descent gives us an iterative, scalable way to optimize the convex loss function, especially for large datasets. It avoids matrix inversion, works with streaming data, and can be distributed across GPUs and clusters.

#### 3.	What happens if the classes are perfectly separable?
If the classes are perfectly separable, logistic regression will try to push the weights toward infinity because it wants to give probabilities of exactly 0 or 1. The model doesn’t converge—its likelihood keeps improving as the weights get larger. In practice, the optimization becomes unstable, and you need regularization like L2 to prevent the weights from blowing up.

#### 4.	Why might gradient descent diverge in logistic regression?
The most common reason is a learning rate that's too high. With a large step size, updates overshoot the minimum, causing the loss to oscillate or explode. Other causes include unscaled features, extreme outliers, and numerical issues when predictions saturate near 0 or 1. Poorly conditioned data or multicollinearity can also cause unstable updates.

#### 5.	What optimization algorithms are typically used in practice? (SGD, LBFGS, Newton’s method)
Logistic regression is usually optimized with variants of gradient descent like batch GD, mini-batch SGD, or momentum-based methods. For smaller datasets, solvers like Newton’s method or quasi-Newton methods such as LBFGS work very well and converge quickly. Most libraries let you choose between LBFGS, SAG, SAGA, or SGD depending on the dataset size and sparsity.


## D. Regularization & Practical Issues
#### 1.	What is L2 regularization in logistic regression? Why use it?
#### 2.	How does L1 regularization affect logistic regression?
#### 3.	Explain the effect of regularization on decision boundaries.
#### 4.	How do you handle class imbalance in logistic regression?
#### 5.	Why do we need feature scaling?
#### 6.	What are common ways logistic regression can fail in real systems?


##  E. Code / Implementation
#### 1.	Implement logistic regression from scratch.
#### 2.	Explain each component of your implementation.
#### 3.	Why do we need to vectorize operations?
We vectorize the make the code fast and numerically efficient. Instead of looping over each sample in python, we let numpy execute optimized C or BLAS code on entire matrices at once. That drastically speeds up training, reduces bugs from indexing and matches the math directly; things like X.T @ error are exactly the gradient formulas. In practice, vectorization is essential for scaling to large dataset.


## 🔥 Advanced Logistic Regression Questions
#### 1.	Explain the relationship between logistic regression and Naive Bayes.
Naive Bayes is a generative model that learns class-conditional feature distributions, while logistic regression is a discriminative model that learns the decision boundary directly. If you assume Naive Bayes with Gaussian features and equal variances, you can actually derive a logistic regression form for the posterior probabilities. So under certain assumptions, Naive Bayes leads to a linear classifier similar to logistic regression, but they learn in very different ways.

#### 2.	How is logistic regression related to MLE under the Bernoulli likelihood?
Logistic regression assumes that each label is Bermoulli distributed, with probability given by a sigmoid of a linear function. If you write the likelihood for that Bernoulli distribution and maximize it, you get the cross-entropy loss. So training logistic regression is just doing maximum likelihood estimation for a Bernoulli model with a logit link.

#### 3.	Why does logistic regression produce a linear decision boundary?
The decision boundary is defined where the probability equals 0.5, which happens when $w^Tx+b=0$. That's a linear equation, so the boundary is a straight line in 2D, a plane in 3D, or a hyperplane in higher dimensions. The sigmoid only squashes the values; it doesn't change the boundary.

#### 4.	What is the ROC curve and how do we use it with logistic regression?
An ROC curve plots the true positive rate against the false positive rate at different probability thresholds. Logistic regression gives us probabilities, so by sweeping the threshold from 0 to 1, we can see how sensitivity and specificity trade off. The AUC score summarizes this into a single number that measures how well the model ranks positive examples above negative ones.

#### 5.	What does the coefficient represent in logistic regression?
Each coefficient represents how much the log-odds change when that feature increases by one unit, holding others constant. If a weight is positive, the probability of class 1 increases; if it’s negative, it decreases. Exponentiating the coefficient gives you an odds ratio, which is often used in statistics.

#### 6.	What is calibration? Why does logistic regression produce well-calibrated probabilities?
Calibration means that predicted probabilities match real-world frequencies. For example, when the model says 0.8, it should be right about 80% of the time. Logistic regression is well-calibrated because it directly optimizes log-likelihood under a Bermoulli model. It your assumptions hold, the output probabilities are naturally algined with true frequencies.

#### 7.	How does logistic regression generalize to multinomial outcomes (Softmax Regression)?
For more than two classes, logistic regression generalizes to Softmax regression. Instead of one sigmoid, you use a softmax function that outputs a probability for each class. Each class has its own weight vector, and training still uses cross-entropy loss. Conceptually, it’s just logistic regression extended from binary outcomes to multiple categories.