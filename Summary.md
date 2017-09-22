# Coursera Machine Learning

## Week 1
- Introduction
- Linear Regression with One Variable
    - Model and Cost function
    - Parameter Learning

### Supervised Learning
Input data set given and knowledge how the correct output should look like.  
Categorized into two kind of problems:

- Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
- Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign

### Unsupervised Learning
Allows to approach problems with little or no idea what the results should look like

- Clustering - Finding a way to automatically group objects based on different variables
- Non-clustering - Finding structure in a chaotic environment

### Hypothesis
$$
h_\theta =
\theta_0 + \ \theta_1 * x^{(1)} + \ \theta_2 * x^{(2)} + \ ...
$$


### Linear Regression
$$
\min_{\theta_i} \ \frac{1}{2m}
\sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2}
$$


### Cost function
$$
J(\theta_0, \theta_1) =
\frac{1}{2m} \sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2}
$$
Equals a squared error function


### Gradient Descent
good, when $n$ (no. of features) is large

repeat until convergence: {
$$
\theta_j :=
\theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)
$$
} at the same time for all $j$

repeat until convergence: {
$$
\theta_j :=
\theta_j - \alpha \frac{1}{m}
\sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2} \ x_j^{(i)}
$$
} at the same time for all $j$

## Week 2
- Linear Regression with Multiple Variables
    - Multivariate Linear Regression
    - Computing Paramters Analytically

### Multiple features
Regular expression
$$
h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n
$$
Vectorization
$$
\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}
$$

### Gradient Descent for multiple variables
repeat until convergence: {
$$
\begin{align*}\theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; &&  \text{for j := 0...n}\end{align*}
$$
}

To make sure gradient descent is working correctly, make a plot with _number of iterations_ on the x-axis.
Then plot the cost function $J(\theta)$ over the number of iterations of gradient descent.  
If $J(\theta)$ ever increases, then $\alpha$ should probably decreased.

### Feature Scaling
Used to get the values into the range $-1 \leq x_{(i)} \leq 1$.
This will speed up gradient descent.
$$
x_i := \frac{x_i - \mu_i}{s_i}
$$
Where $s_i$ is the standard deviation (std) or the range of values

### Normal Equation
- When $n$ goes high (greater ca. 100000)
- performing the minimization explicitly
- no iterative algorithm
- non invertible when redundant or too many features $m \leq n$
- no need to do feature scaling

Minimizes $J$ by explicitly taking its derivatives with respect to the $\theta_j$ 's, and setting them to zero.

$$
\theta = (X^T X)^{-1}X^T y
$$

### Comparison of Gradient Descent to Normal Equation

Gradient Descent            | Normal Equation
--                          |--
Needs to choose alpha       |  No need to choose alpha
Needs many iterations       |  No need to iterate
$O(kn^2)$                   | $O(n^3)$, need to calculate inverse of $X^T X$
Works well when n is large  | Slow if n is very large

## Week 3
- Logistic Regression
    - Classification and Representation
    - Logistic Regression Model
    - Multiclass Classification
- Regularization
    - Solving the Problem of Overfitting

### Hypothesis Representation (Sigmoid function, Logistic function)
When $y\in \{0,1\}$ the hypothesis $h_\theta (x)$ should also satisfy $0 \leq h_\theta (x) \leq 1$.  
Therefore, $\theta^T x$ can be plugged into the "Logistics Function":
$$
\begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}
$$

$h_\theta (x)$ will give the __probability__ that our output is 1.  
$h_\theta (x) = P(y=1|x;\theta)=1-P(y=0|x;\theta)$

### Decision Boundary
In order to get the discrete 0 or 1 classification, the output can be translated by rounding.  
$\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}$

This way it can be said that  
$\begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \newline& \theta^T x < 0 \Rightarrow y = 0 \newline\end{align*}$

The __decision boundary__ is the line that separates the area where $y=0$ and where $y=1$. It is created by the hypothesis function.

### Cost function
Cost function for logistic regression needs to be adopted because the Logistic Function will cause the output to be wavy, causing many local optima (not a convex function).

$$
\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}
$$

### Simplified Cost Function and Gradient Descent
#### Cost Function
Entire cost function:  
$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

Vectorized implementation:  
$$\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}
$$

#### Gradient Descent
Entire cost function:  
$$
\theta_j :=
\theta_j - \frac{\alpha}{m}
\sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})} \ x_j^{(i)}
$$

Vectorized implementation:  
$$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$$

### Multiclass Classification: One-vs-all
More than two categories, thus $y$ doesn't behave binary but instead is assumed to be $y=\{0,1...n\}$.  
The problem will be divided into $n+1$ (+1 because the index starts at 0) binary classification problems.  
This will compare one class against all the others. Then, the hypothesis that returned the highest value will be used as the prediction.

$$
\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}
$$

### Solving the Problem of Overfitting
Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features
    - Manually select which features to keep
    - Use a model selection algorithm
2) Regularization
    - Keep all the features, but reduce the magnitude of parameters $\theta_j$
    - Regularization works well when there are a lot of slightly useful features

### Regularization parameter
To prevent/reduce overfitting from the hypothesis function, the weight that some terms carry can be reduced by increasing their cost.  
The $\lambda$ is the __regularization parameter__.
It determines how much the cost of theta parameters are inflated.
$$
min_\theta\ \dfrac{1}{2m}\  \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2
$$

### Regularized Linear Regression

Regularization can be applied to both linear regression and logistics function.

#### Gradient Descent
Repeat {
$$
\begin{align*} & \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] & j \in \lbrace 1,2...n\rbrace\end{align*}
$$
}

The term $\frac{\lambda}{m}\theta_j$ performs the regularization. It can be also represented in the following form:
$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

#### Normal Equation
To add in regularization, the equation is the same as our original, except that we add another term inside parentheses:
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$

### Regularized Logistics Regression
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$
The second sum __means to explicitly exclude__ the bias term (\theta_0).  
Thus, for computing $\theta_0$ should be calculated in the same "repeat" statement but separate without the bias term.