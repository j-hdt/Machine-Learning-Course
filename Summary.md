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

<div class="pagebreak"></div>
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
\begin{aligned}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \\ x_1 \\ \vdots \\ x_n\end{bmatrix}= \theta^T x\end{aligned}
$$

### Gradient Descent for multiple variables
repeat until convergence: {
$$
\begin{aligned}\theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; &&  \text{for j := 0...n}\end{aligned}
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

<div class="pagebreak"></div>
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
\begin{aligned}& h_\theta (x) = g ( \theta^T x ) \\& z = \theta^T x \\& g(z) = \dfrac{1}{1 + e^{-z}}\end{aligned}
$$

$h_\theta (x)$ will give the __probability__ that our output is 1.  
$h_\theta (x) = P(y=1|x;\theta)=1-P(y=0|x;\theta)$

### Decision Boundary
In order to get the discrete 0 or 1 classification, the output can be translated by rounding.  
$\begin{aligned}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \\& h_\theta(x) < 0.5 \rightarrow y = 0 \\\end{aligned}$

This way it can be said that  
$\begin{aligned}& \theta^T x \geq 0 \Rightarrow y = 1 \\& \theta^T x < 0 \Rightarrow y = 0 \\\end{aligned}$

The __decision boundary__ is the line that separates the area where $y=0$ and where $y=1$. It is created by the hypothesis function.

### Cost function
Cost function for logistic regression needs to be adopted because the Logistic Function will cause the output to be wavy, causing many local optima (not a convex function).

$$
\begin{aligned}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\ & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \\ & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{aligned}
$$

### Simplified Cost Function and Gradient Descent
#### Cost Function
Entire cost function:  
$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

Vectorized implementation:  
$$\begin{aligned} & h = g(X\theta)\\ & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{aligned}
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
\begin{aligned}& y \in \lbrace0, 1 ... n\rbrace \\& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \\& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \\& \cdots \\& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \\& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\\\end{aligned}
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
\min_{\theta}\ \dfrac{1}{2m}\  \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2
$$

### Regularized Linear Regression

Regularization can be applied to both linear regression and logistics function.

#### Gradient Descent
Repeat {
$$
\begin{aligned} & \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\ & \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] & j \in \lbrace 1,2...n\rbrace\end{aligned}
$$
}

The term $\frac{\lambda}{m}\theta_j$ performs the regularization. It can be also represented in the following form:
$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

#### Normal Equation
To add in regularization, the equation is the same as our original, except that we add another term inside parentheses:
$$
\begin{aligned}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \\& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \\ & 1 & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & 1 \\ \end{bmatrix} \end{aligned}
$$

### Regularized Logistics Regression
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$
The second sum __means to explicitly exclude__ the bias term $(\theta_0)$.  
Thus, for computing $\theta_0$ should be calculated in the same "repeat" statement but separate without the bias term.

<div class="pagebreak"></div>
## Week 4
- Neural Networks: Representation

### Model Representation
Neural network consisting of

- Input layer
- Hidden layers
- Output layer

where the current layer is denoted by $j$.  
A model with 3 layers (input, hidden and output) would look like:
$$
\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \\ \end{bmatrix}\rightarrow h_\theta(x)
$$
where
$$
\begin{aligned}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \\& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{aligned}
$$

The values for each of the "activation" nodes is obtained as follows:
$$
\begin{aligned} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\ a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{aligned}
$$
Each layer gets its own matrix of weights $\Theta^{(j)}$.  
The dimensions of the weights is determined as follows:  
$\text{If network has } s_j \text{ units in layer } j \text{ and } s_{j+1} \text{ units in layer } j+1, \text{ then } \Theta^{j} \text{ will be of dimension } s_{j+1} \times (s_j + 1)$.
The +1 comes from the addition $\Theta^{(j)}$ of the "bias nodes" $x_0$ and $\Theta^{(j)}_0$.

To vectorize the functions, a new variable $z^{(j)}_k$ is defined:
$$
\begin{aligned}a_1^{(2)} = g(z_1^{(2)}) \\ a_2^{(2)} = g(z_2^{(2)}) \\ a_3^{(2)} = g(z_3^{(2)}) \\ \end{aligned}
$$
For layer $j=2$ and node $k$, the variable $z$ will be:
$$
z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
$$
and the vector representation is:
$$
\begin{aligned}x = \begin{bmatrix}x_0 \\ x_1 \\\cdots \\ x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \\ z_2^{(j)} \\\cdots \\ z_n^{(j)}\end{bmatrix}\end{aligned}
$$
Setting $x=a^{(1)}$, the equation can be written as
$$
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
$$

<div class="pagebreak"></div>
## Week 5
- Neural Networks: Learning
    - Cost Function and Backpropagation

### Cost Function
$L = \text{ total no. of layers in network}$  
$s_l = \text{ number of units (not counting bias unit) in layer l}$  
$K = \text{ number of output units/classes}$

Neural Network cost function similar to Logistic regression but for $K$ classes
$$
\begin{gathered} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gathered}
$$

### Backpropagation Algorithm

$\delta_j^{l} =$ the "error" of node $j$ in layer $l$.

Backpropagation is neural-network terminology for minimizing the cost function (as done for gradient descent in logistic and linear regression). Goal:  
$\min_\Theta J(\Theta)$

That is, minimizing the cost function $J$ using an optimal set of parameters in theta.

To compute the partial derivative of $J(\Theta)$:
$$
\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
$$
The following algorithm can be applied for the __back propagation algorithm__:

Given training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$

- Set $\Delta^{(l)}_{i,j}$:= 0 for all (l, i, j), (hence end up having a matrix full of zeros)

For training example __t = 1 to m__:

1. Set $a^{(1)}:=x^{(t)}$
2. Perform forward propagation to compute $a^{(l)}$ for l=2,3,...,L
    1. $a^{(1)} = x$
    2. $z^{(2)} = \Theta^{(1)}a^{(1)}$
    3. $a^{(2)} = g(z^{(2)}) \ \ \ \ (\text{add } a_0^{(2)})$
    4. $z^{(3)} = \Theta^{(2)}a^{(2)}$
    5. $a^{(3)} = g(z^{(3)}) \ \ \ \ (\text{add } a_0^{(3)})$
    6. $z^{(4)} = \Theta^{(3)}a^{(3)}$
    7. $a^{(4)} = h_\Theta (x) = g(z^{(4)})$
3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

To get the delta values of the layers before the last layer (the last one is simply the difference between the output $a^{(L)})$ and the correct outputs in $y$), the following equation will step back from right to left:

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$, using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. Then element-wise multiply that with a function called $g'$, or __g-prime__, which is the derivative of the activation function $g$ evaluated with the input values given by $z^{(l)}$.
$$
g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)})
$$

5. $\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$ or with Vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$

Hence update the new $\Delta$ matrix.

- $D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)$
- $D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}$

The capital-delta matrix D is used as an "accumulator" to add up values and eventually compute the partial derivative, resulting in $\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)$.

### Implementation Notes
Assuming a 3 layer network with ten nodes in layer 1 and layer 2, and one node in the output layer.
```
thetaVec  = [Theta1(:); Theta2(:); Theta3(:)]
DVec      = [D1(:); D2(:); D3(:)]

Theta1 = reshape (thetaVec(1:110),10,11);
Theta2 = reshape (thetaVec(111:220),10,11);
Theta3 = reshape (thetaVec(221:231),1,11);
```
#### Gradient checking
Only use to check but disable during learning as it is a really slow algorithm.

Approximate the derivative of the cost function:
$$
\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
$$
With multiple theta matrices, it can be approximated __with respect to $\Theta_j$__ as follows:
$$
\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}
$$
using a small value for epsilon ($\epsilon = 10^{-4}$)
```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

#### Random Initialization
Initializing all $\Theta$ to zero will result in identical $a$ and $\delta$ as well as identical gradients.
Hence, initialize each $\Theta^{(l)}_{ij}$ to a random value between $[-\epsilon,\epsilon]$ ($\epsilon$ is not related to the error from before)
```
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

#### Training a neural network

- Number of input units = dimension of features $x^{(i)}$
- Number of output units = number of classes
- Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
- Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

1. Randomly initialize the weights
2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x(i)$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

After having forward and back propagation performed, loop on every training example:
```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

<div class="pagebreak"></div>
## Week 6

### Evaluating a Hypothesis

A hypothesis may have low error for the training examples but still be inaccurate (because of overfitting).

With a given dataset of training examples, we can split up the data into two sets: a __training set__ and a test set.

The new procedure using these two sets is then:

- Learn $\Theta$ and minimize $J_{train}(\Theta)$ using the training set
- Compute the test set error $J_{test}(\Theta)$

#### The test set error

1. For linear regression: $J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2$
2. For classification ~ Misclassification error (aka 0/1 misclassification error):    
$$
err(h_\Theta(x),y) =
\begin{matrix}
1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\\
0 & \mbox otherwise
\end{matrix}
$$

This gives us a binary 0 or 1 error result based on a misclassification.
The average test error for the test set is
$$
\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})
$$
This gives us the proportion of the test data that was misclassified.

### Model Selection and Train/Validation/Test Sets

1. Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the test set.
3. Estimate the generalization error also using the test set with $J_{test}(\Theta^{(d)})$, (d = theta from polynomial with lower error);

In this case, we have trained one variable, d, or the degree of the polynomial, using the test set. This will cause our error value to be greater for any other set of data.

#### Use of the CV set

To solve this, we can introduce a third set, the __Cross Validation Set__, to serve as an intermediate set that we can train d with. Then our test set will give us an accurate, non-optimistic error.

One example way to break down our dataset into the three sets is:

- Training set: 60%
- Cross validation set: 20%
- Test set: 20%

We can now calculate three separate error values for the three different sets.

#### With the Validation Set (note: this method presumes we do not also use the CV set for regularization)

1. Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $J_{test}(\Theta^{(d)})$, (d = theta from polynomial with lower error);

This way, the degree of the polynomial d has not been trained using the test set.

(Mentor note: be aware that using the CV set to select 'd' means that we cannot also use it for the validation curve process of setting the lambda value).

### Diagnosing Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

- We need to distinguish whether __bias__ or __variance__ is the problem contributing to bad predictions.
- High bias is underfitting and high variance is overfitting. We need to find a golden mean between these two.

The training error will tend to __decrease__ as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to __decrease__ as we increase d up to a point, and then it will __increase__ as d is increased, forming a convex curve.

__High bias (underfitting)__: both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ will be high. Also, $J_{CV}(\Theta) \approx J_{train}(\Theta)$.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yPz7yntYEeam4BLcQYZr8Q_07e80cbb290293d193f3cd92f1148e16_300px-Features-and-polynom-degree.jpg?expiry=1507075200000&hmac=Z6ogirsSfQIoizP2KkMUB-YsHdDAmL1boy_a23ihQV8)

### Regularization and Bias/Variance

Instead of looking at the degree d contributing to bias/variance, now we will look at the regularization parameter $\lambda$.

- Large $\lambda$: High bias (underfitting)
- Intermediate $\lambda$: just right
- Small $\lambda$: High variance (overfitting)
A large lambda heavily penalizes all the $\Theta$ parameters, which greatly simplifies the line of our resulting function, so causes underfitting.

The relationship of $\lambda$ to the training set and the variance set is as follows:

__Low $\lambda$__: $J_{train}(\Theta)$ is low and $J_{CV}(\Theta)$ is high (high variance/overfitting).

__Intermediate $\lambda$__: $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ are somewhat low and $J_{train}(\Theta) \approx J_{CV}(\Theta)$.

__Large $\lambda$__: both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ will be high (underfitting /high bias)

The figure below illustrates the relationship between lambda and the hypothesis:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8K3a0XtZEealOA67wFuqoQ_fb209643920cb669170dfe96a19e2b00_300px-Features-and-polynom-degree-fix.png?expiry=1507075200000&hmac=4CFsD2AIhIpA8VsK1iJVZomcZhcbFGQIOP0SWlv__gE)

In order to choose the model and the regularization $\lambda$, we need:

1. Create a list of lambdas (i.e. $\lambda\in${0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24});
2. Create a set of models with different degrees or any other variants.

3. Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$.

4. Compute the cross validation error using the learned $\Theta$ (computed with $\lambda$) on the $J_{CV}(\Theta)$ without regularization or $\lambda$ = 0.

5. Select the best combo that produces the lowest error on the cross validation set.

6. Using the best combo $\Theta$ and $\lambda$, apply it on $J_{test}(\Theta)$ to see if it has a good generalization of the problem.



<div class="pagebreak"></div>
## Week 7
