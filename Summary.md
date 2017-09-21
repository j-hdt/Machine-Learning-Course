# Coursera Machine Learning

## Week 1

### Hypothesis
$$
h_\theta =
\theta_0 + \ \theta_1 * x^{(1)} + \ \theta_2 * x^{(2)} + \ ...
$$

---
### Linear Regression
$$
\min_{\theta_i} \ \frac{1}{2m}
\sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2}
$$

---
### Cost function
$$
J(\theta_0, \theta_1) =
\frac{1}{2m} \sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2}
$$
Equals a squared error function

---
### Gradient Descent
good, when $n$ (no. of features) is large

repeat until convergence: {
$$
\theta_j =
\theta_j - \alpha \frac{1}{m}
\sum ^{m} _{i=1} {(h_\theta (x^{(i)}) - y^{(i)})^2} * x_j^{(i)}
$$
} at the same time for all $j$

---
### Feature Scaling
$$
x_i := \frac{x_i - \mu_i}{s_i}
$$
Where $s_i$ is the standard deviation (std) or the range of values
