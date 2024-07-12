---
description: Minimise loss
---

# Optimisation

Optimisation algorithms are crucial in training neural networks as they determine how the model’s parameters (weights and biases) are updated based on the computed gradients. These updates aim to minimise the loss function, thus improving the model’s performance.

## Gradient Descent

Gradient descent is a foundational optimisation algorithm in machine learning, enabling models to learn from data by iteratively updating parameters to minimise error. It essentially performs a step wise search along the slope of the derivative's parabola. The derivative being the gradient with respect to the loss function. All gradients across the network are searched across iterations, until the loss has been minimised.

\
Here are a few of the most common optimisation algorithms used in deep learning:\


### Stochastic Gradient Descent (SGD)

SGD is an optimisation algorithm that updates the model’s parameters using only a single or a small batch of training examples at each iteration. This makes it more efficient and faster than traditional gradient descent, which uses the entire dataset to compute the gradients.\
\
\
**Key Points:**

* **Efficiency:** Suitable for large datasets since it updates parameters more frequently.
* **Noise:** Introduces noise into the updates, which can help escape local minima and explore the parameter space more effectively.
* **Learning Rate:** The size of each update step. Choosing an appropriate learning rate is crucial for convergence.
* **Use Case:** Large datasets, online learning.\


**Formula:**

$$
w = w - \eta \cdot \nabla L
$$

Where:

* $$w \text{ is the weight}$$
* $$\eta \text{ is the learning rate}$$
* $$\nabla L \text{ is the gradient of the loss function }L \text{ with respect to } w$$



### SGD with Momentum

SGD with Momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging. It accumulates a velocity vector in the direction of the gradients to smooth out the updates.

\
**Key Points:**

* **Momentum**: A fraction of the previous update is added to the current update, which helps dampen oscillations and smooth out the updates.
* **Velocity**: The velocity term accumulates the gradient of the loss function over time.
* **Use Case:** Cases with significant oscillations in SGD.

\
**Formula:**

$$
v = \gamma v + \eta \cdot \nabla L 
\\ \
\\
 w = w - v
$$

Where:

* $$v \text{ is the velocity}$$
* $$\gamma \text{ is the momentum factor (typically set to 0.9)}$$
* $$\eta \text{ is the learning rate}$$
* $$\nabla L \text{ is the gradient of the loss function }L \text{ with respect to } w$$\


### Root Mean Square Propagation (RMSProp)

RMSProp adapts the learning rate for each parameter by dividing the learning rate by an exponentially decaying average of squared gradients. This helps handle the problem of vanishing or exploding gradients.

\
**Key Points:**

* **Adaptive Learning Rate**: Adjusts the learning rate based on the recent magnitudes of the gradients.
* **Stabilisation**: Prevents oscillations and stabilises training by normalising the gradients.
* **Use Case:** Recurrent neural networks, deep networks.

\
**Formula:**

$$
E[g^2]t = \gamma E[g^2]{t-1} + (1 - \gamma) g_t^2
\\\
\\
 w = w - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

Where:

* $$E[g^2]_t  \text{ is the exponentially decaying average of past squared gradients}$$
* $$\gamma \text{ is the decay rate (typically set to 0.9)}$$
* $$\eta \text{ is the learning rate}$$
* $$\epsilon \text{ is a small constant to prevent division by zero}$$
* $$g_t \text{ is the gradient at time step } t$$\


### Adaptive Moment Estimation (ADAM)

Adam combines the advantages of two other extensions of SGD, namely AdaGrad and RMSProp. It maintains an exponentially decaying average of past gradients (momentum) and an exponentially decaying average of past squared gradients (RMSProp).\


**Key Points:**

* **Adaptive Learning Rate**: Maintains separate learning rates for each parameter.
* **Bias Correction:** Includes bias correction terms to account for the initialisation at zero.
* **Use Case:** General-purpose optimiser, works well in most cases.

\
**Formula:**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t 
\\ \
\\
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 
\\\
\\
 \hat{m}_t = \frac{m_t}{1 - \beta_1^t} 
\\\
\\
 \hat{v}_t = \frac{v_t}{1 - \beta_2^t} 
\\\
\\
 w = w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

Where:

* $$m_t \text{ is the first moment (mean) of the gradients}$$
* $$v_t  \text{ is the second moment (uncentered variance) of the gradients}$$
* $$\beta_1  and \ \beta_2  \text{ are the decay rates for the moment estimates}$$
* $$\hat{m}_t  \ and \ \hat{v}_t  \text{ are the bias-corrected moment estimates}$$
* $$\eta \text{ is the learning rate}$$
* $$\epsilon \text{ is a small constant to prevent division by zero}$$
* $$g_t \text{ is the gradient at time step } t$$\


### Summary

* **SGD:** Good for large datasets and online learning but can be noisy.
* **SGD with Momentum:** Adds a velocity term to smooth updates and accelerate convergence.
* **RMSProp:** Uses adaptive learning rates to stabilise training and handle vanishing/exploding gradients.
* **Adam:** Combines momentum and RMSProp for adaptive learning rates and faster convergence with bias correction.
