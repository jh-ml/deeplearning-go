---
description: Tracking error through the network
---

# Gradients

When we propagate the loss/error back through the network, we compute the gradients of the loss with respect to each parameter. Gradients are the partial derivatives of the loss function with respect to each parameter (weights and biases) in the network. They indicate the direction and magnitude of change required to minimise the loss.

Once the network's gradients are computed, an optimisation algorithm adjusts the weights and biases to minimise the loss function. The gradients determine the step size and direction for these adjustments. Note that the loss function calculates the loss and then using the derivative of the loss function, it calculates the gradient of the loss, which is used during backpropagation, to calculate the gradients as we move back through the layers of the network.\


### Example

We compute loss using a predicted value and a target value, i.e. how incorrect was the prediction. Let's look at an example using the Mean Squared Error calculation. (We will look more in depth at loss functions on the next page)\


**Mean Squared Error (MSE) Loss:**

The MSE loss for a single sample is defined as $$L = \frac{1}{2} (\hat{y} - y)^2$$, where $$L$$ is the loss, $$\hat{y}$$ is the predicted value, and $$y$$ is the actual value.

**Gradient of the Loss:**

To find the gradient of the loss with respect to the predicted value $$\hat{y}$$, we need to compute the partial derivative of $$L$$ with respect to $$\hat{y}$$:

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left( \frac{1}{2} (\hat{y} - y)^2 \right)
$$

Using the chain rule of differentiation:

$$
\frac{\partial L}{\partial \hat{y}} = \frac{1}{2} \cdot 2 (\hat{y} - y) \cdot \frac{\partial (\hat{y} - y)}{\partial \hat{y}}
$$

Simplifying this, we get:

$$
\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y)
$$

**Example Calculation**

\
Actual Value $$y$$: 1.0\
Predicted Value $$\hat{y}$$: 0.8

**Calculate the Loss:**

$$
L = \frac{1}{2} (0.8 - 1.0)^2 = \frac{1}{2} \times 0.04 = 0.02
$$

**Calculate the Gradient:**

$$
\frac{\partial L}{\partial \hat{y}} = 0.8 - 1.0 = -0.2
$$

The gradient -0.2 indicates that increasing the predicted value $$\hat{y}$$ will decrease the loss, as it suggests that $$\hat{y}$$ is less than the actual value $$y$$.

The optimisation algorithm then uses the gradients of each parameter to adjust the model parameters in the direction opposite to the gradient, because we want to move towards the minimum of the loss function.
