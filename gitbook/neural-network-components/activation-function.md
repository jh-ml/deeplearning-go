---
description: Introducing non-linearity into the network
---

# Activation Function

Activation functions are critical to neural networks and they determine whether a neuron should be activated or not, based on the input it receives. They introduce non-linear properties to the network, allowing it to learn complex mappings from inputs to outputs. The function is applied to all nodes in a layer and different functions work better for different layers or tasks. E.g.

* Hidden Layers
  * ReLU and its variants (Leaky ReLU, Parametric ReLU) are often used in hidden layers due to their efficiency and performance in practice. They also help to mitigate the vanishing gradient problem, ideal for hidden layers.
* Output Layer
  * Binary Classification: Sigmoid is typically used.
  * Multi-class Classification: Softmax is commonly used.\


Activation functions are applied during the forward pass $$f(x)$$and their derivative is applied during back propagation $$f{\prime}(x)$$. Here are the most common activation functions with their formulas:\


### Rectified Linear Unit (ReLU)

ReLU is widely used in neural networks because of its simplicity and efficiency. It allows for faster training and mitigates the vanishing gradient problem by keeping gradients from becoming too small.\
\
**Formula:**

$$
f(x) = \max(0, x) 
\\\
\\\
 f{\prime}(x) = \begin{cases}
0 & \text{if } x \leq 0 \\
1 & \text{if } x > 0
\end{cases}
$$



### Sigmoid

The sigmoid function maps input values to the range (0, 1), making it useful for binary classification problems. However, it can suffer from the vanishing gradient problem, making training deep networks slow. Sigmoid is commonly used in binary classification tasks and as an activation function in the output layer for logistic regression models.\
\
**Formula:**

$$
f(x) = \frac{1}{1 + e^{-x}} 
\\ \
\\
 f{\prime}(x) = f(x) (1 - f(x))
$$



### Tanh (Hyperbolic Tangent)

The tanh function maps input values to the range (-1, 1). It is similar to the sigmoid function but centred around zero, making it more suitable for inputs that can be negative. These are often used in hidden layers, particularly in recurrent neural networks (RNNs) and long short-term memory networks (LSTMs).\
\
**Formula:**

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} 
\\ \
\\
 f{\prime}(x) = 1 - (\tanh(x))^2
$$



### Softmax

The softmax function is used to handle multi-class classification problems. It converts raw scores (logits) into probabilities, with each output value in the range (0, 1) and all outputs summing to 1. Softmax is typically used in the output layer of neural networks for multi-class classification problems.\
\
**Formula:**\


$$
f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} \text{ for } i = 1, \ldots, N 
\\ \
\\ \
\\
 \frac{\partial f(x_i)}{\partial x_j} = f(x_i) (\delta_{ij} - f(x_j))
$$

Where:

* $$x_i  \text{ is the input value for the i-th class}$$
* $$f(x_i)   \text{ is the activation output for the i-th class}$$
* $$N \text{ is the total number of classes}$$
* $$\delta_{ij} \text{ is the }Kronecker delta \text{ (1 if } i = j, 0 \text{ otherwise})$$\


### Summary

* **ReLU:** Fast and simple, helps mitigate vanishing gradient problem, ideal for hidden layers.
* **Sigmoid:** Maps to (0, 1), useful for binary classification, can suffer from vanishing gradients.
* **Tanh:** Maps to (-1, 1), centred around zero, useful for hidden layers in RNNs and LSTMs.
* **Softmax:** Converts logits to probabilities for multi-class classification, used in output layers.
