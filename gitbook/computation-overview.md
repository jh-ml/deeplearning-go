---
description: Core concepts of neural network computation
---

# Computation Overview

### Input \* Weight

There are a few mathematical concepts used in deep learning that, when combined, result in powerful computation systems. The basic calculation is multiplying an input with a weighted value, to get an output.

<figure><img src=".gitbook/assets/Computation - Basic.svg" alt="" width="321"><figcaption><p>Input * Weight</p></figcaption></figure>

### Weighted Sum/Dot Product

We can then apply this to a 3 to 1 design.

<figure><img src=".gitbook/assets/Computation - Weighted Sum.svg" alt="" width="375"><figcaption><p>Weighted Sum of 3 inputs to an output</p></figcaption></figure>

This introduces the Weighted Sum, also known as the Dot Product. It is the accumulation of all the multiplications between nodes. As we see here, the result value is 1.5. \


### Activation Functions

When working with numerical ranges we often use the ranges of -1.0 to 1.0 or 0 to 1.0, because it acts as a fixed range, allowing normalisation, scaling etc. Neural networks use the critical component called an activation function, which enables/disables nodes in a network based on their value, like an on/off switch. Let's apply a common activation function used in hidden layers, $$tanh$$, to control the nodes activation.

<figure><img src=".gitbook/assets/Computation - tanh.svg" alt="" width="563"><figcaption><p><span class="math">tanh</span>applied to the output node</p></figcaption></figure>

We will cover $$tanh$$ in detail later on, for now note how the result value was modified within range.\


### Three Layer Network

Going a step forward, we can design a three layer network, where we apply another activation function, known as $$Sigmoid$$, to the output node.

<figure><img src=".gitbook/assets/Computation - 3 layers.svg" alt="" width="563"><figcaption><p>Three layer network</p></figcaption></figure>

Let's work through the calculations\
\
**Variables:**

* Input nodes: $$[0.7, 0.5, 0.3]$$
* Weights from the input to hidden layers: $$[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]$$
* Weights from hidden to output layer: $$[0.3, 0.1]$$
* Hidden layer activation: $$\tanh$$
* Output layer activation: $$\sigma$$ (sigmoid)

\
**Calculations:**\


**Step 1: Input to Hidden Layer**

* Reshape weights for input to hidden layer:\
  \
  &#x20;$$\begin{bmatrix} 1.0 & 0.8 \\ 0.6 & 0.4 \\ 0.2 & 0.0 \end{bmatrix}$$
* Calculate the weighted sum for each hidden node:
  * **Hidden node 1:**&#x20;
    * $$z_1 = (0.7 \times 1.0) + (0.5 \times 0.6) + (0.3 \times 0.2) = 0.7 + 0.3 + 0.06 = 1.06$$
    * Apply $$\tanh$$ activation: $$h_1 = \tanh(1.06) \approx 0.786$$
  * **Hidden node 2:**&#x20;
    * $$z_2 = (0.7 \times 0.8) + (0.5 \times 0.4) + (0.3 \times 0.0) = 0.56 + 0.2 + 0 = 0.76$$
    * Apply $$\tanh$$ activation: $$h_2 = \tanh(0.76) \approx 0.641$$
  * **Hidden layer output:**
    * $$[0.786, 0.641]$$

**Step 2: Hidden to Output Layer**

* Weights for hidden to output layer: $$[0.3, 0.1]$$
* Calculate the weighted sum for the output node:
  * $$z_{\text{out}} = (0.786 \times 0.3) + (0.641 \times 0.1) = 0.2358 + 0.0641 = 0.2999$$
* Apply sigmoid activation:
  * $$\sigma(z_{\text{out}}) = \frac{1}{1 + e^{-0.2999}} \approx 0.5744$$

### Bias

The bias term is an additional parameter added to each node in a layer, except the input layer. It allows the model to better fit the data by providing each node with the ability to shift the activation function, adjusting its threshold independently. It adds flexibility to the model, allowing it to learn more complex relationships in the data and improves the model’s ability to generalise.\
\
**Formula:** For a neuron $$i$$ with input $$x$$, weights $$w_i$$, and bias $$b$$

$$
z_i = w_i \cdot x + b
$$

where $$z_i$$ is the input to the activation function.

We can store a vector of bias terms, one for each node in a layer. $$z_i = \sum_{j} w_{ij} x_j + b_i$$\


### Loss Function

Once we have sent the input tensor through the network, we compare the predicted result with the true sample value, using a loss function. The loss calculation results in both a value and a gradient. The gradient tells us how much we need to adjust the prediction and in what direction.\


<figure><img src=".gitbook/assets/Computation - Loss.svg" alt=""><figcaption><p>Forward pass loss calculation</p></figcaption></figure>

\
Once we know the loss gradient, we can propagate the error back through the network, known as backpropagation. We can use the chain rule, to determine how each weight, connecting two nodes, influenced the final loss value.\


<figure><img src=".gitbook/assets/Computation - Backpropagation.svg" alt=""><figcaption><p>Back propagated loss calculation</p></figcaption></figure>

\
These gradients are then passed to an optimiser algorithm and used to update the weights accordingly, iteratively resulting in minimising the loss of the network.\


### The Chain Rule

The chain rule is a fundamental principle in calculus that allows us to compute the derivative of a composite function. The chain rule allows us to decompose the derivative of a composite function into the product of simpler derivatives. In the context of neural networks, the chain rule is essential for backpropagation, which is used to calculate gradients and update the model’s parameters (weights and biases).\
\
**Formula:**

If we have two functions $$f$$ and $$g$$, and we form a composite function $$h(x) = f(g(x))$$ , the chain rule states that the derivative of $$h$$ with respect to $$x$$ is:

$$
\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

**Example Calculation:**\


**Forward Pass:**

* Input: x
* Hidden layer: h = g(x)
* Output layer: o = f(h)
* Loss: L(o, y)

\
**Backward Pass:**

We want to compute $$\frac{\partial L}{\partial x}$$

$$
\frac{\partial L}{\partial o} = \text{(depends on the loss function)}
\\
\\ \
\\
\frac{\partial o}{\partial h} = f{\prime}(h)
\\
\
\\\
\\
\frac{\partial h}{\partial x} = g{\prime}(x)
$$

Then we apply the chain rule

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial o} \cdot \frac{\partial o}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

\
We calculate all the gradients for the network and then the optimiser updates the weights and biases accordingly before the next training epoch starts.
