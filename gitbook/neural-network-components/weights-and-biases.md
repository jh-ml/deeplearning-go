---
description: Connections within the network
---

# Weights and Biases

Weights and biases are crucial components of a neural network’s architecture, determining how inputs are transformed through the network to produce outputs.

### Weights

Weights are adjustable parameters that scale the input data and control the strength of the connection between neurons in different layers. They are initialised randomly and updated during training to minimise the error in predictions.

### Biases

Biases are additional parameters that allow the model to adjust the output independently of the input data. They help the network model data that do not pass through the origin by providing an extra degree of freedom.

### Role in Neural Networks

Weights and biases are integral in determining the output of a neuron given an input. For a given neuron, the output is typically computed as $$\text{Activation}(W \cdot X + b)$$ , where $$W$$represents the weights,$$X$$ the input, and $$b$$ the bias.

### Training Process

During training, weights and biases are updated through back propagation, a process that uses gradients to adjust these parameters to minimise the error of the network’s predictions for the next iteration.
