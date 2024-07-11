# deeplearning-go

### Introduction

This project implements a deep learning framework in Go, designed to be modular and extensible. 
The framework supports a variety of layers, activation functions, optimisers, and regularisers. 
It provides the ability to build, train, save, and load neural networks.

### Components

* Neural Network
* Tensor
* Layers
  * Fully Connected
  * Convolutional (Conv2D)
  * Embedding
  * Long-Short-Term-Memory (LSTM)
  * Gated Recurrent Unit (GRU)
  * Pooling
    * Average
    * Max
  * Dropout
  * Transformers
    * Flatten
    * Reshape
* Activation Functions
  * ReLU
  * LeakyReLU
  * Sigmoid
  * Tanh
  * Softmax
* Loss Functions
  * Mean Squared Error (MSE)
  * Cross Entropy
    * Binary
    * Categorical
  * Cosine Proximity
* Optimisers
  * Stochastic Gradient Descent (SGD)
  * SGD with Momentum
  * Adaptive Moment Estimation (ADAM)
  * Root Mean Square Propagation (RMSProp)
* Regularisers
  * L1
  * L2
  * ElasticNet

#### Further Work

* GPU processing with metal on MacOS
* More architectures
* Better examples
* New layer types being researched
* Potentially implement a computation graph
* Benchmarks