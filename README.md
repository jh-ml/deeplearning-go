## deeplearning-go

The framework has been designed with a modular structure in mind, allowing for the creation of dynamic neural network architectures and experimentation. It's a work in progress while venturing deeper into the field. Please submit a pull request if you'd like to contribute. 🙂

[There is an accompanying guide written on gitbook](https://deeplearning-go.gitbook.io/deeplearning-go/), primarily for software engineers who would like an overview of this fascinating field of computation. It covers the core components of neural networks and how they function.

### Core Components

* Neural Network
* Tensor
* Layer Interface with implementations for
  * Fully Connected (Dense) Layer
  * Long-Short-Term-Memory (LSTM) Layer
  * Gated Recurrent Unit (GRU) Layer
  * Convolutional (Conv2D) Layer
  * Embedding Layer
  * Dropout Regularization Layer (Interlayer Dropout)
  * Average and Maximum Pooling Layers
  * Flatten and Reshape Layers
* Activation Functions
  * ReLU and Leaky ReLU
  * Sigmoid
  * Tanh
  * Softmax
* Loss Functions
  * Cross Entropy (Binary and Categorical)
  * Mean Squared Error (MSE)
  * Cosine Proximity
* Regularisation
  * Lasso (L1)
  * Ridge (L2)
  * Elastic Net
* Optimisation Algorithms
  * Stochastic Gradient Descent (SGD)
  * SGD with momentum
  * Root Mean Square Propagation (RMSProp)
  * Adaptive momentum (ADAM)

### Examples

The examples are currently showing how to set up a neural network with layers etc.
The ThreeLayer example is running from main() at the moment. The MNIST dataset has
been stored in the /data directory for testing purposes.

### Future Work

* GPU processing using Metal on MacOS, using C via Go.
* Potentially implement a computation graph. (Not sure if I would like to remove
the back propagation mechanism)
* Better Examples
* New Architectures
* New custom Layer types
* The model saving/loading system needs to be looked at further, to enable pipelines.
* Implement inference mechanisms with API deployment, model loading etc.

### Contributing

Please submit a pull request if you'd like to contribute. 🙂