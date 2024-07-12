---
description: Different sublayers of a neural network
---

# Layer Types

Computation can be organised into layer types, which work as smaller processing units within the larger neural network architecture. These layer types, also have sublayers, such as input, hidden, output etc. and certain layers have memory, capturing data and combining it with future inputs.&#x20;

There are many different layer types, used for different tasks and architectures, such as:

* **Fully Connected/Dense layer:** This is a layer where all the nodes in one layer are connected to all the nodes of the next layer. We can build the 3-2-1 network using two fully connected layers \[3, 2] and \[2, 1].
* **RNN (Recurrent Neural Network):** Processes sequential data by maintaining a hidden state that captures information from previous time steps.
* **LSTM (Long Short-Term Memory):** A type of RNN that uses special gating mechanisms to capture long-term dependencies and avoid the vanishing gradient problem.
* **GRU (Gated Recurrent Unit):** A type of RNN similar to LSTM but with a simpler gating mechanism, reducing the number of parameters and computational complexity.
* **CNN (Convolutional Neural Network):** Specialised for processing grid-like data such as images, using convolutional layers to detect local patterns and pooling layers to reduce dimensionality.
* **Embedding:** Transforms high-dimensional categorical data into dense, low-dimensional vectors that capture semantic relationships.

\
Specific layer designs and mathematics will be covered in a future version or published to a software engineering blog.
