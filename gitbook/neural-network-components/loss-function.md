---
description: Calculating the prediction error during training
---

# Loss Function

The Loss Function, also known as the cost function, measures how well the neural network’s predictions match the actual target values. It quantifies the difference between the predicted outputs and the true outputs. The primary goal during training is to minimise the loss function, thereby improving the accuracy of the model. The loss function calculates two primary values, the loss value and the gradient (derivative) of the loss.

Here’s are some of the commonly used loss functions in neural networks:\


### Mean Squared Error (MSE)

MSE measures the average of the squared differences between predicted and actual values. We square them to get absolute values for the sum, because summing negative values would decrease the total error sum, throwing off the average.

**Formula:**

$$
L = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

**Gradient:**

$$
\frac{\partial L}{\partial \hat{y}_i} = \hat{y}_i - y_i
$$



### Cross Entropy

Cross entropy is a loss function used primarily for classification tasks. It effectively guides the training process by penalising incorrect predictions more heavily, thus improving the model’s ability to distinguish between different classes.\


**Benefits:**

* **Probabilistic Interpretation:** It provides a clear probabilistic interpretation, as it directly measures how well the predicted probability distribution matches the true distribution.
* **Smooth and Differentiable:** Cross entropy is smooth and differentiable, making it suitable for gradient-based optimisation methods like gradient descent.



Cross Entropy can be applied to binary and catagorisation tasks.



### Binary Cross Entropy

Binary cross-entropy is suitable for binary classification tasks, where the goal is to distinguish between two possible classes.&#x20;

**Formula:**

$$
L = -\frac{1}{N} \sum_{i=1}^N \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
$$

**Gradient:**

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)}
$$



### Categorical Cross Entropy

Categorical cross-entropy is suitable for classifying a prediction over a number of classes, e.g. the MNIST image dataset.

**Formula:**

$$
L = -\sum_{i=1}^N \sum_{j=1}^C y_{ij} \log(\hat{y}_{ij})
$$

**Gradient:**

$$
\frac{\partial L}{\partial \hat{y}{ij}} = \hat{y}{ij} - y_{ij}
$$



### Cosine Proximity

The cosine proximity loss function measures the cosine similarity between predicted and actual values. It is commonly used for tasks where the angle between vectors is important, such as natural language processing.

**Formula:**

$$
L = -\frac{\sum_{i=1}^N \hat{y}i y_i}{\sqrt{\sum{i=1}^N \hat{y}i^2} \sqrt{\sum{i=1}^N y_i^2}}
$$

**Gradient:**

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{y_i}{\|y\|} - \frac{(\hat{y} \cdot y)}{\|\hat{y}\|^2 \|y\|} \hat{y}_i
$$

\
\
**Summary:**\
\
Loss functions are vital for training neural networks, providing a measure of error between predictions and actual values and guiding the optimisation process through gradients. Different loss functions are suited for different types of tasks, such as regression, binary classification, multi-class classification, and similarity tasks.

