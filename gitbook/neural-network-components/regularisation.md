---
description: Keep the noise down
---

# Regularisation

Regularisation is a crucial technique in machine learning and deep learning that helps prevent overfitting, ensuring that the model generalises well to new, unseen data. Regularisation techniques add constraints or penalties to the learning process, discouraging the model from becoming too complex and capturing noise in the training data. Regularisation is applied to the output of the loss function and is done after the back propagation step, apart from the Dropout technique, which can be added between layers to randomly disable neurons during training. Here are the most common techniques:\


### **L1 Regularisation (Lasso)**&#x20;

L1 Regularisation adds a penalty equal to the absolute value of the magnitude of the coefficients.&#x20;

**Pros:** Encourages sparsity, meaning it can lead to zero weights, effectively performing feature selection. \
**Cons:** This can lead to models that are too simple if too many weights are driven to zero.\


**Formula:**

$$
\text{L1 Loss} = \lambda \sum_{i} |w_i|
$$

Where:

* $$L  \text{ is the total loss (including regularization)}$$
* $$L_0   \text{ is the original loss function (e.g. Mean Squared Error)}$$
* $$\lambda    \text{ is the regularization parameter}$$
* $$w_i  \text{ are the model parameters (weights)}$$\


### **L2 Regularisation (Ridge)**

L2 Regularisation adds a penalty equal to the square of the magnitude of the coefficients. Prevents the weights from becoming too large.

**Pros:** Keeps all features but reduces their impact, stabilises the learning process.\
**Cons:** Does not perform feature selection, all weights are shrunk but not zeroed out.\


**Formula:**

$$
\text{L2 Loss} = \lambda \sum_{i} w_i^2
$$

Where:

* $$L  \text{ is the total loss (including regularization)}$$
* $$L_0   \text{ is the original loss function}$$
* $$\lambda    \text{ is the regularization parameter}$$
* $$w_i  \text{ are the model parameters (weights)}$$\


### **Elastic Net Regularisation**

\
Elastic Net Regularisation combines L1 and L2 penalties, balancing between the benefits of both.

**Pros:** Balances sparsity and weight regularisation.\
**Cons:** Requires tuning two hyper-parameters.

\
**Formula:**

$$
\text{Elastic Net Loss} = \alpha \left( \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2 \right)
$$

Where:

* $$L  \text{ is the total loss (including regularization)}$$
* $$L_0   \text{ is the original loss function}$$
* $$\lambda_1  \text{ is the L1 regularization parameter}$$
* $$\lambda_2  \text{ is the L2 regularization parameter}$$
* $$w_i  \text{ are the model parameters (weights)}$$\


### **Dropout**

Dropout randomly sets a fraction of input units to zero at each update during training, which helps prevent units from co-adapting too much.

**Pros:** Reduces overfitting, simple to implement.\
**Cons:** Adds noise during training, may require longer training times.

\
**Formula:**

$$
\text{During training:} \quad y = f(x; \mathbf{w} \odot \mathbf{m}) 
\\ \
\\ 
 \text{During testing:} \quad y = \frac{1}{p} f(x; \mathbf{w})
$$

Where:

* $$y$$ is the output of a layer before applying Dropout
* $$f(x; w)  \text{ is the neural network function with weights }  w$$
* $$\odot  \text{ denotes element-wise multiplication}$$
* $$m$$ is a binary mask vector with the same shape as $$y$$, where each element is 0 with probability $$1 - p$$ and 1 with probability $$p$$

\
Regularisation techniques are essential tools in the machine learning toolkit. They help in building robust models that generalise well to new data by adding penalties or constraints to the learning process. Understanding and applying the appropriate regularisation technique can significantly enhance model performance and prevent overfitting.
