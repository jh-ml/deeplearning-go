---
description: Deep learning's primary data structure
---

# Tensor

The Tensor is the key data structure flowing through the network. It is a generic type in which we store the data required for processing, such as vectors, matrices (vector of vectors) or higher dimensional structures. Examples of Tensor usage in the network:

* Input data - e.g. vector of pixels or a matrix as input to a layer
* Weights - The connections between nodes can be stored as a matrix,
* Bias - The nodes in a layer can each have a bias term, stored in vector
* Gradients - The partial derivatives of the loss function calculated during back propagation
* Output data - The resulting computation values stored in output nodes of a layer

\
The basic type is as follows:

```go
type Tensor struct {
    data  []float64
    shape []int
    id    uuid.UUID
}
```

We can define an interface for the Tensor

```go
type Interface interface {
    ID() uuid.UUID
    Data() []float64
    Shape() []int
    Clone() Interface
    SetData(data []float64)
    Transpose() Interface
    Add(other Interface) Interface
    Subtract(other Interface) Interface
    Multiply(other Interface) Interface
    Divide(other Interface) Interface
    Dot(other Interface) Interface
    AddScalar(scale float64) Interface
    MultiplyScalar(scalar float64) Interface
    SumAlongBatch() Interface
    Threshold(probability float64) Interface
    Row(rowIndex int) ([]float64, error)
    AddRow(rowIndex int, row []float64) error
    Split([]int) []Interface
    Concatenate([]Interface) Interface
    Size() int
}
```

Different layers and algorithms use combinations of the Tensor functions when calculating within the network.
