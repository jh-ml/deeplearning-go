package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

// Embedding layer
type Embedding struct {
	weights   tensor.Interface
	gradients tensor.Interface
}

// NewEmbedding creates a new embedding layer
func NewEmbedding(vocabSize, embedSize int) *Embedding {
	weights := tensor.NewRandomTensor([]int{vocabSize, embedSize})
	gradients := tensor.NewZerosTensor([]int{vocabSize, embedSize})
	return &Embedding{weights: weights, gradients: gradients}
}

// Forward pass for Embedding
func (e *Embedding) Forward(input tensor.Interface) tensor.Interface {
	indices := input.Data()
	embedSize := e.weights.Shape()[1]
	embedded := tensor.NewZerosTensor([]int{len(indices), embedSize})

	for i, idx := range indices {
		if idx < 0 || int(idx) >= e.weights.Shape()[0] {
			panic("Index out of bounds")
		}
		row, _ := e.weights.Row(int(idx))
		copy(embedded.Data()[i*embedSize:(i+1)*embedSize], row)
	}
	return embedded
}

// Backward pass for Embedding
func (e *Embedding) Backward(grad tensor.Interface) tensor.Interface {
	e.gradients = tensor.NewZerosTensor(e.weights.Shape())
	gradData := grad.Data()
	embedSize := e.weights.Shape()[1]

	for i := 0; i < grad.Shape()[0]; i++ {
		rowGrad := gradData[i*embedSize : (i+1)*embedSize]
		idx := gradData[i*embedSize] // Assuming the index is stored in the first element
		if idx < 0 || int(idx) >= e.weights.Shape()[0] {
			panic("Index out of bounds")
		}
		for j := 0; j < embedSize; j++ {
			e.gradients.Data()[int(idx)*embedSize+j] += rowGrad[j]
		}
	}
	return tensor.NewZerosTensor([]int{}) // Embedding layer does not propagate gradients back to input
}

// GetWeights returns the weights of the embedding layer
func (e *Embedding) GetWeights() tensor.Interface {
	return e.weights
}

// SetWeights sets the weights of the embedding layer
func (e *Embedding) SetWeights(weights tensor.Interface) {
	e.weights = weights
}

// GetBiases returns nil as Embedding layer does not have biases
func (e *Embedding) GetBiases() tensor.Interface {
	return nil
}

// SetBiases does nothing as Embedding layer does not have biases
func (e *Embedding) SetBiases(biases tensor.Interface) {}

// GetGradients returns the gradients of the embedding layer
func (e *Embedding) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return e.gradients, nil
}

// RequiresOptimisation indicates if this layer requires optimisation
func (d *Embedding) RequiresOptimisation() bool {
	return true
}

// RequiresRegularisation indicates if this layer requires regularisation
func (d *Embedding) RequiresRegularisation() bool {
	return true
}

func (e *Embedding) Name() string {
	return "Embedded"
}

func (e *Embedding) Save() (map[string]any, []model.TensorData) {

	tensors := []model.TensorData{
		{
			Name:  "Weights",
			Shape: e.weights.Shape(),
			Data:  e.weights.Data(),
		},
	}

	return nil, tensors
}

func (e *Embedding) Load(config map[string]any, tensors []model.TensorData) error {
	for _, tensorData := range tensors {
		switch tensorData.Name {
		case "Weights":
			e.weights = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		default:
			return errors.New("unexpected tensor name: " + tensorData.Name)
		}
	}

	return nil
}
