package layer

import (
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Interface interface {
	Forward(input tensor.Interface) tensor.Interface
	Backward(grad tensor.Interface) tensor.Interface
	GetWeights() tensor.Interface
	SetWeights(weights tensor.Interface)
	GetBiases() tensor.Interface
	SetBiases(biases tensor.Interface)
	GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface)
	RequiresOptimisation() bool
	RequiresRegularisation() bool
	Name() string
	Save() (map[string]any, []model.TensorData)
	Load(map[string]any, []model.TensorData) error
}
