package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type FullyConnected struct {
	weights, biases   tensor.Interface
	hidden            tensor.Interface
	dWeights, dBiases tensor.Interface
	activationFunc    activation.Interface
}

// NewFullyConnected creates a new fully connected layer with an activation function
func NewFullyConnected(inputSize, outputSize int, activationFunc activation.Interface) *FullyConnected {
	return &FullyConnected{
		weights:        tensor.NewXavierWeightsTensor(inputSize, outputSize),
		biases:         tensor.NewZerosTensor([]int{1, outputSize}),
		activationFunc: activationFunc,
	}
}

// Forward pass for FullyConnected layer
func (fc *FullyConnected) Forward(input tensor.Interface) tensor.Interface {
	fc.hidden = input
	output := input.Dot(fc.weights).Add(fc.biases)
	return fc.activationFunc.Forward(output)
}

// Backward pass for FullyConnected layer
func (fc *FullyConnected) Backward(grad tensor.Interface) tensor.Interface {
	grad = fc.activationFunc.Backward(grad)
	fc.dWeights = fc.hidden.Transpose().Dot(grad)
	fc.dBiases = grad.SumAlongBatch()
	return grad.Dot(fc.weights.Transpose())
}

// GetWeights returns the weights of the FullyConnected layer
func (fc *FullyConnected) GetWeights() tensor.Interface {
	return fc.weights
}

// SetWeights sets the weights of the FullyConnected layer
func (fc *FullyConnected) SetWeights(weights tensor.Interface) {
	fc.weights = weights
}

// GetBiases returns the biases of the FullyConnected layer
func (fc *FullyConnected) GetBiases() tensor.Interface {
	return fc.biases
}

// SetBiases sets the biases of the FullyConnected layer
func (fc *FullyConnected) SetBiases(biases tensor.Interface) {
	fc.biases = biases
}

// GetGradients returns the gradients of the FullyConnected layer
func (fc *FullyConnected) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return fc.dWeights, fc.dBiases
}

// RequiresOptimisation indicates if this layer requires optimisation
func (fc *FullyConnected) RequiresOptimisation() bool {
	return true
}

// RequiresRegularisation indicates if this layer requires regularisation
func (fc *FullyConnected) RequiresRegularisation() bool {
	return true
}

func (fc *FullyConnected) Name() string {
	return "FullyConnected"
}

func (fc *FullyConnected) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"input_dim":  fc.weights.Shape()[0],
		"output_dim": fc.weights.Shape()[1],
		"activation": fc.activationFunc.Name(),
	}

	tensors := []model.TensorData{
		{
			Name:  "Weights",
			Shape: fc.weights.Shape(),
			Data:  fc.weights.Data(),
		},
		{
			Name:  "Biases",
			Shape: fc.biases.Shape(),
			Data:  fc.biases.Data(),
		},
	}

	return config, tensors
}

func (fc *FullyConnected) Load(config map[string]any, tensors []model.TensorData) error {
	for _, tensorData := range tensors {
		switch tensorData.Name {
		case "Weights":
			fc.weights = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Biases":
			fc.biases = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		default:
			return errors.New("unexpected tensor name: " + tensorData.Name)
		}
	}

	activationName, ok := config["activation"].(string)
	if !ok {
		return errors.New("invalid activation")
	}
	activation, err := activation.NewActivationByName(activationName)
	if err != nil {
		return err
	}
	fc.activationFunc = activation

	return nil
}
