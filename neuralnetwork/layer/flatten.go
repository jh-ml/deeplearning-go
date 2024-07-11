package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Flatten struct {
	inputShape  []int
	outputShape []int
}

func NewFlatten(inputShape []int) *Flatten {
	batchSize := inputShape[0]
	flatSize := inputShape[1] * inputShape[2] * inputShape[3]
	return &Flatten{
		inputShape:  inputShape,
		outputShape: []int{batchSize, flatSize},
	}
}

func (flatten *Flatten) Forward(input tensor.Interface) tensor.Interface {
	return tensor.NewTensor(input.Data(), flatten.outputShape)
}

func (flatten *Flatten) Backward(grad tensor.Interface) tensor.Interface {
	return tensor.NewTensor(grad.Data(), flatten.inputShape)
}

// GetWeights returns the weights of the Flatten layer
func (flatten *Flatten) GetWeights() tensor.Interface {
	return nil
}

// SetWeights sets the weights of the Flatten layer
func (flatten *Flatten) SetWeights(weights tensor.Interface) {}

// GetBiases returns the biases of the Flatten layer
func (flatten *Flatten) GetBiases() tensor.Interface {
	return nil
}

// SetBiases sets the biases of the Flatten layer
func (flatten *Flatten) SetBiases(biases tensor.Interface) {}

// GetGradients returns the gradients of the Flatten layer
func (flatten *Flatten) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return nil, nil
}

// RequiresOptimisation indicates if this layer requires optimisation
func (flatten *Flatten) RequiresOptimisation() bool {
	return false
}

// RequiresRegularisation indicates if this layer requires regularisation
func (flatten *Flatten) RequiresRegularisation() bool {
	return false
}

func (flatten *Flatten) Name() string {
	return "Flatten"
}

func (flatten *Flatten) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"input_shape":  flatten.inputShape,
		"output_shape": flatten.outputShape,
	}
	return config, nil
}

func (flatten *Flatten) Load(config map[string]any, tensors []model.TensorData) error {
	inputShape, ok := config["input_shape"].([]any)
	if !ok {
		return errors.New("invalid input_shape")
	}
	flatten.inputShape = make([]int, len(inputShape))
	for i, v := range inputShape {
		flatten.inputShape[i] = int(v.(float64))
	}

	outputShape, ok := config["output_shape"].([]any)
	if !ok {
		return errors.New("invalid output_shape")
	}
	flatten.outputShape = make([]int, len(outputShape))
	for i, v := range outputShape {
		flatten.outputShape[i] = int(v.(float64))
	}

	return nil
}
