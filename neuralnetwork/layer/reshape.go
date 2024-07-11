package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

// Reshape represents a reshape layer
type Reshape struct {
	inputShape  []int
	outputShape []int
}

// NewReshape creates a new reshape layer
func NewReshape(inputShape, outputShape []int) *Reshape {
	return &Reshape{
		inputShape:  inputShape,
		outputShape: outputShape,
	}
}

// Forward pass for Reshape layer
func (r *Reshape) Forward(input tensor.Interface) tensor.Interface {
	r.inputShape = input.Shape()
	return input.Reshape(r.outputShape)
}

// Backward pass for Reshape layer
func (r *Reshape) Backward(grad tensor.Interface) tensor.Interface {
	return grad.Reshape(r.inputShape)
}

// GetWeights returns nil for Reshape layer as it has no weights
func (r *Reshape) GetWeights() tensor.Interface {
	return nil
}

// SetWeights does nothing for Reshape layer as it has no weights
func (r *Reshape) SetWeights(weights tensor.Interface) {}

// GetBiases returns nil for Reshape layer as it has no biases
func (r *Reshape) GetBiases() tensor.Interface {
	return nil
}

// SetBiases does nothing for Reshape layer as it has no biases
func (r *Reshape) SetBiases(biases tensor.Interface) {}

// GetGradients returns nil for Reshape layer as it has no gradients
func (r *Reshape) GetGradients() (tensor.Interface, tensor.Interface) {
	return nil, nil
}

// RequiresOptimisation indicates if this layer requires optimisation
func (r *Reshape) RequiresOptimisation() bool {
	return false
}

// RequiresRegularisation indicates if this layer requires regularisation
func (r *Reshape) RequiresRegularisation() bool {
	return false
}

func (r *Reshape) Name() string {
	return "Reshape"
}

func (r *Reshape) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"input_shape":  r.inputShape,
		"output_shape": r.outputShape,
	}
	return config, nil
}

func (r *Reshape) Load(config map[string]any, tensors []model.TensorData) error {
	inputShape, ok := config["input_shape"].([]any)
	if !ok {
		return errors.New("invalid input_shape")
	}
	r.inputShape = make([]int, len(inputShape))
	for i, v := range inputShape {
		r.inputShape[i] = int(v.(float64))
	}

	outputShape, ok := config["output_shape"].([]any)
	if !ok {
		return errors.New("invalid output_shape")
	}
	r.outputShape = make([]int, len(outputShape))
	for i, v := range outputShape {
		r.outputShape[i] = int(v.(float64))
	}

	return nil
}
