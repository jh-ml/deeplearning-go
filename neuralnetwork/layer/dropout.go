package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math/rand/v2"
)

type Dropout struct {
	rate float64          // The dropout rate, i.e., the fraction of input units to drop
	mask tensor.Interface // The mask tensor that indicates which units are dropped during the forward pass
}

// NewDropout creates a new Dropout layer
func NewDropout(rate float64) *Dropout {
	return &Dropout{rate: rate}
}

// Forward pass for Dropout
func (d *Dropout) Forward(input tensor.Interface) tensor.Interface {
	maskData := make([]float64, len(input.Data()))
	for i := range maskData {
		if rand.Float64() > d.rate {
			maskData[i] = 1.0
		} else {
			maskData[i] = 0.0
		}
	}
	d.mask = tensor.NewTensor(maskData, input.Shape())
	output := input.Multiply(d.mask)
	return output
}

// Backward pass for Dropout
func (d *Dropout) Backward(grad tensor.Interface) tensor.Interface {
	output := grad.Multiply(d.mask)
	return output
}

// GetWeights returns the weights of the Dropout layer
func (d *Dropout) GetWeights() tensor.Interface {
	return nil
}

// SetWeights sets the weights of the Dropout layer
func (d *Dropout) SetWeights(weights tensor.Interface) {}

// GetBiases returns the biases of the Dropout layer
func (d *Dropout) GetBiases() tensor.Interface {
	return nil
}

// SetBiases sets the biases of the Dropout layer
func (d *Dropout) SetBiases(biases tensor.Interface) {}

// GetGradients returns the gradients of the Dropout layer
func (d *Dropout) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return nil, nil
}

// RequiresOptimisation indicates if this layer requires optimisation
func (d *Dropout) RequiresOptimisation() bool {
	return false
}

// RequiresRegularisation indicates if this layer requires regularisation
func (d *Dropout) RequiresRegularisation() bool {
	return false
}

func (d *Dropout) Name() string {
	return "Dropout"
}

func (d *Dropout) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"rate": d.rate,
	}
	return config, nil
}

func (d *Dropout) Load(config map[string]any, tensors []model.TensorData) error {
	rate, ok := config["rate"].(float64)
	if !ok {
		return errors.New("invalid rate")
	}
	d.rate = rate

	return nil
}
