package activation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type Tanh struct{}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (t *Tanh) Forward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	result := make([]float64, len(data))
	for i, v := range data {
		result[i] = math.Tanh(v)
	}
	output := input.Clone()
	output.SetData(result)
	return output
}

func (t *Tanh) Backward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	result := make([]float64, len(data))
	for i, v := range data {
		tanh := math.Tanh(v)
		result[i] = 1 - tanh*tanh
	}
	output := input.Clone()
	output.SetData(result)
	return output
}

func (t *Tanh) Name() string {
	return "Tanh"
}
