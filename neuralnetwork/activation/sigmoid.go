package activation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	result := make([]float64, len(data))
	for i, v := range data {
		result[i] = 1 / (1 + math.Exp(-v))
	}
	output := input.Clone()
	output.SetData(result)
	return output
}

func (s *Sigmoid) Backward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	result := make([]float64, len(data))
	for i, v := range data {
		sigmoid := 1 / (1 + math.Exp(-v))
		result[i] = sigmoid * (1 - sigmoid)
	}
	output := input.Clone()
	output.SetData(result)
	return output
}

func (s *Sigmoid) Name() string {
	return "Sigmoid"
}
