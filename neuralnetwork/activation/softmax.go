package activation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type Softmax struct{}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	maxVal := math.Inf(-1)
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i, v := range data {
		data[i] = math.Exp(v - maxVal)
		sum += data[i]
	}
	for i := range data {
		data[i] /= sum
	}
	output := input.Clone()
	output.SetData(data)
	return output
}

func (s *Softmax) Backward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	result := make([]float64, len(data))
	for i, v := range data {
		result[i] = v * (1 - v)
	}
	output := input.Clone()
	output.SetData(result)
	return output
}

func (s *Softmax) Name() string {
	return "Softmax"
}
