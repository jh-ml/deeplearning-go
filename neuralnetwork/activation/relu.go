package activation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type ReLU struct{}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(input tensor.Interface) tensor.Interface {
	result := input.Clone()
	data := result.Data()
	for i, x := range data {
		if x <= 0 {
			data[i] = 0
		}
	}
	result.SetData(data)
	return result
}

func (r *ReLU) Backward(input tensor.Interface) tensor.Interface {
	result := input.Clone()
	data := result.Data()
	for i, x := range data {
		if x <= 0 {
			data[i] = 0
		} else {
			data[i] = 1
		}
	}
	result.SetData(data)
	return result
}

func (r *ReLU) Name() string {
	return "ReLU"
}
