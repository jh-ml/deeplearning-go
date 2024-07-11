package activation

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

type LeakyReLU struct {
	alpha float64
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{alpha: alpha}
}

func (l *LeakyReLU) Forward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	output := make([]float64, len(data))
	for i, v := range data {
		if v > 0 {
			output[i] = v
		} else {
			output[i] = l.alpha * v
		}
	}
	return tensor.NewTensor(output, input.Shape())
}

func (l *LeakyReLU) Backward(input tensor.Interface) tensor.Interface {
	data := input.Data()
	grad := make([]float64, len(data))
	for i, v := range data {
		if v > 0 {
			grad[i] = 1
		} else {
			grad[i] = l.alpha
		}
	}
	return tensor.NewTensor(grad, input.Shape())
}

func (r *LeakyReLU) Name() string {
	return "LeakyReLU"
}
