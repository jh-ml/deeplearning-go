package activation

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Interface interface {
	Forward(input tensor.Interface) tensor.Interface
	Backward(input tensor.Interface) tensor.Interface
	Name() string
}

func NewActivationByName(name string) (Interface, error) {
	switch name {
	case "ReLU":
		return NewReLU(), nil
	case "Sigmoid":
		return NewSigmoid(), nil
	case "Tanh":
		return NewTanh(), nil
	case "Softmax":
		return NewSoftmax(), nil
	default:
		return nil, errors.New("unknown activation function: " + name)
	}
}
