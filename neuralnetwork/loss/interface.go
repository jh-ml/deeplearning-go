package loss

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

type Interface interface {
	Compute(predicted, actual tensor.Interface) (loss tensor.Interface, grad tensor.Interface)
	Save() map[string]any
}
