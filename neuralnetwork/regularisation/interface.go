package regularisation

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

type Interface interface {
	Apply(weights, gradients tensor.Interface)
	ApplyToLoss(weights tensor.Interface) float64
	Save() map[string]any
}
