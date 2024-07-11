package optimiser

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Interface interface {
	ZeroGradients(gradients tensor.Interface)
	Update(weights, gradients tensor.Interface)
	Save() map[string]any
}
