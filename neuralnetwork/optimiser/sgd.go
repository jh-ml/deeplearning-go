package optimiser

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

type SGD struct {
	LearningRate float64
}

func NewSGD(LearningRate float64) *SGD {
	return &SGD{LearningRate: LearningRate}
}

// ZeroGradients sets all gradients to zero
func (sgd *SGD) ZeroGradients(gradients tensor.Interface) {
	for i := range gradients.Data() {
		gradients.Data()[i] = 0
	}
}

func (sgd *SGD) Update(weights, gradients tensor.Interface) {
	weightData := weights.Data()
	gradientData := gradients.Data()

	for i := range weightData {
		weightData[i] -= sgd.LearningRate * gradientData[i]
	}
}

func (sgd *SGD) Save() map[string]any {
	return map[string]any{
		"type":          "SGD",
		"learning_rate": sgd.LearningRate,
	}
}
