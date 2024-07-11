package loss

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type CategoricalCrossEntropy struct{}

func NewCategoricalCrossEntropy() *CategoricalCrossEntropy {
	return &CategoricalCrossEntropy{}
}

// Compute calculates the categorical cross-entropy loss and its gradient
func (l *CategoricalCrossEntropy) Compute(predicted, actual tensor.Interface) (tensor.Interface, tensor.Interface) {
	loss := tensor.NewZerosTensor([]int{predicted.Shape()[0]})
	gradient := tensor.NewZerosTensor(predicted.Shape())
	epsilon := 1e-12

	for i := 0; i < predicted.Shape()[0]; i++ {
		for j := 0; j < predicted.Shape()[1]; j++ {
			p := predicted.Get(i, j)
			a := actual.Get(i, j)
			loss.Data()[i] += -a * math.Log(p+epsilon)
			gradient.Set((p-a)/(p+epsilon), i, j)
		}
	}
	return loss, gradient
}

func (l *CategoricalCrossEntropy) Save() map[string]any {
	return map[string]any{
		"type": "CategoricalCrossEntropy",
	}
}
