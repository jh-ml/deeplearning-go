package loss

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type BinaryCrossEntropy struct{}

func NewBinaryCrossEntropy() *BinaryCrossEntropy {
	return &BinaryCrossEntropy{}
}

// Compute calculates the binary cross-entropy loss and its gradient
func (l *BinaryCrossEntropy) Compute(predicted, actual tensor.Interface) (tensor.Interface, tensor.Interface) {
	loss := tensor.NewZerosTensor(predicted.Shape())
	gradient := tensor.NewZerosTensor(predicted.Shape())
	epsilon := 1e-12

	for i := 0; i < predicted.Size(); i++ {
		p := predicted.Data()[i]
		a := actual.Data()[i]
		loss.Data()[i] = -a*math.Log(p+epsilon) - (1-a)*math.Log(1-p+epsilon)
		gradient.Data()[i] = (p - a) / ((p * (1 - p)) + epsilon)
	}
	return loss, gradient
}

func (l *BinaryCrossEntropy) Save() map[string]any {
	return map[string]any{
		"type": "BinaryCrossEntropy",
	}
}
