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
	epsilon := 1e-12 // Small value to prevent division by zero

	for i := 0; i < predicted.Size(); i++ {
		p := predicted.Data()[i]
		a := actual.Data()[i]
		// Calculate the binary cross-entropy loss
		loss.Data()[i] = -a*math.Log(p+epsilon) - (1-a)*math.Log(1-p+epsilon)
		// Calculate the gradient (derivative of the loss function with respect to the predicted value)
		gradient.Data()[i] = (p - a) / ((p * (1 - p)) + epsilon)
	}
	return loss, gradient
}

func (l *BinaryCrossEntropy) Save() map[string]any {
	return map[string]any{
		"type": "BinaryCrossEntropy",
	}
}
