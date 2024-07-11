package loss

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type CosineProximityLoss struct{}

func NewCosineProximityLoss() *CosineProximityLoss {
	return &CosineProximityLoss{}
}

// Compute calculates the cosine proximity loss and its gradient
func (l *CosineProximityLoss) Compute(predicted, actual tensor.Interface) (tensor.Interface, tensor.Interface) {
	outputData := predicted.Data()
	targetData := actual.Data()
	length := len(outputData)

	dotProduct := 0.0
	normOutput := 0.0
	normTarget := 0.0

	for i := 0; i < length; i++ {
		dotProduct += outputData[i] * targetData[i]
		normOutput += outputData[i] * outputData[i]
		normTarget += targetData[i] * targetData[i]
	}

	normOutput = math.Sqrt(normOutput)
	normTarget = math.Sqrt(normTarget)

	lossValue := -dotProduct / (normOutput * normTarget)

	loss := tensor.NewZerosTensor([]int{1})
	loss.Data()[0] = lossValue

	gradient := tensor.NewZerosTensor(predicted.Shape())
	gradientData := gradient.Data()

	for i := 0; i < length; i++ {
		gradientData[i] = (targetData[i]/normTarget - (dotProduct/(normOutput*normOutput*normTarget))*outputData[i]) / normOutput
	}

	return loss, gradient
}

func (l *CosineProximityLoss) Save() map[string]any {
	return map[string]any{
		"type": "CosineProximityLoss",
	}
}
