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
	// Extract data from the tensors
	outputData := predicted.Data()
	targetData := actual.Data()
	length := len(outputData)

	// Initialize variables for the dot product and norms
	dotProduct := 0.0
	normOutput := 0.0
	normTarget := 0.0

	// Compute the dot product of the predicted and actual values,
	// and the L2 norms (squared sums) of the predicted and actual values
	for i := 0; i < length; i++ {
		dotProduct += outputData[i] * targetData[i]
		normOutput += outputData[i] * outputData[i]
		normTarget += targetData[i] * targetData[i]
	}

	// Compute the L2 norms
	normOutput = math.Sqrt(normOutput)
	normTarget = math.Sqrt(normTarget)

	// Compute the cosine proximity loss value
	lossValue := -dotProduct / (normOutput * normTarget)

	// Initialize the loss tensor
	loss := tensor.NewZerosTensor([]int{1})
	loss.Data()[0] = lossValue

	// Initialize the gradient tensor
	gradient := tensor.NewZerosTensor(predicted.Shape())
	gradientData := gradient.Data()

	// Compute the gradient of the cosine proximity loss with respect to the predicted values
	for i := 0; i < length; i++ {
		// The formula for the gradient is:
		// gradient = (t_i / norm_t) - ((dot_product / (norm_y * norm_y * norm_t)) * y_i)
		// where t_i is the actual value, y_i is the predicted value,
		// dot_product is the dot product of the predicted and actual values,
		// norm_y is the L2 norm of the predicted values, and norm_t is the L2 norm of the actual values
		gradientData[i] = (targetData[i]/normTarget - (dotProduct/(normOutput*normOutput*normTarget))*outputData[i]) / normOutput
	}

	return loss, gradient
}

func (l *CosineProximityLoss) Save() map[string]any {
	return map[string]any{
		"type": "CosineProximityLoss",
	}
}
