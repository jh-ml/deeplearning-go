package optimiser

import (
	"github.com/google/uuid"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type RMSProp struct {
	LearningRate float64                 // The learning rate for the optimiser
	Beta         float64                 // The decay rate for the moving average of squared gradients
	Epsilon      float64                 // A small constant to prevent division by zero
	MeanSquares  map[uuid.UUID][]float64 // A map to store the moving average of squared gradients for each parameter
}

func NewRMSProp(learningRate, beta, epsilon float64) *RMSProp {
	return &RMSProp{
		LearningRate: learningRate,
		Beta:         beta,
		Epsilon:      epsilon,
		MeanSquares:  make(map[uuid.UUID][]float64),
	}
}

func (o *RMSProp) ZeroGradients(gradients tensor.Interface) {
	for i := range gradients.Data() {
		gradients.Data()[i] = 0
	}
}

func (o *RMSProp) Update(weights, gradients tensor.Interface) {
	weightData := weights.Data()
	gradientData := gradients.Data()
	id := weights.ID()

	// Initialize mean squares if they don't exist
	if _, ok := o.MeanSquares[id]; !ok {
		o.MeanSquares[id] = make([]float64, len(weightData))
	}

	meanSquares := o.MeanSquares[id]

	// Update weights using RMSProp
	for i := range weightData {
		meanSquares[i] = o.Beta*meanSquares[i] + (1-o.Beta)*gradientData[i]*gradientData[i]
		weightData[i] -= o.LearningRate * gradientData[i] / (math.Sqrt(meanSquares[i]) + o.Epsilon)
	}
}

func (o *RMSProp) Save() map[string]any {
	return map[string]any{
		"type":          "RMSProp",
		"learning_rate": o.LearningRate,
		"beta":          o.Beta,
		"epsilon":       o.Epsilon,
	}
}
