package optimiser

import (
	"github.com/google/uuid"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type SGDWithMomentum struct {
	LearningRate float64                 // The learning rate for the optimiser
	Momentum     float64                 // The momentum factor to accelerate gradients vectors in the right directions
	Velocities   map[uuid.UUID][]float64 // A map to store the moving average of the gradients
}

func NewSGDWithMomentum(learningRate, momentum float64) *SGDWithMomentum {
	return &SGDWithMomentum{
		LearningRate: learningRate,
		Momentum:     momentum,
		Velocities:   make(map[uuid.UUID][]float64),
	}
}

func (o *SGDWithMomentum) ZeroGradients(gradients tensor.Interface) {
	for i := range gradients.Data() {
		gradients.Data()[i] = 0
	}
}

func (o *SGDWithMomentum) Update(weights, gradients tensor.Interface) {
	weightData := weights.Data()
	gradientData := gradients.Data()
	id := weights.ID()

	// Initialize velocities if they don't exist
	if _, ok := o.Velocities[id]; !ok {
		o.Velocities[id] = make([]float64, len(weightData))
	}

	velocity := o.Velocities[id]

	// Update weights with momentum
	for i := range weightData {
		velocity[i] = o.Momentum*velocity[i] - o.LearningRate*gradientData[i]
		weightData[i] += velocity[i]
	}
}

func (o *SGDWithMomentum) Save() map[string]any {
	return map[string]any{
		"type":          "SGDWithMomentum",
		"learning_rate": o.LearningRate,
		"momentum":      o.Momentum,
	}
}
