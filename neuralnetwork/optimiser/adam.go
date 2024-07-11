package optimiser

import (
	"github.com/google/uuid"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type Adam struct {
	LearningRate float64                 // The learning rate for the optimiser
	Beta1        float64                 // The exponential decay rate for the first moment estimates
	Beta2        float64                 // The exponential decay rate for the second moment estimates
	Epsilon      float64                 // A small constant to prevent division by zero
	TimeStep     map[uuid.UUID]int       // A map to keep track of the time steps for each parameter
	M            map[uuid.UUID][]float64 // A map to store the first moment estimates (moving average of the gradients)
	V            map[uuid.UUID][]float64 // A map to store the second moment estimates (moving average of the squared gradients)
}

func NewAdam(learningRate, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		TimeStep:     make(map[uuid.UUID]int),
		M:            make(map[uuid.UUID][]float64),
		V:            make(map[uuid.UUID][]float64),
	}
}

func (o *Adam) ZeroGradients(gradients tensor.Interface) {
	for i := range gradients.Data() {
		gradients.Data()[i] = 0
	}
}

func (o *Adam) Update(weights, gradients tensor.Interface) {
	weightData := weights.Data()
	gradientData := gradients.Data()
	id := weights.ID()

	// Initialize m and v if they don't exist
	if _, ok := o.M[id]; !ok {
		o.M[id] = make([]float64, len(weightData))
		o.V[id] = make([]float64, len(weightData))
		o.TimeStep[id] = 0
	}

	m := o.M[id]
	v := o.V[id]
	o.TimeStep[id]++

	// Update weights using Adam
	for i := range weightData {
		m[i] = o.Beta1*m[i] + (1-o.Beta1)*gradientData[i]
		v[i] = o.Beta2*v[i] + (1-o.Beta2)*gradientData[i]*gradientData[i]

		mHat := m[i] / (1 - math.Pow(o.Beta1, float64(o.TimeStep[id])))
		vHat := v[i] / (1 - math.Pow(o.Beta2, float64(o.TimeStep[id])))

		weightData[i] -= o.LearningRate * mHat / (math.Sqrt(vHat) + o.Epsilon)
	}
}

func (o *Adam) Save() map[string]any {
	return map[string]any{
		"type":          "Adam",
		"learning_rate": o.LearningRate,
		"beta1":         o.Beta1,
		"beta2":         o.Beta2,
		"epsilon":       o.Epsilon,
	}
}
