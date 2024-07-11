package regularisation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type ElasticNetRegulariser struct {
	Lambda1 float64
	Lambda2 float64
}

func NewElasticNetRegulariser(lambda1, lambda2 float64) *ElasticNetRegulariser {
	return &ElasticNetRegulariser{Lambda1: lambda1, Lambda2: lambda2}
}

func (r *ElasticNetRegulariser) Apply(weights, gradients tensor.Interface) {
	weightsData := weights.Data()
	gradientsData := gradients.Data()
	for i := range weightsData {
		gradientsData[i] += r.Lambda1*sign(weightsData[i]) + r.Lambda2*weightsData[i]
	}
}

func (r *ElasticNetRegulariser) ApplyToLoss(weights tensor.Interface) float64 {
	weightsData := weights.Data()
	var regLoss float64
	for _, w := range weightsData {
		regLoss += r.Lambda1*math.Abs(w) + r.Lambda2*0.5*w*w
	}
	return regLoss
}

func (r *ElasticNetRegulariser) Save() map[string]any {
	return map[string]any{
		"type":    "ElasticNet",
		"lambda1": r.Lambda1,
		"lambda2": r.Lambda2,
	}
}
