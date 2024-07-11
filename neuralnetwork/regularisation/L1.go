package regularisation

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type L1Regulariser struct {
	Lambda float64
}

func NewL1Regulariser(lambda float64) *L1Regulariser {
	return &L1Regulariser{Lambda: lambda}
}

func (r *L1Regulariser) Apply(weights, gradients tensor.Interface) {
	weightsData := weights.Data()
	gradientsData := gradients.Data()
	for i := range weightsData {
		gradientsData[i] += r.Lambda * sign(weightsData[i])
	}
}

func (r *L1Regulariser) ApplyToLoss(weights tensor.Interface) float64 {
	weightsData := weights.Data()
	var regLoss float64
	for _, w := range weightsData {
		regLoss += r.Lambda * math.Abs(w)
	}
	return regLoss
}

func (r *L1Regulariser) Save() map[string]any {
	return map[string]any{
		"type":   "L1",
		"lambda": r.Lambda,
	}
}
