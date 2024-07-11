package regularisation

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

type L2Regulariser struct {
	Lambda float64
}

func NewL2Regulariser(lambda float64) *L2Regulariser {
	return &L2Regulariser{Lambda: lambda}
}

func (r *L2Regulariser) Apply(weights, gradients tensor.Interface) {
	for i := range weights.Data() {
		gradients.Data()[i] += r.Lambda * weights.Data()[i]
	}
}

func (r *L2Regulariser) ApplyToLoss(weights tensor.Interface) float64 {
	var regLoss float64
	for _, w := range weights.Data() {
		regLoss += r.Lambda * w * w
	}
	return regLoss
}

func (r *L2Regulariser) Save() map[string]any {
	return map[string]any{
		"type":   "L2",
		"lambda": r.Lambda,
	}
}
