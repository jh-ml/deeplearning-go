package loss

import "github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"

// MSELoss implements the Loss interface
type MSELoss struct{}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

// Compute computes the mean squared error loss and its gradient
func (l *MSELoss) Compute(predicted, actual tensor.Interface) (tensor.Interface, tensor.Interface) {
	if len(predicted.Shape()) != len(actual.Shape()) {
		panic("shape mismatch between predicted and actual tensors")
	}

	loss := 0.0
	predData := predicted.Data()
	actData := actual.Data()

	// Compute the loss
	for i := 0; i < len(predData); i++ {
		diff := predData[i] - actData[i]
		loss += diff * diff
	}
	loss *= 0.5

	// Compute the gradient
	grad := make([]float64, len(predData))
	for i := 0; i < len(predData); i++ {
		grad[i] = predData[i] - actData[i]
	}

	lossTensor := tensor.NewTensor([]float64{loss}, []int{1})
	gradTensor := tensor.NewTensor(grad, predicted.Shape())

	return lossTensor, gradTensor
}

func (l *MSELoss) Save() map[string]any {
	return map[string]any{
		"type": "MeanSquaredError",
	}
}
