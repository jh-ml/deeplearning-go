package regularisation_test

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
	"testing"
)

func TestElasticNetRegulariser(t *testing.T) {
	lambda1 := 0.1
	lambda2 := 0.2
	regulariser := regularisation.NewElasticNetRegulariser(lambda1, lambda2)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.0, 0.8}, []int{4})
	gradients := tensor.NewTensor([]float64{0.1, 0.1, 0.1, 0.1}, []int{4})

	regulariser.Apply(weights, gradients)

	expectedGradients := []float64{
		0.1 + lambda1*sign(0.5) + lambda2*0.5,
		0.1 + lambda1*sign(-0.3) + lambda2*(-0.3),
		0.1 + lambda1*sign(0.0) + lambda2*0.0,
		0.1 + lambda1*sign(0.8) + lambda2*0.8,
	}

	if !float64sEqual(gradients.Data(), expectedGradients) {
		t.Errorf("ElasticNetRegulariser Apply() gradients = %v, want %v", gradients.Data(), expectedGradients)
	}

	expectedLoss := lambda1*(math.Abs(0.5)+math.Abs(-0.3)+math.Abs(0.0)+math.Abs(0.8)) +
		lambda2*0.5*(0.5*0.5+(-0.3)*(-0.3)+0.0*0.0+0.8*0.8)

	loss := regulariser.ApplyToLoss(weights)

	if math.Abs(loss-expectedLoss) > 1e-9 {
		t.Errorf("ElasticNetRegulariser ApplyToLoss() loss = %v, want %v", loss, expectedLoss)
	}
}

func TestL1Regulariser(t *testing.T) {
	lambda := 0.1
	regulariser := regularisation.NewL1Regulariser(lambda)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.0, 0.8}, []int{4})
	gradients := tensor.NewTensor([]float64{0.1, 0.1, 0.1, 0.1}, []int{4})

	regulariser.Apply(weights, gradients)

	expectedGradients := []float64{
		0.1 + lambda*sign(0.5),
		0.1 + lambda*sign(-0.3),
		0.1 + lambda*sign(0.0),
		0.1 + lambda*sign(0.8),
	}

	if !float64sEqual(gradients.Data(), expectedGradients) {
		t.Errorf("L1Regulariser Apply() gradients = %v, want %v", gradients.Data(), expectedGradients)
	}

	expectedLoss := lambda * (math.Abs(0.5) + math.Abs(-0.3) + math.Abs(0.0) + math.Abs(0.8))

	loss := regulariser.ApplyToLoss(weights)

	if math.Abs(loss-expectedLoss) > 1e-9 {
		t.Errorf("L1Regulariser ApplyToLoss() loss = %v, want %v", loss, expectedLoss)
	}
}

func TestL2Regulariser(t *testing.T) {
	lambda := 0.1
	regulariser := regularisation.NewL2Regulariser(lambda)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.0, 0.8}, []int{4})
	gradients := tensor.NewTensor([]float64{0.1, 0.1, 0.1, 0.1}, []int{4})

	regulariser.Apply(weights, gradients)

	expectedGradients := []float64{
		0.1 + lambda*0.5,
		0.1 + lambda*(-0.3),
		0.1 + lambda*0.0,
		0.1 + lambda*0.8,
	}

	if !float64sEqual(gradients.Data(), expectedGradients) {
		t.Errorf("L2Regulariser Apply() gradients = %v, want %v", gradients.Data(), expectedGradients)
	}

	expectedLoss := lambda * (0.5*0.5 + (-0.3)*(-0.3) + 0.0*0.0 + 0.8*0.8)

	loss := regulariser.ApplyToLoss(weights)

	if math.Abs(loss-expectedLoss) > 1e-9 {
		t.Errorf("L2Regulariser ApplyToLoss() loss = %v, want %v", loss, expectedLoss)
	}
}

// Helper function for creating float64 slices
func float64sEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			return false
		}
	}
	return true
}

func sign(x float64) float64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
