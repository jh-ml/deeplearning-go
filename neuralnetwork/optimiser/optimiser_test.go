package optimiser_test

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
	"testing"
)

func TestAdam(t *testing.T) {
	adam := optimiser.NewAdam(0.001, 0.9, 0.999, 1e-8)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.8}, []int{3})
	gradients := tensor.NewTensor([]float64{0.1, -0.2, 0.3}, []int{3})

	// Perform one update
	adam.Update(weights, gradients)

	// Expected values after one update
	expectedWeights := []float64{
		0.4990000001,
		-0.29900000005,
		0.7990000000333334,
	}

	if !float64sEqual(weights.Data(), expectedWeights) {
		t.Errorf("Adam Update() weights = %v, want %v", weights.Data(), expectedWeights)
	}
}

func TestRMSProp(t *testing.T) {
	rmsprop := optimiser.NewRMSProp(0.01, 0.9, 1e-8)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.8}, []int{3})
	gradients := tensor.NewTensor([]float64{0.1, -0.2, 0.3}, []int{3})

	// Perform one update
	rmsprop.Update(weights, gradients)

	// Manually calculate the expected values for the first update
	beta := 0.9
	epsilon := 1e-8
	learningRate := 0.01

	meanSquares := []float64{0, 0, 0}
	expectedWeights := make([]float64, len(weights.Data()))

	for i := range meanSquares {
		meanSquares[i] = beta*meanSquares[i] + (1-beta)*gradients.Data()[i]*gradients.Data()[i]
		expectedWeights[i] = weights.Data()[i] - learningRate*gradients.Data()[i]/(math.Sqrt(meanSquares[i])+epsilon)
	}

	expectedWeights = []float64{
		0.5 - 0.01*0.1/(math.Sqrt(0.001)+1e-8),
		-0.3 - 0.01*-0.2/(math.Sqrt(0.004)+1e-8),
		0.8 - 0.01*0.3/(math.Sqrt(0.009)+1e-8),
	}

	if !float64sEqual(weights.Data(), expectedWeights) {
		t.Errorf("RMSProp Update() weights = %v, want %v", weights.Data(), expectedWeights)
	}
}

func TestSGD(t *testing.T) {
	sgd := optimiser.NewSGD(0.01)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.8}, []int{3})
	gradients := tensor.NewTensor([]float64{0.1, -0.2, 0.3}, []int{3})

	// Perform one update
	sgd.Update(weights, gradients)

	// Expected values after one update
	expectedWeights := []float64{
		0.499,
		-0.298,
		0.797,
	}

	if !float64sEqual(weights.Data(), expectedWeights) {
		t.Errorf("SGD Update() weights = %v, want %v", weights.Data(), expectedWeights)
	}
}

func TestSGDWithMomentum(t *testing.T) {
	sgdMomentum := optimiser.NewSGDWithMomentum(0.01, 0.9)

	weights := tensor.NewTensor([]float64{0.5, -0.3, 0.8}, []int{3})
	gradients := tensor.NewTensor([]float64{0.1, -0.2, 0.3}, []int{3})

	// Perform one update
	sgdMomentum.Update(weights, gradients)

	// Expected values after one update (calculated manually or using a verified implementation)
	expectedWeights := []float64{
		0.499,
		-0.298,
		0.797,
	}

	if !float64sEqual(weights.Data(), expectedWeights) {
		t.Errorf("SGDWithMomentum Update() weights = %v, want %v", weights.Data(), expectedWeights)
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
