package loss_test

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
	"testing"
)

func TestBinaryCrossEntropy(t *testing.T) {
	bce := loss.NewBinaryCrossEntropy()

	predicted := tensor.NewTensor([]float64{0.9, 0.1, 0.8}, []int{3})
	actual := tensor.NewTensor([]float64{1, 0, 1}, []int{3})

	expectedLossData := []float64{
		-1 * math.Log(0.9),
		-1 * math.Log(0.9),
		-1 * math.Log(0.8),
	}
	expectedGradData := []float64{
		(0.9 - 1) / (0.9 * 0.1),
		(0.1 - 0) / (0.1 * 0.9),
		(0.8 - 1) / (0.8 * 0.2),
	}

	loss, grad := bce.Compute(predicted, actual)

	if !float64sEqual(loss.Data(), expectedLossData) {
		t.Errorf("BinaryCrossEntropy loss = %v, want %v", loss.Data(), expectedLossData)
	}
	if !float64sEqual(grad.Data(), expectedGradData) {
		t.Errorf("BinaryCrossEntropy grad = %v, want %v", grad.Data(), expectedGradData)
	}
}

func TestCosineProximityLoss(t *testing.T) {
	cosine := loss.NewCosineProximityLoss()

	output := tensor.NewTensor([]float64{1, 2, 3}, []int{3})
	target := tensor.NewTensor([]float64{4, 5, 6}, []int{3})

	// Manual calculation of expected values
	dotProduct := float64(1*4 + 2*5 + 3*6)
	normOutput := math.Sqrt(float64(1*1 + 2*2 + 3*3))
	normTarget := math.Sqrt(float64(4*4 + 5*5 + 6*6))
	expectedLossData := []float64{-dotProduct / (normOutput * normTarget)}

	// Gradient calculation
	expectedGradData := make([]float64, 3)
	for i := 0; i < 3; i++ {
		expectedGradData[i] = (target.Data()[i]/normTarget - (dotProduct/(normOutput*normOutput*normTarget))*output.Data()[i]) / normOutput
	}

	loss, grad := cosine.Compute(output, target)

	if !float64sEqual(loss.Data(), expectedLossData) {
		t.Errorf("CosineProximityLoss loss = %v, want %v", loss.Data(), expectedLossData)
	}
	if !float64sEqual(grad.Data(), expectedGradData) {
		t.Errorf("CosineProximityLoss grad = %v, want %v", grad.Data(), expectedGradData)
	}
}

func TestCategoricalCrossEntropyLoss(t *testing.T) {
	ce := loss.NewCategoricalCrossEntropy()

	// Adjusting the tensors to have shape [1, 3] instead of [3]
	output := tensor.NewTensor([]float64{0.8, 0.7, 0.6}, []int{1, 3})
	target := tensor.NewTensor([]float64{1, 0, 1}, []int{1, 3})

	expectedLossData := []float64{
		-1*math.Log(0.8) + -1*math.Log(0.6),
	}
	expectedGradData := []float64{
		(0.8 - 1) / (0.8 + 1e-12),
		(0.7 - 0) / (0.7 + 1e-12),
		(0.6 - 1) / (0.6 + 1e-12),
	}

	loss, grad := ce.Compute(output, target)

	if !float64sEqual(loss.Data(), expectedLossData) {
		t.Errorf("CrossEntropyLoss loss = %v, want %v", loss.Data(), expectedLossData)
	}
	if !float64sEqual(grad.Data(), expectedGradData) {
		t.Errorf("CrossEntropyLoss grad = %v, want %v", grad.Data(), expectedGradData)
	}
}

func TestMSELoss(t *testing.T) {
	mse := loss.NewMSELoss()

	output := tensor.NewTensor([]float64{3, -0.5, 2, 7}, []int{4})
	target := tensor.NewTensor([]float64{2.5, 0.0, 2, 8}, []int{4})

	expectedLoss := 0.5 * (math.Pow(3-2.5, 2) + math.Pow(-0.5-0.0, 2) + math.Pow(2-2, 2) + math.Pow(7-8, 2))
	expectedGradData := []float64{
		3 - 2.5,
		-0.5 - 0.0,
		2 - 2,
		7 - 8,
	}

	loss, grad := mse.Compute(output, target)

	if len(loss.Data()) != 1 || math.Abs(loss.Data()[0]-expectedLoss) > 1e-6 {
		t.Errorf("MSELoss loss = %v, want %v", loss.Data(), expectedLoss)
	}
	if !float64sEqual(grad.Data(), expectedGradData) {
		t.Errorf("MSELoss grad = %v, want %v", grad.Data(), expectedGradData)
	}
}

// Helper function to compare two float64 slices
func float64sEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			return false
		}
	}
	return true
}
