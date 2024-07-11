package activation_test

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
	"reflect"
	"testing"
)

func TestLeakyReLUForward(t *testing.T) {
	alpha := 0.1
	leakyReLU := activation.NewLeakyReLU(alpha)

	inputData := []float64{1.0, -2.0, 3.0, -4.0}
	expectedOutputData := []float64{1.0, -0.2, 3.0, -0.4}

	input := tensor.NewTensor(inputData, []int{4})
	output := leakyReLU.Forward(input)

	if !reflect.DeepEqual(output.Data(), expectedOutputData) {
		t.Errorf("Forward() output = %v, want %v", output.Data(), expectedOutputData)
	}
}

func TestLeakyReLUBackward(t *testing.T) {
	alpha := 0.1
	leakyReLU := activation.NewLeakyReLU(alpha)

	inputData := []float64{1.0, -2.0, 3.0, -4.0}
	expectedGradData := []float64{1.0, alpha, 1.0, alpha}

	input := tensor.NewTensor(inputData, []int{4})
	grad := leakyReLU.Backward(input)

	if !reflect.DeepEqual(grad.Data(), expectedGradData) {
		t.Errorf("Backward() output = %v, want %v", grad.Data(), expectedGradData)
	}
}

// TestReLUForward tests the Forward method of ReLU
func TestReLUForward(t *testing.T) {
	relu := activation.NewReLU()

	inputData := []float64{1.0, -2.0, 3.0, -4.0}
	expectedOutputData := []float64{1.0, 0.0, 3.0, 0.0}

	input := tensor.NewTensor(inputData, []int{4})
	output := relu.Forward(input)

	if !reflect.DeepEqual(output.Data(), expectedOutputData) {
		t.Errorf("ReLU Forward() output = %v, want %v", output.Data(), expectedOutputData)
	}
}

// TestReLUBackward tests the Backward method of ReLU
func TestReLUBackward(t *testing.T) {
	relu := activation.NewReLU()

	inputData := []float64{1.0, -2.0, 3.0, -4.0}
	expectedGradData := []float64{1.0, 0.0, 1.0, 0.0}

	input := tensor.NewTensor(inputData, []int{4})
	grad := relu.Backward(input)

	if !reflect.DeepEqual(grad.Data(), expectedGradData) {
		t.Errorf("ReLU Backward() output = %v, want %v", grad.Data(), expectedGradData)
	}
}

// TestSigmoidForward tests the Forward method of Sigmoid
func TestSigmoidForward(t *testing.T) {
	sigmoid := activation.NewSigmoid()

	inputData := []float64{0.0, 2.0, -2.0}
	expectedOutputData := []float64{
		1 / (1 + math.Exp(0)),
		1 / (1 + math.Exp(-2)),
		1 / (1 + math.Exp(2)),
	}

	input := tensor.NewTensor(inputData, []int{3})
	output := sigmoid.Forward(input)

	if !reflect.DeepEqual(output.Data(), expectedOutputData) {
		t.Errorf("Sigmoid Forward() output = %v, want %v", output.Data(), expectedOutputData)
	}
}

// TestSigmoidBackward tests the Backward method of Sigmoid
func TestSigmoidBackward(t *testing.T) {
	sigmoid := activation.NewSigmoid()

	inputData := []float64{0.0, 2.0, -2.0}
	expectedGradData := []float64{
		func(x float64) float64 { return x * (1 - x) }(1 / (1 + math.Exp(0))),
		func(x float64) float64 { return x * (1 - x) }(1 / (1 + math.Exp(-2))),
		func(x float64) float64 { return x * (1 - x) }(1 / (1 + math.Exp(2))),
	}

	input := tensor.NewTensor(inputData, []int{3})
	grad := sigmoid.Backward(input)

	if !reflect.DeepEqual(grad.Data(), expectedGradData) {
		t.Errorf("Sigmoid Backward() output = %v, want %v", grad.Data(), expectedGradData)
	}
}

// TestSoftmaxForward tests the Forward method of Softmax
func TestSoftmaxForward(t *testing.T) {
	softmax := activation.NewSoftmax()

	inputData := []float64{1.0, 2.0, 3.0}
	maxVal := 3.0
	expectedOutputData := []float64{
		math.Exp(1.0-maxVal) / (math.Exp(1.0-maxVal) + math.Exp(2.0-maxVal) + math.Exp(3.0-maxVal)),
		math.Exp(2.0-maxVal) / (math.Exp(1.0-maxVal) + math.Exp(2.0-maxVal) + math.Exp(3.0-maxVal)),
		math.Exp(3.0-maxVal) / (math.Exp(1.0-maxVal) + math.Exp(2.0-maxVal) + math.Exp(3.0-maxVal)),
	}

	input := tensor.NewTensor(inputData, []int{3})
	output := softmax.Forward(input)

	if !reflect.DeepEqual(output.Data(), expectedOutputData) {
		t.Errorf("Softmax Forward() output = %v, want %v", output.Data(), expectedOutputData)
	}
}

// TestSoftmaxBackward tests the Backward method of Softmax
func TestSoftmaxBackward(t *testing.T) {
	softmax := activation.NewSoftmax()

	inputData := []float64{0.2, 0.5, 0.3}
	expectedGradData := []float64{
		0.2 * (1 - 0.2),
		0.5 * (1 - 0.5),
		0.3 * (1 - 0.3),
	}

	input := tensor.NewTensor(inputData, []int{3})
	grad := softmax.Backward(input)

	roundedGrad := make([]float64, len(grad.Data()))
	for i, v := range grad.Data() {
		roundedGrad[i] = math.Round(v*1e9) / 1e9 // Adjust the precision for comparison
	}

	roundedExpectedGradData := make([]float64, len(expectedGradData))
	for i, v := range expectedGradData {
		roundedExpectedGradData[i] = math.Round(v*1e9) / 1e9 // Adjust the precision for comparison
	}

	if !reflect.DeepEqual(roundedGrad, roundedExpectedGradData) {
		t.Errorf("Softmax Backward() output = %v, want %v", roundedGrad, roundedExpectedGradData)
	}
}

// TestTanhForward tests the Forward method of Tanh
func TestTanhForward(t *testing.T) {
	tanh := activation.NewTanh()

	inputData := []float64{0.0, 1.0, -1.0}
	expectedOutputData := []float64{
		math.Tanh(0.0),
		math.Tanh(1.0),
		math.Tanh(-1.0),
	}

	input := tensor.NewTensor(inputData, []int{3})
	output := tanh.Forward(input)

	if !reflect.DeepEqual(output.Data(), expectedOutputData) {
		t.Errorf("Tanh Forward() output = %v, want %v", output.Data(), expectedOutputData)
	}
}

// TestTanhBackward tests the Backward method of Tanh
func TestTanhBackward(t *testing.T) {
	tanh := activation.NewTanh()

	inputData := []float64{0.0, 1.0, -1.0}
	expectedGradData := []float64{
		1 - math.Pow(math.Tanh(0.0), 2),
		1 - math.Pow(math.Tanh(1.0), 2),
		1 - math.Pow(math.Tanh(-1.0), 2),
	}

	input := tensor.NewTensor(inputData, []int{3})
	grad := tanh.Backward(input)

	if !reflect.DeepEqual(grad.Data(), expectedGradData) {
		t.Errorf("Tanh Backward() output = %v, want %v", grad.Data(), expectedGradData)
	}
}
