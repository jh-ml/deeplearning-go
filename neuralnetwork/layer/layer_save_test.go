package layer_test

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"testing"
)

func TestConv2DSaveAndLoad(t *testing.T) {
	original := layer.NewConv2D(1, 32, 5, 1, 2, activation.NewReLU())

	// Save the layer
	config, tensors := original.Save()

	// Create a new layer and load the saved configuration
	loaded := &layer.Conv2D{}
	err := loaded.Load(config, tensors)
	if err != nil {
		t.Fatalf("Error loading Conv2D layer: %v", err)
	}

	// Check the configurations
	if loaded.InputDim != original.InputDim {
		t.Errorf("Expected InputDim to be %d, got %d", original.InputDim, loaded.InputDim)
	}
	if loaded.Stride != original.Stride {
		t.Errorf("Expected Stride to be %d, got %d", original.Stride, loaded.Stride)
	}
	if loaded.Padding != original.Padding {
		t.Errorf("Expected Padding to be %d, got %d", original.Padding, loaded.Padding)
	}
	if loaded.Activation.Name() != original.Activation.Name() {
		t.Errorf("Expected Activation to be %s, got %s", original.Activation.Name(), loaded.Activation.Name())
	}

	// Check the tensor data
	if !tensorEqual(original.Weights, loaded.Weights) {
		t.Error("Weights tensor mismatch")
	}
	if !tensorEqual(original.Biases, loaded.Biases) {
		t.Error("Biases tensor mismatch")
	}
}

func TestFullyConnectedSaveAndLoad(t *testing.T) {
	original := layer.NewFullyConnected(784, 128, activation.NewReLU())

	// Save the layer
	config, tensors := original.Save()

	// Create a new layer and load the saved configuration
	loaded := &layer.FullyConnected{}
	err := loaded.Load(config, tensors)
	if err != nil {
		t.Fatalf("Error loading FullyConnected layer: %v", err)
	}

	// Check the configurations
	if len(loaded.GetWeights().Shape()) != len(original.GetWeights().Shape()) {
		t.Errorf("Expected Weights shape length to be %d, got %d", len(original.GetWeights().Shape()), len(loaded.GetWeights().Shape()))
	}

	// Check the tensor data
	if !tensorEqual(original.GetWeights(), loaded.GetWeights()) {
		t.Error("Weights tensor mismatch")
	}
	if !tensorEqual(original.GetWeights(), loaded.GetWeights()) {
		t.Error("Biases tensor mismatch")
	}
}

func TestGRUSaveAndLoad(t *testing.T) {
	original := layer.NewGRU(128, 64)

	// Save the layer
	config, tensors := original.Save()

	// Create a new layer and load the saved configuration
	loaded := &layer.GRU{}
	err := loaded.Load(config, tensors)
	if err != nil {
		t.Fatalf("Error loading GRU layer: %v", err)
	}

	// Check the configurations
	if loaded.Wu.Shape()[0] != original.Wu.Shape()[0] || loaded.Wu.Shape()[1] != original.Wu.Shape()[1] {
		t.Errorf("Expected Wu shape to be %v, got %v", original.Wu.Shape(), loaded.Wu.Shape())
	}

	// Check the tensor data
	if !tensorEqual(original.Wu, loaded.Wu) {
		t.Error("Wu tensor mismatch")
	}
	if !tensorEqual(original.Bu, loaded.Bu) {
		t.Error("Bu tensor mismatch")
	}
}

// Helper function to compare two tensors
func tensorEqual(t1, t2 tensor.Interface) bool {
	if len(t1.Data()) != len(t2.Data()) || len(t1.Shape()) != len(t2.Shape()) {
		return false
	}
	for i := range t1.Data() {
		if t1.Data()[i] != t2.Data()[i] {
			return false
		}
	}
	for i := range t1.Shape() {
		if t1.Shape()[i] != t2.Shape()[i] {
			return false
		}
	}
	return true
}

func TestLSTMSaveAndLoad(t *testing.T) {
	original := layer.NewLSTM(128, 64)

	// Save the layer
	config, tensors := original.Save()

	// Create a new layer and load the saved configuration
	loaded := &layer.LSTM{}
	err := loaded.Load(config, tensors)
	if err != nil {
		t.Fatalf("Error loading LSTM layer: %v", err)
	}

	// Check the configurations
	if loaded.Wf.Shape()[0] != original.Wf.Shape()[0] || loaded.Wf.Shape()[1] != original.Wf.Shape()[1] {
		t.Errorf("Expected Wf shape to be %v, got %v", original.Wf.Shape(), loaded.Wf.Shape())
	}

	// Check the tensor data
	if !tensorEqual(original.Wf, loaded.Wf) {
		t.Error("Wf tensor mismatch")
	}
	if !tensorEqual(original.Bf, loaded.Bf) {
		t.Error("Bf tensor mismatch")
	}
}
