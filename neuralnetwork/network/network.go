package network

import (
	"fmt"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type NeuralNetwork struct {
	layers         []layer.Interface
	lossFunction   loss.Interface
	regularisation regularisation.Interface
	optimiser      optimiser.Interface
}

// NewNeuralNetwork creates a new NeuralNetwork
func NewNeuralNetwork(layers []layer.Interface,
	optimiser optimiser.Interface,
	lossFunction loss.Interface,
	regularization regularisation.Interface) *NeuralNetwork {
	return &NeuralNetwork{
		layers:         layers,
		lossFunction:   lossFunction,
		regularisation: regularization,
		optimiser:      optimiser,
	}
}

// AddLayer adds a layer to the neural network
func (nn *NeuralNetwork) AddLayer(l layer.Interface) {
	nn.layers = append(nn.layers, l)
}

// GetLayers returns all the layers in the neural network
func (nn *NeuralNetwork) GetLayers() []layer.Interface {
	return nn.layers
}

// Forward executes the forward pass
func (nn *NeuralNetwork) Forward(input tensor.Interface) tensor.Interface {
	output := input
	for _, l := range nn.layers {
		output = l.Forward(output)
	}
	return output
}

// Backward executes the backward pass and returns the gradients for each layer
func (nn *NeuralNetwork) Backward(grad tensor.Interface) tensor.Interface {
	output := grad
	for i := len(nn.layers) - 1; i >= 0; i-- {
		output = nn.layers[i].Backward(output)
	}
	return output
}

// Train trains the network
func (nn *NeuralNetwork) Train(data, targets []tensor.Interface, epochs int) float64 {
	var totalLoss float64
	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float64
		for i := 0; i < len(data); i++ {
			output := nn.Forward(data[i])
			lossV, grad := nn.lossFunction.Compute(output, targets[i])
			epochLoss += lossV.Data()[0]
			nn.Backward(grad)
			nn.Regularise()
			nn.Optimise()
			nn.ZeroGradients()
		}
		epochLoss /= float64(len(data)) // Average the loss over the number of samples
		totalLoss += epochLoss          // Accumulate the averaged loss
		fmt.Printf("Epoch %d, Loss: %f\n", epoch, epochLoss)
	}
	return totalLoss
}

func (nn *NeuralNetwork) Predict(input tensor.Interface) tensor.Interface {
	return nn.Forward(input)
}

func (nn *NeuralNetwork) Regularise() {
	for _, l := range nn.GetLayers() {
		if !l.RequiresOptimisation() {
			continue
		}
		weights := l.GetWeights()
		biases := l.GetBiases()
		gradWeights, gradBiases := l.GetGradients()

		// Apply regularisation to gradients
		nn.regularisation.Apply(weights, gradWeights)
		nn.regularisation.Apply(biases, gradBiases)
	}
}

func (nn *NeuralNetwork) ZeroGradients() {
	for _, l := range nn.GetLayers() {
		if !l.RequiresOptimisation() {
			continue
		}
		gradWeights, gradBiases := l.GetGradients()
		nn.optimiser.ZeroGradients(gradWeights)
		nn.optimiser.ZeroGradients(gradBiases)
	}
}

func (nn *NeuralNetwork) Optimise() {
	for _, l := range nn.GetLayers() {
		if !l.RequiresOptimisation() {
			continue
		}
		weights := l.GetWeights()
		biases := l.GetBiases()
		gradWeights, gradBiases := l.GetGradients()

		// Update weights and biases using the optimiser
		nn.optimiser.Update(weights, gradWeights)
		nn.optimiser.Update(biases, gradBiases)
	}
}
