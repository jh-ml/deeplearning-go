package examples

import (
	"fmt"
	"github.com/jh-ml/deeplearning-go/data"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/network"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
)

func ConvNet() {
	// Define the path to the MNIST data
	dataPath := "./data/mnist"

	// Load MNIST data
	trainImages, trainLabels := data.LoadTrainingData(dataPath)
	testImages, testLabels := data.LoadTestData(dataPath)

	layers := []layer.Interface{
		layer.NewReshape([]int{1, 784}, []int{1, 1, 28, 28}),
		layer.NewConv2D(1, 32, 5, 1, 2, activation.NewReLU()),
		layer.NewMaxPooling(2, 2),
		layer.NewConv2D(32, 64, 5, 1, 2, activation.NewReLU()),
		layer.NewMaxPooling(2, 2),
		layer.NewFlatten([]int{1, 64, 7, 7}),
		layer.NewFullyConnected(64*7*7, 128, activation.NewReLU()),
		layer.NewFullyConnected(128, 10, activation.NewSoftmax()),
	}

	nn := network.NewNeuralNetwork(
		layers,
		optimiser.NewSGD(0.001),
		loss.NewCategoricalCrossEntropy(),
		regularisation.NewL2Regulariser(0.01),
	)

	epochs := 10
	totalLoss := nn.Train(trainImages, trainLabels, epochs)

	modelFilename := "Conv2D_model_config.json"
	// Save the model
	if err := nn.SaveModel(modelFilename, "Conv2D", "MNIST", totalLoss); err != nil {
		fmt.Println("Error saving model:", err)
		return
	}

	fmt.Println("Model saved successfully.")

	// Load the model
	loadedNN, err := network.LoadModel(modelFilename)
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}

	fmt.Println("Model loaded successfully")

	// Now we need to test the predictions
	test(loadedNN, testImages, testLabels)
}
