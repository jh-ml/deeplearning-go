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

func ThreeLayer() {
	// Define the path to the MNIST data
	dataPath := "./data/mnist"

	// Load MNIST data
	trainImages, trainLabels := data.LoadTrainingData(dataPath)
	testImages, testLabels := data.LoadTestData(dataPath)

	layers := []layer.Interface{
		layer.NewFullyConnected(784, 128, activation.NewReLU()),
		layer.NewDropout(0.2),
		layer.NewFullyConnected(128, 64, activation.NewReLU()),
		layer.NewDropout(0.3),
		layer.NewFullyConnected(64, 10, activation.NewSoftmax()),
	}

	nn := network.NewNeuralNetwork(
		layers,
		optimiser.NewSGD(0.1),
		loss.NewMSELoss(),
		regularisation.NewL2Regulariser(0.1))

	epochs := 100
	nn.Train(trainImages, trainLabels, epochs)

	modelFilename := "ThreeLayer_model_config.json"

	// Save the model
	if err := nn.SaveModel(modelFilename, "ThreeLayer", "MNIST"); err != nil {
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
