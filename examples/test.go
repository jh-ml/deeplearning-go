package examples

import (
	"fmt"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/network"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

func test(nn *network.NeuralNetwork, testImages []tensor.Interface, testLabels []tensor.Interface) {
	correct := 0
	for i := 0; i < len(testImages); i++ {
		output := nn.Predict(testImages[i])
		if argmax(output.Data()) == argmax(testLabels[i].Data()) {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testImages))
	fmt.Printf("Test Accuracy: %f\n", accuracy)
}

func argmax(data []float64) int {
	maxIdx := 0
	maxValue := data[0]
	for i, v := range data {
		if v > maxValue {
			maxIdx = i
			maxValue = v
		}
	}
	return maxIdx
}
