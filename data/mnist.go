package data

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"github.com/petar/GoMNIST"
	"log"
)

func LoadTrainingData(path string) ([]tensor.Interface, []tensor.Interface) {
	train, _, err := GoMNIST.Load(path)
	if err != nil {
		log.Fatalf("Error loading MNIST training data: %v", err)
	}

	trainImages := make([]tensor.Interface, len(train.Images))
	trainLabels := make([]tensor.Interface, len(train.Labels))
	for i := range train.Images {
		img, label := train.Images[i], train.Labels[i]
		trainImages[i] = tensor.NewTensor(flatten(img), []int{1, 784})
		labelTensor := make([]float64, 10)
		labelTensor[label] = 1
		trainLabels[i] = tensor.NewTensor(labelTensor, []int{1, 10})
	}

	return trainImages, trainLabels
}

func LoadTestData(path string) ([]tensor.Interface, []tensor.Interface) {
	_, test, err := GoMNIST.Load(path)
	if err != nil {
		log.Fatalf("Error loading MNIST test data: %v", err)
	}

	testImages := make([]tensor.Interface, len(test.Images))
	testLabels := make([]tensor.Interface, len(test.Labels))
	for i := range test.Images {
		img, label := test.Images[i], test.Labels[i]
		testImages[i] = tensor.NewTensor(flatten(img), []int{1, 784})
		labelTensor := make([]float64, 10)
		labelTensor[label] = 1
		testLabels[i] = tensor.NewTensor(labelTensor, []int{1, 10})
	}

	return testImages, testLabels
}

func flatten(img GoMNIST.RawImage) []float64 {
	flat := make([]float64, len(img))
	for i := range img {
		flat[i] = float64(img[i]) / 255.0
	}
	return flat
}

func ReshapeData(data []tensor.Interface, newShape []int) []tensor.Interface {
	reshapedData := make([]tensor.Interface, len(data))
	for i, d := range data {
		reshapedData[i] = d.Reshape(newShape)
	}
	return reshapedData
}
