package network

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Interface interface {
	AddLayer(layer layer.Interface)
	GetLayers() []layer.Interface
	Forward(input tensor.Interface) tensor.Interface
	Backward(grad tensor.Interface) tensor.Interface
	Regularise()
	ZeroGradients()
	Optimise()
	Train(data, targets []tensor.Interface, epochs int) float64
	Predict(input tensor.Interface) tensor.Interface
	SaveModel(configPath string, name, datasetName string, totalLoss float64) error
}
