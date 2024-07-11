package network

import (
	"encoding/json"
	"errors"
	m "github.com/jh-ml/deeplearning-go/model"
	l "github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
	"os"
)

func LoadModel(configPath string) (*NeuralNetwork, error) {
	// Read the model configuration from JSON
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var model m.Model
	if err := json.Unmarshal(configData, &model); err != nil {
		return nil, err
	}

	// Reconstruct the network layers
	var layers []l.Interface
	for _, layerConfig := range model.Layers {
		var layer l.Interface
		switch layerConfig.LayerName {
		case "Conv2D":
			layer = &l.Conv2D{}
		case "FullyConnected":
			layer = &l.FullyConnected{}
		case "Flatten":
			layer = &l.Flatten{}
		case "MaxPooling":
			layer = &l.MaxPooling{}
		case "AvgPooling":
			layer = &l.AveragePooling{}
		case "Dropout":
			layer = &l.Dropout{}
		case "Embedding":
			layer = &l.Embedding{}
		case "GRU":
			layer = &l.GRU{}
		case "LSTM":
			layer = &l.LSTM{}
		case "Reshape":
			layer = &l.Reshape{}
		// Add cases for other layer types here
		default:
			return nil, errors.New("unknown layer type: " + layerConfig.LayerName)
		}

		if err := layer.Load(layerConfig.Config, layerConfig.Tensors); err != nil {
			return nil, err
		}

		layers = append(layers, layer)
	}

	// Reconstruct the optimizer
	var opt optimiser.Interface
	if model.Optimiser != nil {
		switch model.Optimiser["type"] {
		case "SGD":
			opt = optimiser.NewSGD(model.Optimiser["learning_rate"].(float64))
		case "SGDWithMomentum":
			opt = optimiser.NewSGDWithMomentum(
				model.Optimiser["learning_rate"].(float64),
				model.Optimiser["momentum"].(float64),
			)
		case "Adam":
			opt = optimiser.NewAdam(
				model.Optimiser["learning_rate"].(float64),
				model.Optimiser["beta1"].(float64),
				model.Optimiser["beta2"].(float64),
				model.Optimiser["epsilon"].(float64),
			)
		case "RMSProp":
			opt = optimiser.NewRMSProp(
				model.Optimiser["learning_rate"].(float64),
				model.Optimiser["beta"].(float64),
				model.Optimiser["epsilon"].(float64),
			)
		default:
			return nil, errors.New("unknown optimiser type: " + model.Optimiser["type"].(string))
		}
	}

	// Reconstruct the loss function
	var lossFunc loss.Interface
	if model.LossFunction != nil {
		switch model.LossFunction["type"] {
		case "BinaryCrossEntropy":
			lossFunc = loss.NewBinaryCrossEntropy()
		case "CategoricalCrossEntropy":
			lossFunc = loss.NewCategoricalCrossEntropy()
		case "CosineProximityLoss":
			lossFunc = loss.NewCosineProximityLoss()
		case "MeanSquaredError":
			lossFunc = loss.NewMSELoss()
		default:
			return nil, errors.New("unknown loss function type: " + model.LossFunction["type"].(string))
		}
	}

	// Reconstruct the regularisation
	var reg regularisation.Interface
	if model.Regularisation != nil {
		switch model.Regularisation["type"] {
		case "L2":
			reg = regularisation.NewL2Regulariser(model.Regularisation["lambda"].(float64))
		case "L1":
			reg = regularisation.NewL1Regulariser(model.Regularisation["lambda"].(float64))
		case "ElasticNet":
			reg = regularisation.NewElasticNetRegulariser(
				model.Regularisation["lambda1"].(float64),
				model.Regularisation["lambda2"].(float64),
			)
		default:
			return nil, errors.New("unknown regularisation type: " + model.Regularisation["type"].(string))
		}
	}

	return &NeuralNetwork{
		layers:         layers,
		optimiser:      opt,
		lossFunction:   lossFunc,
		regularisation: reg,
	}, nil
}
