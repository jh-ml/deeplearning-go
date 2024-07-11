package network

import (
	"encoding/json"
	"github.com/google/uuid"
	m "github.com/jh-ml/deeplearning-go/model"
	"os"
	"time"
)

func (nn *NeuralNetwork) SaveModel(configPath string, name, datasetName string, totalLoss float64) error {
	model := m.Model{
		Metadata: m.Metadata{
			Name:         name,
			CreationDate: time.Now(),
			ID:           uuid.New().String(),
			TotalLoss:    totalLoss,
			DatasetName:  datasetName,
		},
	}

	// Iterate over the layers of the network
	for _, layer := range nn.GetLayers() {
		config, tensors := layer.Save()

		layerConfig := m.LayerConfig{
			LayerName: layer.Name(),
			Config:    config,
			Tensors:   tensors,
		}

		model.Layers = append(model.Layers, layerConfig)
	}

	// Serialize the optimizer
	if nn.optimiser != nil {
		model.Optimiser = nn.optimiser.Save()
	}

	// Serialize the loss function
	if nn.lossFunction != nil {
		model.LossFunction = nn.lossFunction.Save()
	}

	// Serialize the regularization
	if nn.regularisation != nil {
		model.Regularisation = nn.regularisation.Save()
	}

	// Save the model configuration to JSON
	configData, err := json.Marshal(model)
	if err != nil {
		return err
	}
	if err := os.WriteFile(configPath, configData, 0644); err != nil {
		return err
	}

	return nil
}
