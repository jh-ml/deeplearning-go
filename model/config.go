package model

import "time"

type Metadata struct {
	Name         string    `json:"name"`
	CreationDate time.Time `json:"creation_date"`
	ID           string    `json:"id"`
	TotalLoss    float64   `json:"total_loss"`
	DatasetName  string    `json:"dataset_name"`
}

// Model represents the entire model configuration and tensors
type Model struct {
	Metadata       Metadata       `json:"metadata"`
	Layers         []LayerConfig  `json:"layers"`
	Optimiser      map[string]any `json:"optimiser"`
	LossFunction   map[string]any `json:"lossFunction"`
	Regularisation map[string]any `json:"regularisation"`
}

// LayerConfig represents the configuration of a layer
type LayerConfig struct {
	LayerName string         `json:"layerName"`
	Config    map[string]any `json:"config"`
	Tensors   []TensorData   `json:"tensors"`
}

// TensorData represents the serialized form of a tensor
type TensorData struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}
