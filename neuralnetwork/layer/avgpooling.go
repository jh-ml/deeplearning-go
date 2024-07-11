package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

// AveragePooling represents an average pooling layer
type AveragePooling struct {
	poolSize int
	stride   int
	input    tensor.Interface
}

// NewAveragePooling creates a new average pooling layer
func NewAveragePooling(poolSize, stride int) *AveragePooling {
	return &AveragePooling{
		poolSize: poolSize,
		stride:   stride,
	}
}

// Forward pass for AveragePooling
func (p *AveragePooling) Forward(input tensor.Interface) tensor.Interface {
	p.input = input
	inputShape := input.Shape()
	batchSize, channels, height, width := inputShape[0], inputShape[1], inputShape[2], inputShape[3]
	outputHeight := (height-p.poolSize)/p.stride + 1
	outputWidth := (width-p.poolSize)/p.stride + 1
	output := tensor.NewZerosTensor([]int{batchSize, channels, outputHeight, outputWidth})

	for n := 0; n < batchSize; n++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					sum := 0.0
					count := 0

					for m := 0; m < p.poolSize; m++ {
						for k := 0; k < p.poolSize; k++ {
							h := i*p.stride + m
							w := j*p.stride + k
							if h < height && w < width {
								sum += input.Get(n, c, h, w)
								count++
							}
						}
					}
					output.Set(sum/float64(count), n, c, i, j)
				}
			}
		}
	}

	return output
}

// Backward pass for AveragePooling
func (p *AveragePooling) Backward(grad tensor.Interface) tensor.Interface {
	inputShape := p.input.Shape()
	batchSize, channels, height, width := inputShape[0], inputShape[1], inputShape[2], inputShape[3]
	outputHeight, outputWidth := grad.Shape()[2], grad.Shape()[3]
	gradInput := tensor.NewZerosTensor(inputShape)

	for n := 0; n < batchSize; n++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					count := 0
					for m := 0; m < p.poolSize; m++ {
						for k := 0; k < p.poolSize; k++ {
							h := i*p.stride + m
							w := j*p.stride + k
							if h < height && w < width {
								count++
							}
						}
					}

					gradVal := grad.Get(n, c, i, j) / float64(count)

					for m := 0; m < p.poolSize; m++ {
						for k := 0; k < p.poolSize; k++ {
							h := i*p.stride + m
							w := j*p.stride + k
							if h < height && w < width {
								gradInput.Set(gradInput.Get(n, c, h, w)+gradVal, n, c, h, w)
							}
						}
					}
				}
			}
		}
	}

	return gradInput
}

// GetWeights returns the weights of the AveragePooling layer
func (p *AveragePooling) GetWeights() tensor.Interface {
	return nil
}

// SetWeights sets the weights of the AveragePooling layer
func (p *AveragePooling) SetWeights(weights tensor.Interface) {}

// GetBiases returns the biases of the AveragePooling layer
func (p *AveragePooling) GetBiases() tensor.Interface {
	return nil
}

// SetBiases sets the biases of the AveragePooling layer
func (p *AveragePooling) SetBiases(biases tensor.Interface) {}

// GetGradients returns the gradients of the AveragePooling layer
func (p *AveragePooling) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return nil, nil
}

// RequiresOptimisation indicates if this layer requires optimization
func (p *AveragePooling) RequiresOptimisation() bool {
	return false
}

// RequiresRegularisation indicates if this layer requires regularization
func (p *AveragePooling) RequiresRegularisation() bool {
	return false
}

func (p *AveragePooling) Name() string {
	return "AveragePooling"
}

func (p *AveragePooling) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"pool_size": p.poolSize,
		"stride":    p.stride,
	}
	return config, nil
}

func (ap *AveragePooling) Load(config map[string]any, tensors []model.TensorData) error {
	poolSize, ok := config["pool_size"].(float64)
	if !ok {
		return errors.New("invalid pool_size")
	}
	ap.poolSize = int(poolSize)

	stride, ok := config["stride"].(float64)
	if !ok {
		return errors.New("invalid stride")
	}
	ap.stride = int(stride)

	return nil
}
