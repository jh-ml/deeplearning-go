package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
)

type MaxPooling struct {
	poolSize int              // Size of the pooling window
	stride   int              // Stride of the pooling window
	input    tensor.Interface // Cached input tensor for backward pass
	mask     tensor.Interface // Mask to record the position of max values for backward pass
}

// NewMaxPooling creates a new max pooling layer
func NewMaxPooling(poolSize, stride int) *MaxPooling {
	return &MaxPooling{
		poolSize: poolSize,
		stride:   stride,
	}
}

// Forward pass for MaxPooling
func (p *MaxPooling) Forward(input tensor.Interface) tensor.Interface {
	p.input = input
	inShape := input.Shape()
	batchSize, channels, inHeight, inWidth := inShape[0], inShape[1], inShape[2], inShape[3]
	outHeight := (inHeight-p.poolSize)/p.stride + 1
	outWidth := (inWidth-p.poolSize)/p.stride + 1
	output := tensor.NewZerosTensor([]int{batchSize, channels, outHeight, outWidth})
	p.mask = tensor.NewZerosTensor(inShape)

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < outHeight; i++ {
				for j := 0; j < outWidth; j++ {
					maxVal := -math.MaxFloat64
					maxIdx := []int{0, 0}

					for m := 0; m < p.poolSize; m++ {
						for n := 0; n < p.poolSize; n++ {
							h := i*p.stride + m
							w := j*p.stride + n
							if h < inHeight && w < inWidth {
								val := input.Get(b, c, h, w)
								if val > maxVal {
									maxVal = val
									maxIdx = []int{h, w}
								}
							}
						}
					}

					output.Set(maxVal, b, c, i, j)
					p.mask.Set(1, b, c, maxIdx[0], maxIdx[1])
				}
			}
		}
	}

	return output
}

// Backward pass for MaxPooling
func (p *MaxPooling) Backward(grad tensor.Interface) tensor.Interface {
	inShape := p.input.Shape()
	batchSize, channels, inHeight, inWidth := inShape[0], inShape[1], inShape[2], inShape[3]
	outHeight, outWidth := grad.Shape()[2], grad.Shape()[3]
	gradInput := tensor.NewZerosTensor(inShape)

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < outHeight; i++ {
				for j := 0; j < outWidth; j++ {
					for m := 0; m < p.poolSize; m++ {
						for n := 0; n < p.poolSize; n++ {
							h := i*p.stride + m
							w := j*p.stride + n
							if h < inHeight && w < inWidth && p.mask.Get(b, c, h, w) == 1 {
								gradInput.Set(grad.Get(b, c, i, j), b, c, h, w)
							}
						}
					}
				}
			}
		}
	}

	return gradInput
}

// GetWeights returns the weights of the MaxPooling layer (not applicable)
func (p *MaxPooling) GetWeights() tensor.Interface {
	return nil
}

// SetWeights sets the weights of the MaxPooling layer (not applicable)
func (p *MaxPooling) SetWeights(weights tensor.Interface) {}

// GetBiases returns the biases of the MaxPooling layer (not applicable)
func (p *MaxPooling) GetBiases() tensor.Interface {
	return nil
}

// SetBiases sets the biases of the MaxPooling layer (not applicable)
func (p *MaxPooling) SetBiases(biases tensor.Interface) {}

// GetGradients returns the gradients of the MaxPooling layer (not applicable)
func (p *MaxPooling) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return nil, nil
}

// RequiresOptimisation indicates if this layer requires optimisation
func (p *MaxPooling) RequiresOptimisation() bool {
	return false
}

// RequiresRegularisation indicates if this layer requires regularisation
func (p *MaxPooling) RequiresRegularisation() bool {
	return false
}

func (mp *MaxPooling) Name() string {
	return "MaxPooling"
}

func (mp *MaxPooling) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"pool_size": mp.poolSize,
		"stride":    mp.stride,
	}
	return config, nil
}

func (mp *MaxPooling) Load(config map[string]any, tensors []model.TensorData) error {
	poolSize, ok := config["pool_size"].(float64)
	if !ok {
		return errors.New("invalid pool_size")
	}
	mp.poolSize = int(poolSize)

	stride, ok := config["stride"].(float64)
	if !ok {
		return errors.New("invalid stride")
	}
	mp.stride = int(stride)

	return nil
}
