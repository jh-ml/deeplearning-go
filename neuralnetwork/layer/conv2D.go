package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type Conv2D struct {
	Weights    tensor.Interface     // The weights of the convolutional layer
	Biases     tensor.Interface     // The biases of the convolutional layer
	dWeights   tensor.Interface     // The gradients of the weights
	dBiases    tensor.Interface     // The gradients of the biases
	Stride     int                  // The stride of the convolution operation
	Padding    int                  // The padding added to the input
	InputDim   int                  // The number of input channels
	input      tensor.Interface     // Cached input tensor for backward pass
	Activation activation.Interface // Activation function applied after convolution
}

// NewConv2D creates a new convolutional layer
func NewConv2D(inputDim, outputDim, kernelSize, stride, padding int, activation activation.Interface) *Conv2D {
	weightsShape := []int{outputDim, inputDim, kernelSize, kernelSize}
	biasesShape := []int{outputDim}
	return &Conv2D{
		Weights:    tensor.NewRandomTensor(weightsShape),
		Biases:     tensor.NewZerosTensor(biasesShape),
		Stride:     stride,
		Padding:    padding,
		InputDim:   inputDim,
		Activation: activation,
	}
}

// Forward pass for Convolutional layer
func (conv *Conv2D) Forward(input tensor.Interface) tensor.Interface {
	conv.input = input
	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic("Input dimension mismatch: expected 4D tensor")
	}

	output := input.Conv2D(conv.Weights, conv.Stride, conv.Padding)
	batchSize := inputShape[0]
	numOutputChannels := conv.Weights.Shape()[0]
	outputHeight := output.Shape()[2]
	outputWidth := output.Shape()[3]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for outputChannel := 0; outputChannel < numOutputChannels; outputChannel++ {
			for outputHeightIndex := 0; outputHeightIndex < outputHeight; outputHeightIndex++ {
				for outputWidthIndex := 0; outputWidthIndex < outputWidth; outputWidthIndex++ {
					currentOutputValue := output.Get(batchIndex, outputChannel, outputHeightIndex, outputWidthIndex)
					biasValue := conv.Biases.Get(outputChannel)
					output.Set(currentOutputValue+biasValue, batchIndex, outputChannel, outputHeightIndex, outputWidthIndex)
				}
			}
		}
	}
	return conv.Activation.Forward(output)
}

// Backward pass for Convolutional layer
func (conv *Conv2D) Backward(grad tensor.Interface) tensor.Interface {
	grad = conv.Activation.Backward(grad)
	inputShape := conv.input.Shape()
	batchSize := inputShape[0]
	inputChannels := inputShape[1]
	inputHeight := inputShape[2]
	inputWidth := inputShape[3]

	kernelHeight := conv.Weights.Shape()[2]
	kernelWidth := conv.Weights.Shape()[3]
	outputChannels := conv.Weights.Shape()[0]
	outputHeight := (inputHeight+2*conv.Padding-kernelHeight)/conv.Stride + 1
	outputWidth := (inputWidth+2*conv.Padding-kernelWidth)/conv.Stride + 1

	dInput := tensor.NewZerosTensor([]int{batchSize, inputChannels, inputHeight, inputWidth})
	conv.dWeights = tensor.NewZerosTensor(conv.Weights.Shape())
	conv.dBiases = tensor.NewZerosTensor(conv.Biases.Shape())

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for outputChannel := 0; outputChannel < outputChannels; outputChannel++ {
			for outputHeightIndex := 0; outputHeightIndex < outputHeight; outputHeightIndex++ {
				for outputWidthIndex := 0; outputWidthIndex < outputWidth; outputWidthIndex++ {
					gradientValue := grad.Get(batchIndex, outputChannel, outputHeightIndex, outputWidthIndex)
					for inputChannel := 0; inputChannel < inputChannels; inputChannel++ {
						for kernelHeightIndex := 0; kernelHeightIndex < kernelHeight; kernelHeightIndex++ {
							for kernelWidthIndex := 0; kernelWidthIndex < kernelWidth; kernelWidthIndex++ {
								inputHeightIndex := outputHeightIndex*conv.Stride + kernelHeightIndex - conv.Padding
								inputWidthIndex := outputWidthIndex*conv.Stride + kernelWidthIndex - conv.Padding
								if inputHeightIndex >= 0 && inputHeightIndex < inputHeight && inputWidthIndex >= 0 && inputWidthIndex < inputWidth {
									inputValue := conv.input.Get(batchIndex, inputChannel, inputHeightIndex, inputWidthIndex)
									weightGradient := conv.dWeights.Get(outputChannel, inputChannel, kernelHeightIndex, kernelWidthIndex)
									weightUpdate := gradientValue * inputValue
									conv.dWeights.Set(weightGradient+weightUpdate, outputChannel, inputChannel, kernelHeightIndex, kernelWidthIndex)

									dInputValue := dInput.Get(batchIndex, inputChannel, inputHeightIndex, inputWidthIndex)
									weightValue := conv.Weights.Get(outputChannel, inputChannel, kernelHeightIndex, kernelWidthIndex)
									dInput.Set(dInputValue+gradientValue*weightValue, batchIndex, inputChannel, inputHeightIndex, inputWidthIndex)
								}
							}
						}
					}
					biasGradient := conv.dBiases.Get(outputChannel)
					conv.dBiases.Set(biasGradient+gradientValue, outputChannel)
				}
			}
		}
	}

	return dInput
}

// GetWeights returns the weights of the Convolutional layer
func (conv *Conv2D) GetWeights() tensor.Interface {
	return conv.Weights
}

// SetWeights sets the weights of the Convolutional layer
func (conv *Conv2D) SetWeights(weights tensor.Interface) {
	conv.Weights = weights
}

// GetBiases returns the biases of the Convolutional layer
func (conv *Conv2D) GetBiases() tensor.Interface {
	return conv.Biases
}

// SetBiases sets the biases of the Convolutional layer
func (conv *Conv2D) SetBiases(biases tensor.Interface) {
	conv.Biases = biases
}

// GetGradients returns the gradients of the Convolutional layer
func (conv *Conv2D) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	return conv.dWeights, conv.dBiases
}

// RequiresOptimisation enables optimisation
func (conv *Conv2D) RequiresOptimisation() bool {
	return true
}

// RequiresRegularisation indicates if this layer requires regularisation
func (conv *Conv2D) RequiresRegularisation() bool {
	return true
}

func (conv *Conv2D) Name() string {
	return "Conv2D"
}

func (c *Conv2D) Save() (map[string]interface{}, []model.TensorData) {
	config := map[string]interface{}{
		"input_dim":   c.InputDim,
		"output_dim":  c.Weights.Shape()[0],
		"kernel_size": c.Weights.Shape()[2],
		"stride":      c.Stride,
		"padding":     c.Padding,
		"activation":  c.Activation.Name(),
	}

	tensors := []model.TensorData{
		{
			Name:  "Weights",
			Shape: c.Weights.Shape(),
			Data:  c.Weights.Data(),
		},
		{
			Name:  "Biases",
			Shape: c.Biases.Shape(),
			Data:  c.Biases.Data(),
		},
	}

	return config, tensors
}

func (c *Conv2D) Load(config map[string]interface{}, tensors []model.TensorData) error {
	// Ensure proper type assertions and conversions
	inputDim, ok := config["input_dim"].(int)
	if !ok {
		return errors.New("invalid input_dim")
	}
	c.InputDim = int(inputDim)

	stride, ok := config["stride"].(int)
	if !ok {
		return errors.New("invalid stride")
	}
	c.Stride = int(stride)

	padding, ok := config["padding"].(int)
	if !ok {
		return errors.New("invalid padding")
	}
	c.Padding = int(padding)

	activationName, ok := config["activation"].(string)
	if !ok {
		return errors.New("invalid activation")
	}
	activation, err := activation.NewActivationByName(activationName)
	if err != nil {
		return err
	}
	c.Activation = activation

	for _, tensorData := range tensors {
		switch tensorData.Name {
		case "Weights":
			c.Weights = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Biases":
			c.Biases = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		default:
			return errors.New("unexpected tensor name: " + tensorData.Name)
		}
	}

	return nil
}
