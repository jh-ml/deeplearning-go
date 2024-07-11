package examples

import (
	"github.com/jh-ml/deeplearning-go/architectures"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

func main() {
	// Define generator layers
	generatorLayers := []layer.Interface{
		layer.NewFullyConnected(100, 256, activation.NewReLU()),
		layer.NewFullyConnected(256, 512, activation.NewReLU()),
		layer.NewFullyConnected(512, 1024, activation.NewReLU()),
		layer.NewFullyConnected(1024, 784, activation.NewSigmoid()),
	}

	// Define discriminator layers
	discriminatorLayers := []layer.Interface{
		layer.NewFullyConnected(784, 1024, activation.NewLeakyReLU(0.2)),
		layer.NewDropout(0.3),
		layer.NewFullyConnected(1024, 512, activation.NewLeakyReLU(0.2)),
		layer.NewDropout(0.3),
		layer.NewFullyConnected(512, 256, activation.NewLeakyReLU(0.2)),
		layer.NewDropout(0.3),
		layer.NewFullyConnected(256, 1, activation.NewSigmoid()),
	}

	// Define GAN configuration
	ganConfig := architectures.GANConfig{
		GeneratorLayers:     generatorLayers,
		DiscriminatorLayers: discriminatorLayers,
		LossFunction:        loss.NewBinaryCrossEntropy(),
		OptimiserG:          optimiser.NewAdam(0.0002, 0.5, 0.999, 1e-8),
		OptimiserD:          optimiser.NewAdam(0.0002, 0.5, 0.999, 1e-8),
		Regularisation:      regularisation.NewL2Regulariser(0.01),
	}

	// Create and train GAN
	gan := architectures.NewGAN(ganConfig)
	realData := tensor.NewRandomTensor([]int{64, 784}) // Example real data
	noise := tensor.NewRandomTensor([]int{64, 100})    // Random noise for generator

	// Train GAN
	gan.Train(1000, realData, noise)
}
