package architectures

import (
	"github.com/jh-ml/deeplearning-go/neuralnetwork/layer"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/loss"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/network"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/optimiser"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/regularisation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type GANConfig struct {
	GeneratorLayers     []layer.Interface
	DiscriminatorLayers []layer.Interface
	LossFunction        loss.Interface
	OptimiserG          optimiser.Interface
	OptimiserD          optimiser.Interface
	Regularisation      regularisation.Interface
}

type GAN struct {
	generator     network.Interface
	discriminator network.Interface
}

// NewGAN creates a new GAN with the provided configuration
func NewGAN(config GANConfig) *GAN {
	generator := network.NewNeuralNetwork(config.GeneratorLayers, config.OptimiserG, config.LossFunction, config.Regularisation)
	discriminator := network.NewNeuralNetwork(config.DiscriminatorLayers, config.OptimiserD, config.LossFunction, config.Regularisation)

	return &GAN{
		generator:     generator,
		discriminator: discriminator,
	}
}

// Train trains the GAN with the provided data
func (gan *GAN) Train(epochs int, realData tensor.Interface, noise tensor.Interface) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Generate fake data
		fakeData := gan.generator.Predict(noise)

		// Train discriminator on real data
		realLabels := tensor.NewOnesTensor([]int{realData.Shape()[0], 1})
		fakeLabels := tensor.NewZerosTensor([]int{fakeData.Shape()[0], 1})

		// Train discriminator on both real and fake data
		gan.trainDiscriminator(realData, realLabels, fakeData, fakeLabels, epoch)

		// Train generator
		gan.trainGenerator(noise, realLabels, epoch)
	}
}

func (gan *GAN) trainDiscriminator(realData, realLabels, fakeData, fakeLabels tensor.Interface, epoch int) {
	// Prepare data for discriminator training
	combinedData := tensor.Concatenate([]tensor.Interface{realData, fakeData})
	combinedLabels := tensor.Concatenate([]tensor.Interface{realLabels, fakeLabels})

	gan.discriminator.Train([]tensor.Interface{combinedData}, []tensor.Interface{combinedLabels}, epoch)
}

func (gan *GAN) trainGenerator(noise, realLabels tensor.Interface, epoch int) {
	// Train the generator to fool the discriminator
	gan.generator.Train([]tensor.Interface{noise}, []tensor.Interface{realLabels}, epoch)
}
