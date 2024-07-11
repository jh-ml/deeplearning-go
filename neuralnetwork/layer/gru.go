package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

type GRU struct {
	Wu, Wr, Wh tensor.Interface // Weights for update gate, reset gate, and hidden state
	Ru, Rr, Rh tensor.Interface // Recurrent weights for update gate, reset gate, and hidden state
	Bu, Br, Bh tensor.Interface // Biases for update gate, reset gate, and hidden state
	H          tensor.Interface // Hidden state

	// Gradients
	dWu, dWr, dWh tensor.Interface // Gradients of weights for update gate, reset gate, and hidden state
	dRu, dRr, dRh tensor.Interface // Gradients of recurrent weights for update gate, reset gate, and hidden state
	dBu, dBr, dBh tensor.Interface // Gradients of biases for update gate, reset gate, and hidden state

	// Cache for backward pass
	zt, rt, hHat  tensor.Interface     // Cached values of gates and candidate hidden state
	sigmoid, tanh activation.Interface // Activation functions
}

// NewGRU creates a new GRU layer
func NewGRU(inputSize, hiddenSize int) *GRU {
	return &GRU{
		Wu:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Wr:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Wh:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Ru:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Rr:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Rh:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Bu:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		Br:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		Bh:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		H:       tensor.NewZerosTensor([]int{inputSize, hiddenSize}),
		sigmoid: activation.NewSigmoid(),
		tanh:    activation.NewTanh(),
	}
}

// Forward pass for GRU
func (g *GRU) Forward(input tensor.Interface) tensor.Interface {
	// Compute gates
	g.zt = g.sigmoid.Forward(input.Dot(g.Wu).Add(g.H.Dot(g.Ru)).Add(g.Bu))
	g.rt = g.sigmoid.Forward(input.Dot(g.Wr).Add(g.H.Dot(g.Rr)).Add(g.Br))
	g.hHat = g.tanh.Forward(input.Dot(g.Wh).Add(g.rt.Multiply(g.H).Dot(g.Rh)).Add(g.Bh))

	g.H = g.zt.Multiply(g.H).Add(g.zt.MultiplyScalar(-1).AddScalar(1).Multiply(g.hHat))

	return g.H
}

// Backward pass for GRU
func (g *GRU) Backward(dh tensor.Interface) tensor.Interface {
	// Derivative of the loss with respect to the hidden state
	dhHat := dh.Multiply(g.zt.MultiplyScalar(-1).AddScalar(1)).Multiply(g.tanh.Backward(g.hHat))
	dzt := dh.Multiply(g.H.Subtract(g.hHat)).Multiply(g.sigmoid.Backward(g.zt))

	// Compute gradients for weights and biases
	g.dWh = g.H.Transpose().Dot(dhHat)
	g.dRh = g.rt.Multiply(g.H).Transpose().Dot(dhHat)
	g.dBh = dhHat.SumAlongBatch()

	g.dWu = g.H.Transpose().Dot(dzt)
	g.dRu = g.H.Transpose().Dot(dzt)
	g.dBu = dzt.SumAlongBatch()

	drt := dhHat.Dot(g.Rh.Transpose()).Multiply(g.H).Multiply(g.sigmoid.Backward(g.rt))
	g.dWr = g.H.Transpose().Dot(drt)
	g.dRr = g.H.Transpose().Dot(drt)
	g.dBr = drt.SumAlongBatch()

	// Compute gradient with respect to the input
	dInput := dhHat.Dot(g.Wh.Transpose()).Add(dzt.Dot(g.Wu.Transpose())).Add(drt.Dot(g.Wr.Transpose()))

	return dInput
}

// GetWeights returns the weights of the GRU layer
func (g *GRU) GetWeights() tensor.Interface {
	weights := tensor.Concatenate([]tensor.Interface{g.Wu, g.Wr, g.Wh, g.Ru, g.Rr, g.Rh})
	return weights
}

// SetWeights sets the weights of the GRU layer
func (g *GRU) SetWeights(weights tensor.Interface) {
	w := weights.Split([]int{g.Wu.Size(), g.Wr.Size(), g.Wh.Size(), g.Ru.Size(), g.Rr.Size(), g.Rh.Size()})
	g.Wu, g.Wr, g.Wh, g.Ru, g.Rr, g.Rh = w[0], w[1], w[2], w[3], w[4], w[5]
}

// GetBiases returns the biases of the GRU layer
func (g *GRU) GetBiases() tensor.Interface {
	biases := tensor.Concatenate([]tensor.Interface{g.Bu, g.Br, g.Bh})
	return biases
}

// SetBiases sets the biases of the GRU layer
func (g *GRU) SetBiases(biases tensor.Interface) {
	b := biases.Split([]int{g.Bu.Size(), g.Br.Size(), g.Bh.Size()})
	g.Bu, g.Br, g.Bh = b[0], b[1], b[2]
}

// GetGradients returns the gradients of the GRU layer
func (g *GRU) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	weightsGrad = tensor.Concatenate([]tensor.Interface{g.dWu, g.dWr, g.dWh, g.dRu, g.dRr, g.dRh})
	biasesGrad = tensor.Concatenate([]tensor.Interface{g.dBu, g.dBr, g.dBh})
	return weightsGrad, biasesGrad
}

// RequiresOptimisation indicates if this layer requires optimisation
func (g *GRU) RequiresOptimisation() bool {
	return true
}

// RequiresRegularisation indicates if this layer requires regularisation
func (g *GRU) RequiresRegularisation() bool {
	return true
}

func (l *GRU) Name() string {
	return "GRU"
}

func (g *GRU) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"activation_sigmoid": g.sigmoid.Name(),
		"activation_tanh":    g.tanh.Name(),
	}

	tensors := []model.TensorData{
		{Name: "Wu", Shape: g.Wu.Shape(), Data: g.Wu.Data()},
		{Name: "Wr", Shape: g.Wr.Shape(), Data: g.Wr.Data()},
		{Name: "Wh", Shape: g.Wh.Shape(), Data: g.Wh.Data()},
		{Name: "Ru", Shape: g.Ru.Shape(), Data: g.Ru.Data()},
		{Name: "Rr", Shape: g.Rr.Shape(), Data: g.Rr.Data()},
		{Name: "Rh", Shape: g.Rh.Shape(), Data: g.Rh.Data()},
		{Name: "Bu", Shape: g.Bu.Shape(), Data: g.Bu.Data()},
		{Name: "Br", Shape: g.Br.Shape(), Data: g.Br.Data()},
		{Name: "Bh", Shape: g.Bh.Shape(), Data: g.Bh.Data()},
	}

	return config, tensors
}

func (g *GRU) Load(config map[string]any, tensors []model.TensorData) error {
	sigmoidName, ok := config["activation_sigmoid"].(string)
	if !ok {
		return errors.New("invalid activation_sigmoid")
	}
	sigmoid, err := activation.NewActivationByName(sigmoidName)
	if err != nil {
		return err
	}
	g.sigmoid = sigmoid

	tanhName, ok := config["activation_tanh"].(string)
	if !ok {
		return errors.New("invalid activation_tanh")
	}
	tanh, err := activation.NewActivationByName(tanhName)
	if err != nil {
		return err
	}
	g.tanh = tanh

	for _, tensorData := range tensors {
		switch tensorData.Name {
		case "Wu":
			g.Wu = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Wr":
			g.Wr = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Wh":
			g.Wh = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Ru":
			g.Ru = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Rr":
			g.Rr = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Rh":
			g.Rh = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bu":
			g.Bu = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Br":
			g.Br = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bh":
			g.Bh = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		default:
			return errors.New("unexpected tensor name: " + tensorData.Name)
		}
	}

	return nil
}
