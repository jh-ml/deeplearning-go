package layer

import (
	"errors"
	"github.com/jh-ml/deeplearning-go/model"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/activation"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
)

// LSTM represents an LSTM layer
type LSTM struct {
	Wf, Wi, Wc, Wo tensor.Interface
	Uf, Ui, Uc, Uo tensor.Interface
	Bf, Bi, Bc, Bo tensor.Interface
	H, C           tensor.Interface

	// Gradients
	dWf, dWi, dWc, dWo tensor.Interface
	dUf, dUi, dUc, dUo tensor.Interface
	dBf, dBi, dBc, dBo tensor.Interface

	// Cache for backward pass
	ft, it, gt, ot tensor.Interface
	sigmoid, tanh  activation.Interface
}

// NewLSTM creates a new LSTM layer
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	return &LSTM{
		Wf:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Wi:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Wc:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Wo:      tensor.NewRandomTensor([]int{inputSize, hiddenSize}),
		Uf:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Ui:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Uc:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Uo:      tensor.NewRandomTensor([]int{hiddenSize, hiddenSize}),
		Bf:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		Bi:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		Bc:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		Bo:      tensor.NewZerosTensor([]int{1, hiddenSize}),
		H:       tensor.NewZerosTensor([]int{1, hiddenSize}),
		C:       tensor.NewZerosTensor([]int{1, hiddenSize}),
		sigmoid: activation.NewSigmoid(),
		tanh:    activation.NewTanh(),
	}
}

// Forward pass for LSTM
func (l *LSTM) Forward(input tensor.Interface) tensor.Interface {
	// Compute gates
	l.ft = l.sigmoid.Forward(input.Dot(l.Wf).Add(l.H.Dot(l.Uf)).Add(l.Bf))
	l.it = l.sigmoid.Forward(input.Dot(l.Wi).Add(l.H.Dot(l.Ui)).Add(l.Bi))
	l.gt = l.tanh.Forward(input.Dot(l.Wc).Add(l.H.Dot(l.Uc)).Add(l.Bc))
	l.ot = l.sigmoid.Forward(input.Dot(l.Wo).Add(l.H.Dot(l.Uo)).Add(l.Bo))

	// Update cell state and hidden state
	l.C = l.ft.Multiply(l.C).Add(l.it.Multiply(l.gt))
	l.H = l.ot.Multiply(l.tanh.Forward(l.C))

	return l.H
}

// Backward pass for LSTM
func (l *LSTM) Backward(dh tensor.Interface) tensor.Interface {
	// Derivative of the loss with respect to the output gate
	do := dh.Multiply(l.tanh.Forward(l.C)).Multiply(l.sigmoid.Backward(l.ot))

	// Derivative of the loss with respect to the cell state
	dC := dh.Multiply(l.ot).Multiply(l.tanh.Backward(l.C)).Add(l.C.Multiply(l.ft))

	// Derivative of the loss with respect to the candidate memory cell
	dg := dC.Multiply(l.it).Multiply(l.tanh.Backward(l.gt))

	// Derivative of the loss with respect to the input gate
	di := dC.Multiply(l.gt).Multiply(l.sigmoid.Backward(l.it))

	// Derivative of the loss with respect to the forget gate
	df := dC.Multiply(l.C).Multiply(l.sigmoid.Backward(l.ft))

	// Compute gradients for weights and biases
	l.dWo = l.H.Transpose().Dot(do)
	l.dBo = do.SumAlongBatch()
	l.dWc = l.H.Transpose().Dot(dg)
	l.dBc = dg.SumAlongBatch()
	l.dWi = l.H.Transpose().Dot(di)
	l.dBi = di.SumAlongBatch()
	l.dWf = l.H.Transpose().Dot(df)
	l.dBf = df.SumAlongBatch()

	// Compute gradient with respect to the input
	dInput := do.Dot(l.Wo.Transpose()).Add(dg.Dot(l.Wc.Transpose())).Add(di.Dot(l.Wi.Transpose())).Add(df.Dot(l.Wf.Transpose()))
	return dInput
}

// GetWeights returns the weights of the LSTM layer
func (l *LSTM) GetWeights() tensor.Interface {
	weights := []tensor.Interface{l.Wf, l.Wi, l.Wc, l.Wo}
	return tensor.Concatenate(weights)
}

// SetWeights sets the weights of the LSTM layer
func (l *LSTM) SetWeights(weights tensor.Interface) {
	w := weights.Split([]int{l.Wf.Size(), l.Wi.Size(), l.Wc.Size(), l.Wo.Size()})
	l.Wf, l.Wi, l.Wc, l.Wo = w[0], w[1], w[2], w[3]
}

// GetBiases returns the biases of the LSTM layer
func (l *LSTM) GetBiases() tensor.Interface {
	biases := []tensor.Interface{l.Bf, l.Bi, l.Bc, l.Bo}
	return tensor.Concatenate(biases)
}

// SetBiases sets the biases of the LSTM layer
func (l *LSTM) SetBiases(biases tensor.Interface) {
	b := biases.Split([]int{l.Bf.Size(), l.Bi.Size(), l.Bc.Size(), l.Bo.Size()})
	l.Bf, l.Bi, l.Bc, l.Bo = b[0], b[1], b[2], b[3]
}

// GetGradients returns the gradients of the LSTM layer
func (l *LSTM) GetGradients() (weightsGrad tensor.Interface, biasesGrad tensor.Interface) {
	weightsGrad = tensor.Concatenate([]tensor.Interface{l.dWf, l.dWi, l.dWc, l.dWo})
	biasesGrad = tensor.Concatenate([]tensor.Interface{l.dBf, l.dBi, l.dBc, l.dBo})
	return weightsGrad, biasesGrad
}

// RequiresOptimisation indicates if this layer requires optimisation
func (l *LSTM) RequiresOptimisation() bool {
	return true
}

// RequiresRegularisation indicates if this layer requires regularisation
func (l *LSTM) RequiresRegularisation() bool {
	return true
}

func (l *LSTM) Name() string {
	return "LSTM"
}

func (l *LSTM) Save() (map[string]any, []model.TensorData) {
	config := map[string]any{
		"activation_sigmoid": l.sigmoid.Name(),
		"activation_tanh":    l.tanh.Name(),
	}

	tensors := []model.TensorData{
		{Name: "Wf", Shape: l.Wf.Shape(), Data: l.Wf.Data()},
		{Name: "Wi", Shape: l.Wi.Shape(), Data: l.Wi.Data()},
		{Name: "Wc", Shape: l.Wc.Shape(), Data: l.Wc.Data()},
		{Name: "Wo", Shape: l.Wo.Shape(), Data: l.Wo.Data()},
		{Name: "Bf", Shape: l.Bf.Shape(), Data: l.Bf.Data()},
		{Name: "Bi", Shape: l.Bi.Shape(), Data: l.Bi.Data()},
		{Name: "Bc", Shape: l.Bc.Shape(), Data: l.Bc.Data()},
		{Name: "Bo", Shape: l.Bo.Shape(), Data: l.Bo.Data()},
	}

	return config, tensors
}

func (l *LSTM) Load(config map[string]any, tensors []model.TensorData) error {
	sigmoidName, ok := config["activation_sigmoid"].(string)
	if !ok {
		return errors.New("invalid activation_sigmoid")
	}
	sigmoid, err := activation.NewActivationByName(sigmoidName)
	if err != nil {
		return err
	}
	l.sigmoid = sigmoid

	tanhName, ok := config["activation_tanh"].(string)
	if !ok {
		return errors.New("invalid activation_tanh")
	}
	tanh, err := activation.NewActivationByName(tanhName)
	if err != nil {
		return err
	}
	l.tanh = tanh

	for _, tensorData := range tensors {
		switch tensorData.Name {
		case "Wf":
			l.Wf = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Wi":
			l.Wi = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Wc":
			l.Wc = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Wo":
			l.Wo = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bf":
			l.Bf = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bi":
			l.Bi = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bc":
			l.Bc = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		case "Bo":
			l.Bo = tensor.NewTensor(tensorData.Data, tensorData.Shape)
		default:
			return errors.New("unexpected tensor name: " + tensorData.Name)
		}
	}

	return nil
}
