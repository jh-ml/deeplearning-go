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
	ft, it, gt, ot           []tensor.Interface
	hiddenStates, cellStates []tensor.Interface
	sigmoid, tanh            activation.Interface
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
func (l *LSTM) Forward(inputs []tensor.Interface) []tensor.Interface {
	sequenceLength := len(inputs)
	l.hiddenStates = make([]tensor.Interface, sequenceLength)
	l.cellStates = make([]tensor.Interface, sequenceLength)
	l.ft = make([]tensor.Interface, sequenceLength)
	l.it = make([]tensor.Interface, sequenceLength)
	l.gt = make([]tensor.Interface, sequenceLength)
	l.ot = make([]tensor.Interface, sequenceLength)

	for t := 0; t < sequenceLength; t++ {
		input := inputs[t]

		// Compute gates
		l.ft[t] = l.sigmoid.Forward(input.Dot(l.Wf).Add(l.H.Dot(l.Uf)).Add(l.Bf))
		l.it[t] = l.sigmoid.Forward(input.Dot(l.Wi).Add(l.H.Dot(l.Ui)).Add(l.Bi))
		l.gt[t] = l.tanh.Forward(input.Dot(l.Wc).Add(l.H.Dot(l.Uc)).Add(l.Bc))
		l.ot[t] = l.sigmoid.Forward(input.Dot(l.Wo).Add(l.H.Dot(l.Uo)).Add(l.Bo))

		// Update cell state and hidden state
		l.C = l.ft[t].Multiply(l.C).Add(l.it[t].Multiply(l.gt[t]))
		l.H = l.ot[t].Multiply(l.tanh.Forward(l.C))

		// Store states
		l.hiddenStates[t] = l.H
		l.cellStates[t] = l.C
	}

	return l.hiddenStates
}

// Backward pass for LSTM
func (l *LSTM) Backward(dh []tensor.Interface) []tensor.Interface {
	sequenceLength := len(dh)
	dhPrev := tensor.NewZerosTensor(l.H.Shape()) // Initialize previous hidden state gradient
	dCPrev := tensor.NewZerosTensor(l.C.Shape()) // Initialize previous cell state gradient
	dInputs := make([]tensor.Interface, sequenceLength)

	for t := sequenceLength - 1; t >= 0; t-- {
		dhCurrent := dh[t].Add(dhPrev).(*tensor.Tensor) // Ensure type is *Tensor

		// Derivative of the loss with respect to the output gate
		do := dhCurrent.Multiply(l.tanh.Forward(l.cellStates[t])).Multiply(l.sigmoid.Backward(l.ot[t])).(*tensor.Tensor)

		// Derivative of the loss with respect to the cell state
		dC := dhCurrent.Multiply(l.ot[t]).Multiply(l.tanh.Backward(l.cellStates[t])).Add(dCPrev.Multiply(l.ft[t])).(*tensor.Tensor)

		// Derivative of the loss with respect to the candidate memory cell
		dg := dC.Multiply(l.it[t]).Multiply(l.tanh.Backward(l.gt[t])).(*tensor.Tensor)

		// Derivative of the loss with respect to the input gate
		di := dC.Multiply(l.gt[t]).Multiply(l.sigmoid.Backward(l.it[t])).(*tensor.Tensor)

		// Derivative of the loss with respect to the forget gate
		df := dC.Multiply(l.cellStates[t]).Multiply(l.sigmoid.Backward(l.ft[t])).(*tensor.Tensor)

		// Compute gradients for weights and biases
		l.dWo = l.dWo.Add(l.hiddenStates[t].Transpose().Dot(do)).(*tensor.Tensor)
		l.dBo = l.dBo.Add(do.SumAlongBatch()).(*tensor.Tensor)
		l.dWc = l.dWc.Add(l.hiddenStates[t].Transpose().Dot(dg)).(*tensor.Tensor)
		l.dBc = l.dBc.Add(dg.SumAlongBatch()).(*tensor.Tensor)
		l.dWi = l.dWi.Add(l.hiddenStates[t].Transpose().Dot(di)).(*tensor.Tensor)
		l.dBi = l.dBi.Add(di.SumAlongBatch()).(*tensor.Tensor)
		l.dWf = l.dWf.Add(l.hiddenStates[t].Transpose().Dot(df)).(*tensor.Tensor)
		l.dBf = l.dBf.Add(df.SumAlongBatch()).(*tensor.Tensor)

		// Compute gradient with respect to the input
		dInput := do.Dot(l.Wo.Transpose()).Add(dg.Dot(l.Wc.Transpose())).Add(di.Dot(l.Wi.Transpose())).Add(df.Dot(l.Wf.Transpose())).(*tensor.Tensor)
		dInputs[t] = dInput

		// Update gradients for previous time step
		dhPrev = do.Dot(l.Uo.Transpose()).Add(dg.Dot(l.Uc.Transpose())).Add(di.Dot(l.Ui.Transpose())).Add(df.Dot(l.Uf.Transpose())).(*tensor.Tensor)
		dCPrev = dC.Multiply(l.ft[t]).(*tensor.Tensor) // Ensure the correct type
	}

	return dInputs
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
