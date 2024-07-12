---
description: Guiding a network to find patterns in a dataset
---

# Training and Predicting

### Training

```go
func (nn *NeuralNetwork) Train(data, targets []tensor.Interface, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float64
		for i := 0; i < len(data); i++ {
			output := nn.Forward(data[i])
			lossV, grad := nn.lossFunction.Compute(output, targets[i])
			epochLoss += lossV.Data()[0]
			nn.Backward(grad)
			nn.Regularise()
			nn.Optimise()
			nn.ZeroGradients()
		}
		epochLoss /= float64(len(data))
		fmt.Printf("Epoch %d, Loss: %f\n", epoch, epochLoss)
	}
}
```

We pass two tensors, one with the training data and another with the true values, used to compare the predictions to the targets. We pass in the number of epochs, or iterations to perform. The data is processed through the forward pass, the loss calculated and propagated backwards for the gradients to be calculated by each layer. The Regularise and Optimise functions then use the gradients to update the weights and biases. We then zero the gradients before the next data sample is processed. Te epoch loss value is accumulated and averaged.

\
If the accuracy of the network reaches a desired amount, the state of the network is saved as a model, to be loaded and used for predictions later on.

### Prediction

Prediction involves using a trained neural network to make predictions on new, unseen data. The process is simpler than training, as it only involves a forward pass through the network. The save model is loaded and used to initialise a neural network instance, setting up all the layers, weights and biases etc. that were saved after training. This would ideally be put behind an inference API, where samples are sent to the system, and the resulting prediction returned to the caller.
