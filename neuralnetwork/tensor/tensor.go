package tensor

import (
	"errors"
	"fmt"
	"github.com/google/uuid"
	"math"
	"math/rand/v2"
)

// Tensor is an implementation of the Tensor interface
type Tensor struct {
	data  []float64
	shape []int
	id    uuid.UUID
}

// NewTensor creates a new Tensor
func NewTensor(data []float64, shape []int) *Tensor {
	return &Tensor{data: data, shape: shape, id: uuid.New()}
}

// NewRandomTensor creates a new Tensor with random values
func NewRandomTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.Float64()*2 - 1
	}
	return NewTensor(data, shape)
}

// NewOnesTensor creates a new Tensor filled with zeros
func NewOnesTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(data, shape)
}

// NewZerosTensor creates a new Tensor filled with zeros
func NewZerosTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	return NewTensor(data, shape)
}

func NewXavierWeightsTensor(inputSize, outputSize int) *Tensor {
	data := make([]float64, inputSize*outputSize)
	scale := math.Sqrt(2.0 / float64(inputSize+outputSize))
	for i := range data {
		data[i] = rand.NormFloat64() * scale
	}
	return NewTensor(data, []int{inputSize, outputSize})
}

func (t *Tensor) ID() uuid.UUID {
	return t.id
}

func (t *Tensor) Data() []float64 {
	return t.data
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Clone() Interface {
	cloneData := make([]float64, len(t.data))
	copy(cloneData, t.data)
	return NewTensor(cloneData, t.shape)
}

func (t *Tensor) SetData(data []float64) {
	if len(data) != len(t.data) {
		panic("Data length does not match tensor shape")
	}
	t.data = data
}

func (t *Tensor) Index(indices ...int) int {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("number of indices (%d) does not match number of dimensions (%d)", len(indices), len(t.shape)))
	}
	index := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			panic(fmt.Sprintf("index out of range: indices[%d]=%d out of shape[%d]=%d", i, idx, i, t.shape[i]))
		}
		if i == 0 {
			index = idx
		} else {
			index = index*t.shape[i] + idx
		}
	}
	return index
}

func (t *Tensor) Get(indices ...int) float64 {
	index := t.Index(indices...)
	return t.data[index]
}

func (t *Tensor) Set(value float64, indices ...int) {
	index := t.Index(indices...)
	t.data[index] = value
}

func (t *Tensor) Transpose() Interface {
	if len(t.shape) != 2 {
		panic("Transpose requires a 2D tensor")
	}
	resultShape := []int{t.shape[1], t.shape[0]}
	result := make([]float64, len(t.data))
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			result[j*resultShape[1]+i] = t.data[i*t.shape[1]+j]
		}
	}
	return NewTensor(result, resultShape)
}

func (t *Tensor) Sum() float64 {
	sum := 0.0
	for _, v := range t.data {
		sum += v
	}
	return sum
}

func (t *Tensor) Add(other Interface) Interface {
	otherData := other.Data()
	if !t.SameShape(other) {
		panic("Shapes do not match for addition")
	}
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] + otherData[i]
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) Slice(index int, axis int) Interface {
	shape := t.Shape()
	if axis < 0 || axis >= len(shape) {
		panic("Invalid axis for slicing")
	}

	// Compute the new shape after slicing
	newShape := append([]int{}, shape[:axis]...)
	newShape = append(newShape, shape[axis+1:]...)

	// Compute the offset for the slice
	stride := 1
	for i := axis + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	offset := index * stride

	// Extract the slice data
	sliceData := t.Data()[offset : offset+stride]

	return NewTensor(sliceData, newShape)
}

func (t *Tensor) Subtract(other Interface) Interface {
	otherData := other.Data()
	if !t.SameShape(other) {
		panic("Shapes do not match for subtraction")
	}
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] - otherData[i]
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) Multiply(other Interface) Interface {
	otherData := other.Data()
	if !t.SameShape(other) {
		panic("Shapes do not match for multiplication")
	}
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] * otherData[i]
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) Divide(other Interface) Interface {
	otherData := other.Data()
	if !t.SameShape(other) {
		panic("Shapes do not match for division")
	}
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] / otherData[i]
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) Dot(other Interface) Interface {
	otherShape := other.Shape()
	if len(t.shape) != 2 || len(otherShape) != 2 {
		panic("Dot product requires 2D tensors")
	}
	if t.shape[1] != otherShape[0] {
		panic("Shapes do not match for dot product")
	}
	resultShape := []int{t.shape[0], otherShape[1]}
	result := make([]float64, resultShape[0]*resultShape[1])
	otherData := other.Data()
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < otherShape[1]; j++ {
			sum := 0.0
			for k := 0; k < t.shape[1]; k++ {
				sum += t.data[i*t.shape[1]+k] * otherData[k*otherShape[1]+j]
			}
			result[i*resultShape[1]+j] = sum
		}
	}
	return NewTensor(result, resultShape)
}

func (t *Tensor) AddScalar(scalar float64) Interface {
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] + scalar
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) MultiplyScalar(scalar float64) Interface {
	result := make([]float64, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] * scalar
	}
	return NewTensor(result, t.shape)
}

func (t *Tensor) SameShape(other Interface) bool {
	otherShape := other.Shape()
	if len(t.shape) != len(otherShape) {
		return false
	}
	for i := range t.shape {
		if t.shape[i] != otherShape[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) SumAlongBatch() Interface {
	if len(t.shape) < 2 {
		panic("SumAlongBatch requires at least 2 dimensions")
	}

	batchSize := t.shape[0]
	featureSize := t.shape[1]
	sums := make([]float64, featureSize)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < featureSize; j++ {
			sums[j] += t.data[i*featureSize+j]
		}
	}
	return NewTensor(sums, []int{1, featureSize})
}

func (t *Tensor) Threshold(probability float64) Interface {
	result := make([]float64, len(t.data))
	for i := range t.data {
		if rand.Float64() < probability {
			result[i] = 1.0
		} else {
			result[i] = 0.0
		}
	}
	return NewTensor(result, t.shape)
}

// Row retrieves a specific row from the tensor
func (t *Tensor) Row(rowIndex int) ([]float64, error) {
	if len(t.shape) != 2 {
		return nil, errors.New("Row operation only supports 2D tensors")
	}
	if rowIndex >= t.shape[0] || rowIndex < 0 {
		return nil, errors.New("rowIndex out of bounds")
	}

	start := rowIndex * t.shape[1]
	end := start + t.shape[1]
	return t.data[start:end], nil
}

// AddRow adds a given row (vector) to a specific row in the tensor
func (t *Tensor) AddRow(rowIndex int, row []float64) error {
	if len(t.shape) != 2 {
		return errors.New("AddRow operation only supports 2D tensors")
	}
	if rowIndex >= t.shape[0] || rowIndex < 0 {
		return errors.New("rowIndex out of bounds")
	}
	if len(row) != t.shape[1] {
		return errors.New("row length mismatch")
	}

	start := rowIndex * t.shape[1]
	for i := 0; i < t.shape[1]; i++ {
		t.data[start+i] += row[i]
	}
	return nil
}

func (t *Tensor) Concatenate(tensors []Interface) Interface {
	totalSize := len(t.data)
	for _, tensor := range tensors {
		totalSize += len(tensor.Data())
	}

	resultData := make([]float64, totalSize)
	offset := copy(resultData, t.data)
	for _, tensor := range tensors {
		offset += copy(resultData[offset:], tensor.Data())
	}

	resultShape := []int{totalSize}
	return NewTensor(resultData, resultShape)
}

func (t *Tensor) Split(sizes []int) []Interface {
	var result []Interface
	offset := 0
	for _, size := range sizes {
		slice := NewZerosTensor([]int{size})
		copy(slice.Data(), t.Data()[offset:offset+size])
		result = append(result, slice)
		offset += size
	}
	return result
}

func (t *Tensor) Size() int {
	return len(t.data)
}

func (t *Tensor) Reshape(newShape []int) Interface {
	if shapeSize(newShape) != shapeSize(t.shape) {
		panic("new shape must have the same number of elements as the old shape")
	}
	return &Tensor{data: t.data, shape: newShape}
}

func shapeSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

func (t *Tensor) Conv2D(other Interface, stride, padding int) Interface {
	if len(t.shape) != 4 || len(other.Shape()) != 4 {
		panic("Conv2D requires 4D tensors")
	}

	batchSize, inputChannels, inputHeight, inputWidth := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
	outputChannels, _, kernelHeight, kernelWidth := other.Shape()[0], other.Shape()[1], other.Shape()[2], other.Shape()[3]

	outputHeight := (inputHeight+2*padding-kernelHeight)/stride + 1
	outputWidth := (inputWidth+2*padding-kernelWidth)/stride + 1

	output := NewZerosTensor([]int{batchSize, outputChannels, outputHeight, outputWidth})

	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outputChannels; oc++ {
			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					sum := 0.0
					for ic := 0; ic < inputChannels; ic++ {
						for kh := 0; kh < kernelHeight; kh++ {
							for kw := 0; kw < kernelWidth; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
									sum += t.Get(b, ic, ih, iw) * other.Get(oc, ic, kh, kw)
								}
							}
						}
					}
					output.Set(sum, b, oc, oh, ow)
				}
			}
		}
	}

	return output
}

func (t *Tensor) Mean() float64 {
	sum := 0.0
	for _, v := range t.data {
		sum += v
	}
	return sum / float64(len(t.data))
}

func (t *Tensor) Min() float64 {
	min := math.MaxFloat64
	for _, v := range t.data {
		if v < min {
			min = v
		}
	}
	return min
}

func (t *Tensor) Max() float64 {
	max := -math.MaxFloat64
	for _, v := range t.data {
		if v > max {
			max = v
		}
	}
	return max
}

func (t *Tensor) StdDev() float64 {
	mean := t.Mean()
	sum := 0.0
	for _, v := range t.data {
		sum += (v - mean) * (v - mean)
	}
	variance := sum / float64(len(t.data))
	return math.Sqrt(variance)
}

func (t *Tensor) PrintStats(label string) {
	fmt.Printf("%s - Shape: %v, Mean: %f, Min: %f, Max: %f, StdDev: %f\n", label, t.shape, t.Mean(), t.Min(), t.Max(), t.StdDev())
}
