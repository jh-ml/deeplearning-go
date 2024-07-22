package tensor

import "github.com/google/uuid"

type Interface interface {
	ID() uuid.UUID
	Data() []float64
	Shape() []int
	Clone() Interface
	SetData(data []float64)
	Index(indices ...int) int
	Get(indices ...int) float64
	Set(value float64, indices ...int)
	Transpose() Interface
	Sum() float64
	Add(other Interface) Interface
	Slice(index int, axis int) Interface
	Subtract(other Interface) Interface
	Multiply(other Interface) Interface
	Divide(other Interface) Interface
	Dot(other Interface) Interface
	AddScalar(scale float64) Interface
	MultiplyScalar(scalar float64) Interface
	SumAlongBatch() Interface
	Threshold(probability float64) Interface
	Row(rowIndex int) ([]float64, error)
	AddRow(rowIndex int, row []float64) error
	Split([]int) []Interface
	Concatenate([]Interface) Interface
	Size() int
	Reshape(newShape []int) Interface
	Conv2D(other Interface, stride, padding int) Interface
	Mean() float64
	Min() float64
	Max() float64
	StdDev() float64
	PrintStats(label string)
}
