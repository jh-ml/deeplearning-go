package tensor_test

import (
	"github.com/google/uuid"
	"github.com/jh-ml/deeplearning-go/neuralnetwork/tensor"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestNewTensor(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	if !float64sEqual(t1.Data(), data) {
		t.Errorf("NewTensor Data() = %v, want %v", t1.Data(), data)
	}
	if !reflect.DeepEqual(t1.Shape(), shape) {
		t.Errorf("NewTensor Shape() = %v, want %v", t1.Shape(), shape)
	}
}

func TestNewRandomTensor(t *testing.T) {
	shape := []int{3, 3}
	t1 := tensor.NewRandomTensor(shape)

	if !reflect.DeepEqual(t1.Shape(), shape) {
		t.Errorf("NewRandomTensor Shape() = %v, want %v", t1.Shape(), shape)
	}

	for _, value := range t1.Data() {
		if value < -1 || value > 1 {
			t.Errorf("NewRandomTensor Data() value out of range: %v", value)
		}
	}
}

func TestNewOnesTensor(t *testing.T) {
	shape := []int{3, 3}
	t1 := tensor.NewOnesTensor(shape)

	expectedData := make([]float64, 9)
	for i := range expectedData {
		expectedData[i] = 1.0
	}

	if !float64sEqual(t1.Data(), expectedData) {
		t.Errorf("NewOnesTensor Data() = %v, want %v", t1.Data(), expectedData)
	}
	if !reflect.DeepEqual(t1.Shape(), shape) {
		t.Errorf("NewOnesTensor Shape() = %v, want %v", t1.Shape(), shape)
	}
}

func TestNewZerosTensor(t *testing.T) {
	shape := []int{3, 3}
	t1 := tensor.NewZerosTensor(shape)

	expectedData := make([]float64, 9)

	if !float64sEqual(t1.Data(), expectedData) {
		t.Errorf("NewZerosTensor Data() = %v, want %v", t1.Data(), expectedData)
	}
	if !reflect.DeepEqual(t1.Shape(), shape) {
		t.Errorf("NewZerosTensor Shape() = %v, want %v", t1.Shape(), shape)
	}
}

func float64sEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			return false
		}
	}
	return true
}

func TestTensorID(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	id := t1.ID()

	if id == uuid.Nil {
		t.Errorf("ID() = %v, want non-nil UUID", id)
	}
}

func TestTensorData(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	if !float64sEqual(t1.Data(), data) {
		t.Errorf("Data() = %v, want %v", t1.Data(), data)
	}
}

func TestTensorShape(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	if !reflect.DeepEqual(t1.Shape(), shape) {
		t.Errorf("Shape() = %v, want %v", t1.Shape(), shape)
	}
}

func TestTensorClone(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	clone := t1.Clone()

	if !float64sEqual(clone.Data(), data) {
		t.Errorf("Clone Data() = %v, want %v", clone.Data(), data)
	}
	if !reflect.DeepEqual(clone.Shape(), shape) {
		t.Errorf("Clone Shape() = %v, want %v", clone.Shape(), shape)
	}
}

func TestTensorSetData(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	newData := []float64{5, 6, 7, 8}
	t1.SetData(newData)

	if !float64sEqual(t1.Data(), newData) {
		t.Errorf("SetData Data() = %v, want %v", t1.Data(), newData)
	}
}

func TestTensorTranspose(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	transposed := t1.Transpose()

	expectedData := []float64{1, 3, 2, 4}
	expectedShape := []int{2, 2}

	if !float64sEqual(transposed.Data(), expectedData) {
		t.Errorf("Transpose Data() = %v, want %v", transposed.Data(), expectedData)
	}
	if !reflect.DeepEqual(transposed.Shape(), expectedShape) {
		t.Errorf("Transpose Shape() = %v, want %v", transposed.Shape(), expectedShape)
	}
}

func TestTensorAdd(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data1, shape)
	t2 := tensor.NewTensor(data2, shape)
	result := t1.Add(t2)

	expectedData := []float64{6, 8, 10, 12}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Add Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("Add Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorSubtract(t *testing.T) {
	data1 := []float64{5, 6, 7, 8}
	data2 := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data1, shape)
	t2 := tensor.NewTensor(data2, shape)
	result := t1.Subtract(t2)

	expectedData := []float64{4, 4, 4, 4}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Subtract Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("Subtract Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorMultiply(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data1, shape)
	t2 := tensor.NewTensor(data2, shape)
	result := t1.Multiply(t2)

	expectedData := []float64{5, 12, 21, 32}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Multiply Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("Multiply Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorDivide(t *testing.T) {
	data1 := []float64{10, 20, 30, 40}
	data2 := []float64{2, 4, 6, 8}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data1, shape)
	t2 := tensor.NewTensor(data2, shape)
	result := t1.Divide(t2)

	expectedData := []float64{5, 5, 5, 5}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Divide Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("Divide Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorDot(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}
	shape1 := []int{2, 2}
	shape2 := []int{2, 2}
	t1 := tensor.NewTensor(data1, shape1)
	t2 := tensor.NewTensor(data2, shape2)
	result := t1.Dot(t2)

	expectedData := []float64{19, 22, 43, 50}
	expectedShape := []int{2, 2}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Dot Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), expectedShape) {
		t.Errorf("Dot Shape() = %v, want %v", result.Shape(), expectedShape)
	}
}

func TestTensorAddScalar(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	result := t1.AddScalar(5)

	expectedData := []float64{6, 7, 8, 9}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("AddScalar Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("AddScalar Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorMultiplyScalar(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	result := t1.MultiplyScalar(2)

	expectedData := []float64{2, 4, 6, 8}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("MultiplyScalar Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), shape) {
		t.Errorf("MultiplyScalar Shape() = %v, want %v", result.Shape(), shape)
	}
}

func TestTensorSameShape(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)
	t2 := tensor.NewTensor(data, shape)
	t3 := tensor.NewTensor(data, []int{4, 1})

	if !t1.SameShape(t2) {
		t.Errorf("sameShape = false, want true")
	}
	if t1.SameShape(t3) {
		t.Errorf("sameShape = true, want false")
	}
}

func TestTensorSumAlongBatch(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	t1 := tensor.NewTensor(data, shape)
	result := t1.SumAlongBatch()

	expectedData := []float64{5, 7, 9}
	expectedShape := []int{1, 3}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("SumAlongBatch Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), expectedShape) {
		t.Errorf("SumAlongBatch Shape() = %v, want %v", result.Shape(), expectedShape)
	}
}

func TestTensorThreshold(t *testing.T) {
	data := []float64{0.1, 0.5, 0.9}
	shape := []int{3}
	t1 := tensor.NewTensor(data, shape)
	probability := 0.5
	rand.Seed(0) // For reproducibility
	result := t1.Threshold(probability)

	for _, v := range result.Data() {
		if v != 0.0 && v != 1.0 {
			t.Errorf("Threshold Data() = %v, want 0.0 or 1.0", result.Data())
		}
	}
}

func TestTensorRow(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	row, err := t1.Row(1)
	if err != nil {
		t.Errorf("Row() error = %v, want nil", err)
	}

	expectedRow := []float64{3, 4}
	if !float64sEqual(row, expectedRow) {
		t.Errorf("Row() = %v, want %v", row, expectedRow)
	}

	_, err = t1.Row(2)
	if err == nil {
		t.Errorf("Row() error = nil, want out of bounds error")
	}
}

func TestTensorAddRow(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	err := t1.AddRow(1, []float64{1, 1})
	if err != nil {
		t.Errorf("AddRow() error = %v, want nil", err)
	}

	expectedData := []float64{1, 2, 4, 5}
	if !float64sEqual(t1.Data(), expectedData) {
		t.Errorf("AddRow Data() = %v, want %v", t1.Data(), expectedData)
	}

	err = t1.AddRow(2, []float64{1, 1})
	if err == nil {
		t.Errorf("AddRow() error = nil, want out of bounds error")
	}
}

func TestTensorConcatenate(t *testing.T) {
	t1 := tensor.NewTensor([]float64{1, 2}, []int{2})
	t2 := tensor.NewTensor([]float64{3, 4}, []int{2})
	t3 := tensor.NewTensor([]float64{5, 6}, []int{2})

	result := t1.Concatenate([]tensor.Interface{t2, t3})

	expectedData := []float64{1, 2, 3, 4, 5, 6}
	expectedShape := []int{6}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Concatenate Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), expectedShape) {
		t.Errorf("Concatenate Shape() = %v, want %v", result.Shape(), expectedShape)
	}
}

func TestTensorSplit(t *testing.T) {
	t1 := tensor.NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{6})
	sizes := []int{2, 2, 2}

	results := t1.Split(sizes)

	expectedData := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}

	for i, result := range results {
		if !float64sEqual(result.Data(), expectedData[i]) {
			t.Errorf("Split Data() = %v, want %v", result.Data(), expectedData[i])
		}
	}
}

func TestTensorSize(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	t1 := tensor.NewTensor(data, shape)

	expectedSize := len(data)
	if t1.Size() != expectedSize {
		t.Errorf("Size() = %v, want %v", t1.Size(), expectedSize)
	}
}

func TestConcatenate(t *testing.T) {
	t1 := tensor.NewTensor([]float64{1, 2}, []int{2})
	t2 := tensor.NewTensor([]float64{3, 4}, []int{2})
	t3 := tensor.NewTensor([]float64{5, 6}, []int{2})

	result := tensor.Concatenate([]tensor.Interface{t1, t2, t3})

	expectedData := []float64{1, 2, 3, 4, 5, 6}
	expectedShape := []int{6}

	if !float64sEqual(result.Data(), expectedData) {
		t.Errorf("Concatenate Data() = %v, want %v", result.Data(), expectedData)
	}
	if !reflect.DeepEqual(result.Shape(), expectedShape) {
		t.Errorf("Concatenate Shape() = %v, want %v", result.Shape(), expectedShape)
	}
}
