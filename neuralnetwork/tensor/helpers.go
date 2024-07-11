package tensor

// Concatenate function for concatenating multiple tensors
func Concatenate(tensors []Interface) Interface {
	totalSize := 0
	for _, tensor := range tensors {
		totalSize += len(tensor.Data())
	}

	result := NewZerosTensor([]int{totalSize})
	offset := 0
	for _, tensor := range tensors {
		copy(result.Data()[offset:], tensor.Data())
		offset += len(tensor.Data())
	}
	return result
}
