package ndarray

import "fmt"

// flatIndex returns the flat offset into a.data for the element at the
// given multi-index. Panics if the number of indices does not match the
// tensor's rank, or if any index is out of range for its axis.
//
// Examples:
//
//	a := NewNDArray(2, 3)
//	a.flatIndex([]int{1, 1}) // 4
func (a *NDArray) flatIndex(indices []int) int {
	if len(a.shape) != len(indices) {
		panic(fmt.Sprintf("ndarray: expected %d indices for rank-%d tensor, got %d", len(a.shape), len(a.shape), len(indices)))
	}

	scalarIndex := a.offset

	for i, index := range indices {
		if index < 0 || index >= a.shape[i] {
			panic(fmt.Sprintf("ndarray: index %d on axis %d is out of range [0, %d)", index, i, a.shape[i]))
		}

		scalarIndex += index * a.strides[i]
	}

	return scalarIndex
}

// Get returns the value at the given multi-index. Panics if the number of
// indices does not match the tensor's rank, or if any index is out of range
// for its axis.
//
// Examples:
//
//	a := NewNDArray(2, 3)
//	a.Set([]int{0, 0}, 1.0)
//	a.Get([]int{0, 0})    // 1.0
func (a *NDArray) Get(indices []int) float64 {
	return a.data[a.flatIndex(indices)]
}

// Set assigns a value at the given multi-index. Panics if the number of
// indices does not match the tensor's rank, or if any index is out of range
// for its axis.
//
// Examples:
//
//	a := NewNDArray(2, 3)
//	a.Set([]int{0, 0}, 1.0)
//	a.Get([]int{0, 0})    // 1.0
func (a *NDArray) Set(indices []int, value float64) {
	a.data[a.flatIndex(indices)] = value
}
