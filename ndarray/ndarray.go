package ndarray

import (
	"fmt"
	"slices"
)

// NDArray is a strided view of a flat float64 buffer. The triple
// (shape, strides, offset) determines how the buffer data is interpreted:
// Element [i_0, ..., i_{n-1}] lives at data[offset + sum(i_k * strides[k])].
//
// Allocations from NewNDArray use a row-major interpretation by default:
// strides = rowMajorStrides(shape) and offset = 0, so the last axis varies fastest.
//
// A view is contiguous when its strides equal rowMajorStrides(shape) for its
// current shape, i.e. iterating its elements in shape order walks the
// buffer linearly. Fresh allocations are contiguous.
type NDArray struct {
	data    []float64
	shape   []int
	strides []int
	offset  int
}

// rowMajorStrides returns the "canonical" strides for a contiguous
// tensor of the given shape with offset zero.
//
// For a shape (d_0, d_1, ..., d_{n-1}), the row-major (C-order) strides are:
//
//	strides[k]   = d_{k+1} * d_{k+2} * ... * d_{n-1}
//	strides[n-1] = 1
//
// Examples:
//
//	shape []        -> strides []          // scalar
//	shape [3]       -> strides [1]
//	shape [2, 3]    -> strides [3, 1]      // step a row = skip 3 elements
//	shape [2, 3, 4] -> strides [12, 4, 1]
//
// The returned slice always has the same length as shape.
func rowMajorStrides(shape []int) []int {
	strides := make([]int, len(shape))

	if len(shape) == 0 {
		return strides
	}

	strides[len(strides)-1] = 1

	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return strides
}

// NewNDArray allocates a fresh, zero-initialized NDArray of the given shape
// with a row-major interpretation (strides = rowMajorStrides(shape),
// offset = 0). The returned view is contiguous.
//
// The total number of elements is the product of the shape entries; an empty
// shape produces a rank-0 scalar (size 1, the empty product).
//
// Examples:
//
//	NewNDArray()      // scalar (rank 0, size 1)
//	NewNDArray(5)     // length-5 vector
//	NewNDArray(2, 3)  // 2x3 matrix
func NewNDArray(shape ...int) *NDArray {
	shape = slices.Clone(shape)

	size := 1

	for i, dimension := range shape {
		if dimension < 0 {
			panic(fmt.Sprintf("ndarray: shape dimension on axis %d must be non-negative, got %d", i, dimension))
		}

		size *= dimension
	}

	strides := rowMajorStrides(shape)

	return &NDArray{
		data:    make([]float64, size),
		shape:   shape,
		strides: strides,
	}
}

// Copy returns a new contiguous NDArray with the same shape and view
// contents as an existing NDArray. The result has strides = rowMajorStrides(shape)
// and offset = 0 regardless of whether a was contiguous; non-contiguous views
// (transposed, broadcast, sliced) are traversed in row-major order, so
// the returned tensor always satisfies result.IsContiguous().
//
// Modifying the returned NDArray does not affect a.
func (a *NDArray) Copy() *NDArray {
	result := NewNDArray(a.shape...)

	i := 0
	for _, off := range a.indicesAndOffsets() {
		result.data[i] = a.data[off]
		i++
	}

	return result
}

// Ndim returns the number of dimensions/axes of the tensor represented
// by the NDArray.
func (a *NDArray) Ndim() int {
	return len(a.shape)
}

// Size returns the total number of elements in the tensor represented by
// the NDArray. This is not necessarily equal to the number of elements in the
// backing buffer.
func (a *NDArray) Size() int {
	size := 1

	for _, dim := range a.shape {
		size *= dim
	}

	return size
}

// Shape returns the size of each axis as a fresh copy. Modifying the returned
// slice does not affect the NDArray.
//
// The returned slice has length equal to the tensor's rank (0 for a scalar).
func (a *NDArray) Shape() []int {
	return slices.Clone(a.shape)
}

// Strides returns the per-axis stride values as a fresh copy. Stride k is the
// number of buffer elements advanced when the k-th index increments by one;
// see the NDArray docs for the indexing formula. Modifying the returned slice
// does not affect the NDArray.
//
// Mostly useful for tests and debugging — public ops should rely on Shape()
// and the view methods rather than reasoning about strides directly.
func (a *NDArray) Strides() []int {
	return slices.Clone(a.strides)
}

// IsContiguous reports whether the view's strides equal rowMajorStrides(shape)
// for its current shape — i.e., iterating elements in shape order walks the
// underlying buffer linearly with no skips, gaps, or revisits.
//
// Fresh allocations from NewNDArray are contiguous. Transpose, broadcast, and
// stepped slices generally are not. Reshape uses this to decide whether it
// can return a zero-copy view or must materialize a fresh contiguous buffer.
func (a *NDArray) IsContiguous() bool {
	return slices.Equal(a.strides, rowMajorStrides(a.shape))
}

// FromSlice wraps an existing flat float64 buffer as a row-major NDArray of
// the given shape. The returned NDArray aliases data (no copy is made).
// Subsequent mutations to data are visible through the NDArray and vice
// versa; call Copy on the result if you need an independent buffer.
func FromSlice(data []float64, shape ...int) *NDArray {
	shape = slices.Clone(shape)

	size := 1

	for i, dimension := range shape {
		if dimension < 0 {
			panic(fmt.Sprintf("ndarray: shape dimension on axis %d must be non-negative, got %d", i, dimension))
		}

		size *= dimension
	}

	if len(data) != size {
		panic(fmt.Sprintf("ndarray: length of data slice must be %d, got %d", size, len(data)))
	}

	return &NDArray{
		data:    data,
		shape:   shape,
		strides: rowMajorStrides(shape),
	}
}
