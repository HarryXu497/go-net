package tensor

import (
	"testing"

	"harryxu.ca/goml/ndarray"
)

// TestSumGrad covers the three reshape regimes the backward has to handle:
// same-rank (keepDims=true), rank-changing (keepDims=false), and full
// reduction. The keepDims=false case is the one that exercises a non-
// trivial rank change in Reshape and is most likely to surface a bug.
func TestSumGrad(t *testing.T) {
	t.Run("axes=[0], keepDims=true (2,3) -> (1,3)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Sum(x, [0], true)", x, func(x *Tensor) *Tensor {
			return Sum(x, []int{0}, true)
		})
	})

	t.Run("axes=[0], keepDims=false (2,3) -> (3,)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Sum(x, [0], false)", x, func(x *Tensor) *Tensor {
			return Sum(x, []int{0}, false)
		})
	})

	t.Run("axes=nil, keepDims=true full reduction", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Sum(x, nil, true)", x, func(x *Tensor) *Tensor {
			return Sum(x, nil, true)
		})
	})
}

// TestMeanGrad: one partial-axis case and one full-reduction case. The
// latter exercises the count = t.data.Size() path that the partial case
// never touches.
func TestMeanGrad(t *testing.T) {
	t.Run("axes=[1], keepDims=true (2,3) -> (2,1)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Mean(x, [1], true)", x, func(x *Tensor) *Tensor {
			return Mean(x, []int{1}, true)
		})
	})

	t.Run("axes=nil, keepDims=true full reduction (denominator = Size)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Mean(x, nil, true)", x, func(x *Tensor) *Tensor {
			return Mean(x, nil, true)
		})
	})
}
