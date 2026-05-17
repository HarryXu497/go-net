package tensor

import (
	"slices"
	"testing"

	"harryxu.ca/gonet/ndarray"
)

// TestReshapeGrad exercises both a rank-preserving reshape and a
// rank-changing one (flatten). The flatten case is the one MNIST will
// hit, so it's the load-bearing variant.
func TestReshapeGrad(t *testing.T) {
	t.Run("rank-preserving (2,3) -> (3,2)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Reshape(x, 3, 2)", x, func(x *Tensor) *Tensor {
			return Reshape(x, 3, 2)
		})
	})

	t.Run("flatten (2,3) -> (6,)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Reshape(x, 6)", x, func(x *Tensor) *Tensor {
			return Reshape(x, 6)
		})
	})
}

// TestTransposeGrad covers the default reverse-all case and an explicit
// 3-axis permutation whose inverse differs from itself. A 2D swap would
// be self-inverse and so wouldn't catch an inversePerm bug.
func TestTransposeGrad(t *testing.T) {
	t.Run("default reverse (2,3)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Transpose(x)", x, func(x *Tensor) *Tensor {
			return Transpose(x)
		})
	})

	t.Run("explicit perm [2,0,1] (2,3,4)", func(t *testing.T) {
		data := make([]float64, 24)
		for i := range data {
			data[i] = float64(i + 1)
		}

		x := NewLeaf(ndarray.FromSlice(data, 2, 3, 4))
		gradCheck(t, "Transpose(x, 2, 0, 1)", x, func(x *Tensor) *Tensor {
			return Transpose(x, 2, 0, 1)
		})
	})
}

// TestInversePerm pins the helper's contract on the case that matters
// (a non-self-inverse permutation) so a regression in inversePerm
// surfaces here rather than as a confusing Transpose gradCheck failure.
func TestInversePerm(t *testing.T) {
	got := inversePerm([]int{2, 0, 1})

	want := []int{1, 2, 0}
	if !slices.Equal(got, want) {
		t.Errorf("inversePerm([2,0,1]) = %v, want %v", got, want)
	}
}
