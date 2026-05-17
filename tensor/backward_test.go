package tensor

import (
	"math"
	"slices"
	"testing"

	"harryxu.ca/goml/ndarray"
)

const (
	gradCheckH   = 1e-7
	gradCheckTol = 1e-4
)

// GradCheck compares the analytic gradient produced by Backward against
// a numeric gradient computed via central differences. It treats
// L = sum(f(x)) as the scalar loss and verifies that x.grad matches
// dL/dx to within gradCheckTol.
//
// x must be a fresh leaf (requiresGrad = true, grad nil or zero). The
// caller is responsible for not reusing x across tests, since gradCheck
// populates x.grad.
func gradCheck(t *testing.T, name string, x *Tensor, f func(*Tensor) *Tensor) {
	t.Helper()

	// Analytic gradient
	y := f(x)
	y.Backward()

	analytic := x.grad.Copy()

	// Numeric gradient
	shape := x.Shape()
	size := x.data.Size()
	numeric := ndarray.NewNDArray(shape...)

	for flat := range size {
		multiIndex := flatToMulti(flat, shape)
		original := x.data.Get(multiIndex)

		x.data.Set(multiIndex, original+gradCheckH)
		lossPlus := scalarSum(f(x).data)
		x.data.Set(multiIndex, original-gradCheckH)
		lossMinus := scalarSum(f(x).data)

		x.data.Set(multiIndex, original)

		numeric.Set(multiIndex, (lossPlus-lossMinus)/(2*gradCheckH))
	}

	// Compare
	for flat := range size {
		idx := flatToMulti(flat, shape)
		a := analytic.Get(idx)

		n := numeric.Get(idx)
		if math.Abs(a-n) > gradCheckTol {
			t.Errorf("%s: grad at %v: analytic=%g, numeric=%g, |diff|=%g",
				name, idx, a, n, math.Abs(a-n))
		}
	}
}

// flatToMulti converts a flat index in [0, prod(shape)) to a multi-index
// in row-major order.
func flatToMulti(flat int, shape []int) []int {
	out := make([]int, len(shape))
	for i, v := range slices.Backward(shape) {
		out[i] = flat % v
		flat /= v
	}

	return out
}

// scalarSum returns the sum of every element of a, regardless of rank.
func scalarSum(a *ndarray.NDArray) float64 {
	var s float64
	for v := range a.All() {
		s += v
	}

	return s
}

// TestBackwardNoRequiresGradIsNoOp verifies that calling Backward on a
// result whose graph has no grad-requiring leaves is a clean no-op:
// no panic, no grad allocation anywhere.
func TestBackwardNoRequiresGradIsNoOp(t *testing.T) {
	a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	b := NewTensor(ndarray.FromSlice([]float64{4, 5, 6}, 3))
	out := Add(a, b)

	if out.RequiresGrad() {
		t.Fatalf("Add of two non-grad tensors must not require grad")
	}

	out.Backward()

	if a.Grad() != nil {
		t.Errorf("a.Grad() should remain nil; got %v", a.Grad())
	}

	if b.Grad() != nil {
		t.Errorf("b.Grad() should remain nil; got %v", b.Grad())
	}

	if out.Grad() != nil {
		t.Errorf("out.Grad() should remain nil; got %v", out.Grad())
	}
}

// TestBackwardSharedParentAccumulates exercises the diamond case where
// the same leaf appears twice in a node's parent list. topoSort must
// dedup the leaf, but the consumer's backward must call accumulateGrad
// once per parent slot, so the leaf's gradient doubles.
func TestBackwardSharedParentAccumulates(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4}, 2, 2))
	gradCheck(t, "Add(x,x)", x, func(x *Tensor) *Tensor {
		return Add(x, x)
	})
}

// TestBackwardBranchRejoin exercises a leaf that feeds two distinct
// intermediate nodes which both feed the output. Both intermediates
// must run their backward before the leaf is touched, and their
// contributions must accumulate into the leaf's grad.
func TestBackwardBranchRejoin(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{0.5, 1.0, 2.0, 3.0}, 4))
	gradCheck(t, "Exp(x)+Log(x)", x, func(x *Tensor) *Tensor {
		return Add(Exp(x), Log(x))
	})
}

// TestBackwardAccumulatesAcrossCalls locks in the contract that
// successive Backward calls without ZeroGrad accumulate gradients
// into leaves. Training loops rely on this to call ZeroGrad
// explicitly between steps.
func TestBackwardAccumulatesAcrossCalls(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	y := Neg(x)

	y.Backward()
	y.Backward()

	got := slices.Collect(x.grad.All())

	want := []float64{-2, -2, -2}
	if !slices.Equal(got, want) {
		t.Errorf("two Backward calls without ZeroGrad: want %v, got %v", want, got)
	}
}

// TestZeroGradResetsBetweenSteps verifies the training-loop pattern:
// Backward, ZeroGrad, Backward again must produce the single-call
// gradient (not 2x).
func TestZeroGradResetsBetweenSteps(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	y := Neg(x)

	y.Backward()
	x.ZeroGrad()
	y.Backward()

	got := slices.Collect(x.grad.All())

	want := []float64{-1, -1, -1}
	if !slices.Equal(got, want) {
		t.Errorf("ZeroGrad between Backward calls: want %v, got %v", want, got)
	}
}
