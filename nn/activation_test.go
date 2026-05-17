package nn

import (
	"slices"
	"testing"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

// TestReLUForward verifies the forward elementwise: positives pass
// through, non-positives clamp to 0, and the output shape matches
// the input.
func TestReLUForward(t *testing.T) {
	r := &ReLU{}
	x := tensor.NewTensor(ndarray.FromSlice([]float64{-2, -0.5, 0, 0.5, 2}, 5))

	y := r.Forward(x)

	if !slices.Equal(y.Shape(), []int{5}) {
		t.Errorf("output shape = %v, want [5]", y.Shape())
	}

	got := slices.Collect(y.Data().All())

	want := []float64{0, 0, 0, 0.5, 2}
	if !slices.Equal(got, want) {
		t.Errorf("output = %v, want %v", got, want)
	}
}

// TestReLUParametersEmpty: a parameter-less module must produce zero
// parameters.
func TestReLUParametersEmpty(t *testing.T) {
	r := &ReLU{}
	if got := r.Parameters(); len(got) != 0 {
		t.Errorf("Parameters() length = %d, want 0", len(got))
	}
}

// TestReLUBackwardPopulatesGrad build a small computational graph,
// runs Backward, and checks that the leaf's grad matches the ReLU derivative
// (1 where positive, 0 elsewhere).
func TestReLUBackwardPopulatesGrad(t *testing.T) {
	r := &ReLU{}
	x := tensor.NewLeaf(ndarray.FromSlice([]float64{-1.5, -0.1, 0.3, 1.0}, 4))

	loss := tensor.Sum(r.Forward(x), nil, true)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatalf("x.Grad() is nil after Backward")
	}

	got := slices.Collect(x.Grad().All())

	want := []float64{0, 0, 1, 1}
	if !slices.Equal(got, want) {
		t.Errorf("x.Grad() = %v, want %v", got, want)
	}
}
