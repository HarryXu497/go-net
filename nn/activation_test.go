package nn

import (
	"math"
	"slices"
	"testing"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

const activationTol = 1e-9

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

// TestTanhForward verifies the elementwise forward against math.Tanh
// and that the output shape matches the input.
func TestTanhForward(t *testing.T) {
	th := &Tanh{}
	in := []float64{-2, -0.5, 0, 0.5, 2}
	x := tensor.NewTensor(ndarray.FromSlice(in, 5))

	y := th.Forward(x)

	if !slices.Equal(y.Shape(), []int{5}) {
		t.Errorf("output shape = %v, want [5]", y.Shape())
	}

	got := slices.Collect(y.Data().All())

	want := make([]float64, len(in))
	for i, v := range in {
		want[i] = math.Tanh(v)
	}

	for i := range got {
		if math.Abs(got[i]-want[i]) > activationTol {
			t.Errorf("output[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestTanhParametersEmpty: a parameter-less module must produce zero
// parameters.
func TestTanhParametersEmpty(t *testing.T) {
	th := &Tanh{}
	if got := th.Parameters(); len(got) != 0 {
		t.Errorf("Parameters() length = %d, want 0", len(got))
	}
}

// TestTanhBackwardPopulatesGrad runs a Sum(Tanh(x)) loss backwards and
// checks the leaf gradient against the closed-form 1 - tanh²(x).
func TestTanhBackwardPopulatesGrad(t *testing.T) {
	th := &Tanh{}
	in := []float64{-1.5, -0.1, 0.3, 1.0}
	x := tensor.NewLeaf(ndarray.FromSlice(in, 4))

	loss := tensor.Sum(th.Forward(x), nil, true)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatalf("x.Grad() is nil after Backward")
	}

	got := slices.Collect(x.Grad().All())

	want := make([]float64, len(in))
	for i, v := range in {
		want[i] = 1 - math.Tanh(v)*math.Tanh(v)
	}

	for i := range got {
		if math.Abs(got[i]-want[i]) > activationTol {
			t.Errorf("x.Grad()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestSigmoidForward verifies the elementwise forward against the
// closed-form 1 / (1 + exp(-x)) and that the output shape matches the
// input.
func TestSigmoidForward(t *testing.T) {
	s := &Sigmoid{}
	in := []float64{-2, -0.5, 0, 0.5, 2}
	x := tensor.NewTensor(ndarray.FromSlice(in, 5))

	y := s.Forward(x)

	if !slices.Equal(y.Shape(), []int{5}) {
		t.Errorf("output shape = %v, want [5]", y.Shape())
	}

	got := slices.Collect(y.Data().All())

	want := make([]float64, len(in))
	for i, v := range in {
		want[i] = 1 / (1 + math.Exp(-v))
	}

	for i := range got {
		if math.Abs(got[i]-want[i]) > activationTol {
			t.Errorf("output[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestSigmoidParametersEmpty: a parameter-less module must produce zero
// parameters.
func TestSigmoidParametersEmpty(t *testing.T) {
	s := &Sigmoid{}
	if got := s.Parameters(); len(got) != 0 {
		t.Errorf("Parameters() length = %d, want 0", len(got))
	}
}

// TestSigmoidBackwardPopulatesGrad runs a Sum(Sigmoid(x)) loss backwards
// and checks the leaf gradient against the closed-form σ(x) * (1 - σ(x)).
func TestSigmoidBackwardPopulatesGrad(t *testing.T) {
	s := &Sigmoid{}
	in := []float64{-1.5, -0.1, 0.3, 1.0}
	x := tensor.NewLeaf(ndarray.FromSlice(in, 4))

	loss := tensor.Sum(s.Forward(x), nil, true)
	loss.Backward()

	if x.Grad() == nil {
		t.Fatalf("x.Grad() is nil after Backward")
	}

	got := slices.Collect(x.Grad().All())

	want := make([]float64, len(in))
	for i, v := range in {
		sig := 1 / (1 + math.Exp(-v))
		want[i] = sig * (1 - sig)
	}

	for i := range got {
		if math.Abs(got[i]-want[i]) > activationTol {
			t.Errorf("x.Grad()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
