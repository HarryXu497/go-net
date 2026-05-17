package nn

import (
	"math/rand"
	"slices"
	"testing"

	"harryxu.ca/goml/ndarray"
	"harryxu.ca/goml/tensor"
)

// TestNewLinearShapesAndInit pins the post-construction state: weight
// shape, bias shape, He fired (W not all zeros) and Zeros fired (b
// exactly zeros). Bundled into one test because they share setup.
func TestNewLinearShapesAndInit(t *testing.T) {
	const inFeatures, outFeatures = 4, 3

	l := NewLinear(rand.New(rand.NewSource(1)), inFeatures, outFeatures, He(), Zeros())

	if !slices.Equal(l.W.Shape(), []int{inFeatures, outFeatures}) {
		t.Errorf("W shape = %v, want [%d %d]", l.W.Shape(), inFeatures, outFeatures)
	}

	if !slices.Equal(l.b.Shape(), []int{outFeatures}) {
		t.Errorf("b shape = %v, want [%d]", l.b.Shape(), outFeatures)
	}

	// He must have populated W with non-zero values.
	allZero := true

	for _, v := range slices.Collect(l.W.Data().All()) {
		if v != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Errorf("W is all zero; expected He init to populate")
	}

	// Zeros must leave b at zero.
	for i, v := range slices.Collect(l.b.Data().All()) {
		if v != 0 {
			t.Errorf("b[%d] = %g, want 0", i, v)
		}
	}
}

// TestLinearForwardShape verifies the input-to-output shape mapping.
// Catches a silent transposition of W (e.g. using (out, in) instead of
// (in, out)) -- the matmul would error or produce the wrong shape.
func TestLinearForwardShape(t *testing.T) {
	const batch, in, out = 5, 4, 3

	l := NewLinear(rand.New(rand.NewSource(1)), in, out, He(), Zeros())
	x := tensor.NewTensor(ndarray.NewNDArray(batch, in))

	y := l.Forward(x)
	if !slices.Equal(y.Shape(), []int{batch, out}) {
		t.Errorf("Forward output shape = %v, want [%d %d]", y.Shape(), batch, out)
	}
}

// TestLinearParameters: returns [W, b] in stable order, both
// requiresGrad, both pointing at the same tensors held by the layer.
// Pointer identity is the load-bearing assertion: the optimizer
// mutates these in place, so if Parameters() returned copies the
// updates wouldn't take effect.
func TestLinearParameters(t *testing.T) {
	l := NewLinear(rand.New(rand.NewSource(1)), 4, 3, He(), Zeros())
	params := l.Parameters()

	if len(params) != 2 {
		t.Fatalf("Parameters() length = %d, want 2", len(params))
	}

	if params[0] != l.W {
		t.Errorf("Parameters()[0] should be the same pointer as l.W")
	}

	if params[1] != l.b {
		t.Errorf("Parameters()[1] should be the same pointer as l.b")
	}

	for i, p := range params {
		if !p.RequiresGrad() {
			t.Errorf("Parameters()[%d].RequiresGrad() = false, want true", i)
		}
	}
}

// TestLinearBackwardPopulatesGrads is the smoke test that exercises
// the full ndarray -> tensor -> nn stack end-to-end. Forward through
// the layer, sum to a scalar loss, Backward, then assert both W.grad
// and b.grad are populated with the expected shapes. If this passes,
// the layer is wired into the autograd graph correctly.
func TestLinearBackwardPopulatesGrads(t *testing.T) {
	const batch, in, out = 5, 4, 3

	l := NewLinear(rand.New(rand.NewSource(1)), in, out, He(), Zeros())

	xData := ndarray.NewNDArray(batch, in)
	xData.FillNormal(rand.New(rand.NewSource(2)), 0, 1)
	x := tensor.NewTensor(xData)

	loss := tensor.Sum(l.Forward(x), nil, true)
	loss.Backward()

	if l.W.Grad() == nil {
		t.Fatalf("W.Grad() is nil after Backward")
	}

	if !slices.Equal(l.W.Grad().Shape(), []int{in, out}) {
		t.Errorf("W.Grad() shape = %v, want [%d %d]", l.W.Grad().Shape(), in, out)
	}

	if l.b.Grad() == nil {
		t.Fatalf("b.Grad() is nil after Backward")
	}

	if !slices.Equal(l.b.Grad().Shape(), []int{out}) {
		t.Errorf("b.Grad() shape = %v, want [%d]", l.b.Grad().Shape(), out)
	}
}

// TestNewLinearPanicsOnBadDims locks in the explicit dim check at the
// top of NewLinear -- a clear panic message is much friendlier than a
// downstream NewNDArray failure.
func TestNewLinearPanicsOnBadDims(t *testing.T) {
	cases := []struct {
		name string
		in   int
		out  int
	}{
		{"zero in", 0, 3},
		{"negative in", -1, 3},
		{"zero out", 4, 0},
		{"negative out", 4, -2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic, got nil")
				}
			}()

			NewLinear(rand.New(rand.NewSource(1)), tc.in, tc.out, He(), Zeros())
		})
	}
}
