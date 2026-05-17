package optim

import (
	"math"
	"slices"
	"testing"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

// TestAdamFirstStepMatchesClosedForm pins the formula end-to-end
// including bias correction. With m₀=v₀=0, β1=0.9, β2=0.999, ε=1e-8,
// lr=1e-3 and a single scalar parameter p=0 with grad g=1, one Step
// must move p to exactly -lr / (1 + ε), because on t=1 the bias
// correction makes m̂=g and √v̂=|g|.
func TestAdamFirstStepMatchesClosedForm(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{0}, 1))
	p.ZeroGrad()
	p.Grad().AddInPlace(ndarray.FromSlice([]float64{1}, 1))

	opt := NewAdamDefault([]*tensor.Tensor{p})
	opt.Step()

	got := p.Data().Get([]int{0})
	want := -DefaultLR / (1 + DefaultEps)

	if math.Abs(got-want) > 1e-15 {
		t.Errorf("after first Step: got %v, want %v (|diff|=%g)", got, want, math.Abs(got-want))
	}
}

// TestAdamFirstStepIsMagnitudeInvariant is Adam's signature property:
// on t=1, the parameter delta is approximately -lr·sign(g) regardless
// of |g|, because m̂/√v̂ ≈ g/|g|. Verify across three very different
// gradient magnitudes that the post-step value lands within a couple
// of percent of -lr.
func TestAdamFirstStepIsMagnitudeInvariant(t *testing.T) {
	cases := []struct {
		name string
		g    float64
	}{
		{"small grad", 0.01},
		{"unit grad", 1},
		{"large grad", 100},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := tensor.NewLeaf(ndarray.FromSlice([]float64{0}, 1))
			p.ZeroGrad()
			p.Grad().AddInPlace(ndarray.FromSlice([]float64{c.g}, 1))

			opt := NewAdamDefault([]*tensor.Tensor{p})
			opt.Step()

			got := p.Data().Get([]int{0})

			// Closed form: -lr · g / (|g| + ε), simplified to -lr·sign(g)/(1 + ε/|g|).
			want := -DefaultLR * c.g / (math.Abs(c.g) + DefaultEps)

			if math.Abs(got-want) > 1e-15 {
				t.Errorf("got %v, want %v (|diff|=%g)", got, want, math.Abs(got-want))
			}
		})
	}
}

// TestAdamMultipleParameters exercises two parameters of different
// shapes in one optimizer. Each must update independently with the
// correct m/v allocation for its own shape -- catches a bug where
// NewAdam mis-aligns the m/v slices or shares a single buffer.
func TestAdamMultipleParameters(t *testing.T) {
	p1 := tensor.NewLeaf(ndarray.FromSlice([]float64{0, 0, 0}, 3))
	p2 := tensor.NewLeaf(ndarray.FromSlice([]float64{0, 0, 0, 0, 0, 0}, 2, 3))

	p1.ZeroGrad()
	p2.ZeroGrad()
	p1.Grad().AddInPlace(ndarray.FromSlice([]float64{1, 1, 1}, 3))
	p2.Grad().AddInPlace(ndarray.FromSlice([]float64{1, 1, 1, 1, 1, 1}, 2, 3))

	opt := NewAdamDefault([]*tensor.Tensor{p1, p2})
	opt.Step()

	want := -DefaultLR / (1 + DefaultEps)

	for i, v := range slices.Collect(p1.Data().All()) {
		if math.Abs(v-want) > 1e-15 {
			t.Errorf("p1[%d] = %v, want %v", i, v, want)
		}
	}

	for i, v := range slices.Collect(p2.Data().All()) {
		if math.Abs(v-want) > 1e-15 {
			t.Errorf("p2[%d] = %v, want %v", i, v, want)
		}
	}
}

// TestAdamScalarParameter: a rank-0 parameter must round-trip through
// the optimizer without panicking on its empty shape. Also verifies
// the m/v allocation handles shape=().
func TestAdamScalarParameter(t *testing.T) {
	p := tensor.NewLeaf(ndarray.NewNDArray())
	p.Data().Set([]int{}, 0)
	p.ZeroGrad()
	p.Grad().Set([]int{}, 1)

	opt := NewAdamDefault([]*tensor.Tensor{p})
	opt.Step()

	got := p.Data().Get([]int{})

	want := -DefaultLR / (1 + DefaultEps)
	if math.Abs(got-want) > 1e-15 {
		t.Errorf("scalar param after Step: got %v, want %v", got, want)
	}
}

// TestAdamStepSkipsNilGrad: a parameter that wasn't reached by the
// backward pass (Grad() is nil) must be skipped, not crashed on. Its
// data, m, and v must all be untouched.
func TestAdamStepSkipsNilGrad(t *testing.T) {
	withGrad := tensor.NewLeaf(ndarray.FromSlice([]float64{0, 0}, 2))
	noGrad := tensor.NewLeaf(ndarray.FromSlice([]float64{5, 6, 7}, 3))

	withGrad.ZeroGrad()
	withGrad.Grad().AddInPlace(ndarray.FromSlice([]float64{1, 1}, 2))
	// noGrad has nil Grad

	opt := NewAdamDefault([]*tensor.Tensor{withGrad, noGrad})
	opt.Step()

	// noGrad's data must be untouched.
	if got := slices.Collect(noGrad.Data().All()); !slices.Equal(got, []float64{5, 6, 7}) {
		t.Errorf("noGrad data changed: %v", got)
	}

	// noGrad's m and v slots must still be the original zero buffers.
	for _, v := range slices.Collect(opt.m[1].All()) {
		if v != 0 {
			t.Errorf("noGrad's m mutated: %v", slices.Collect(opt.m[1].All()))
			break
		}
	}

	for _, v := range slices.Collect(opt.v[1].All()) {
		if v != 0 {
			t.Errorf("noGrad's v mutated: %v", slices.Collect(opt.v[1].All()))
			break
		}
	}
}

// TestAdamZeroGradPreservesMandV: ZeroGrad must reset p.grad but leave
// the optimizer's first/second-moment state alone. m and v persist
// across steps by design; resetting them would defeat Adam.
func TestAdamZeroGradPreservesMandV(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{0}, 1))
	p.ZeroGrad()
	p.Grad().AddInPlace(ndarray.FromSlice([]float64{1}, 1))

	opt := NewAdamDefault([]*tensor.Tensor{p})
	opt.Step()

	mBefore := opt.m[0].Get([]int{0})
	vBefore := opt.v[0].Get([]int{0})

	opt.ZeroGrad()

	if got := opt.m[0].Get([]int{0}); got != mBefore {
		t.Errorf("m[0] changed after ZeroGrad: was %v, now %v", mBefore, got)
	}

	if got := opt.v[0].Get([]int{0}); got != vBefore {
		t.Errorf("v[0] changed after ZeroGrad: was %v, now %v", vBefore, got)
	}

	if got := p.Grad().Get([]int{0}); got != 0 {
		t.Errorf("p.grad after ZeroGrad: got %v, want 0", got)
	}
}

// TestAdamConvergesOnSimpleQuadratic is the "does the whole thing work
// in a real loop" smoke test. Minimize f(x) = sum(x²) (gradient is 2x)
// starting from a non-zero x. After ~1000 steps Adam should drive x
// to within 1e-3 of zero.
func TestAdamConvergesOnSimpleQuadratic(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{1.0, -2.0, 3.0}, 3))

	opt := NewAdam([]*tensor.Tensor{p}, 0.1, DefaultBeta1, DefaultBeta2, DefaultEps)

	for range 1000 {
		opt.ZeroGrad()
		// grad of f(x) = sum(x²) is 2x.
		p.Grad().AddInPlace(p.Data().Scale(2))
		opt.Step()
	}

	for i, v := range slices.Collect(p.Data().All()) {
		if math.Abs(v) > 1e-3 {
			t.Errorf("after 1000 steps: x[%d] = %v, want |x[%d]| < 1e-3", i, v, i)
		}
	}
}
