package optim

import (
	"slices"
	"testing"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

// TestSGDStepUpdatesParameters tests the formula: p.data -= lr * p.grad,
// elementwise.
func TestSGDStepUpdatesParameters(t *testing.T) {
	// Values chosen so the math is exact in float64: lr * grad lands on integers
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{10, 20, 30, 40}, 2, 2))
	p.ZeroGrad()
	p.Grad().AddInPlace(ndarray.FromSlice([]float64{10, 20, 30, 40}, 2, 2))

	optim := NewSGD([]*tensor.Tensor{p}, ConstantLR{Rate: 0.5})
	optim.Step()

	got := slices.Collect(p.Data().All())

	want := []float64{5, 10, 15, 20} // 10 - 0.5*10, 20 - 0.5*20, ...
	if !slices.Equal(got, want) {
		t.Errorf("after Step: got %v, want %v", got, want)
	}
}

// TestSGDStepSkipsNilGrad tests that a param with no grad must not
// crash and must not have its data touched.
func TestSGDStepSkipsNilGrad(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	// no ZeroGrad, no Backward -- p.Grad() is nil

	optim := NewSGD([]*tensor.Tensor{p}, ConstantLR{Rate: 0.1})
	optim.Step()

	got := slices.Collect(p.Data().All())

	want := []float64{1, 2, 3}
	if !slices.Equal(got, want) {
		t.Errorf("param with nil grad must be unchanged: got %v, want %v", got, want)
	}
}

// stepRecordingLR is a test-only schedule that returns the step number
// itself as the LR. Lets a test prove that SGD advances `step` between
// calls and feeds the result to LR before applying the update.
type stepRecordingLR struct{}

func (stepRecordingLR) LR(step int) float64 { return float64(step) }

// TestSGDUsesScheduleStep: with a schedule of LR(step) = step, the
// first Step() uses lr=0 (no change), the second uses lr=1, the third
// lr=2. After three calls a param starting at 0 with grad 1 should
// move by -(0+1+2) = -3. Catches a bug where SGD ignores its schedule
// or fails to advance the step counter.
func TestSGDUsesScheduleStep(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{0, 0}, 2))
	p.ZeroGrad()
	p.Grad().AddInPlace(ndarray.FromSlice([]float64{1, 1}, 2))

	optim := NewSGD([]*tensor.Tensor{p}, stepRecordingLR{})
	optim.Step()
	optim.Step()
	optim.Step()

	got := slices.Collect(p.Data().All())

	want := []float64{-3, -3} // 0 - (0+1+2)*1
	if !slices.Equal(got, want) {
		t.Errorf("after 3 Steps with LR(step)=step: got %v, want %v", got, want)
	}
}

// TestSGDZeroGrad allocates grads if missing and zeroes them in place.
// Pointer identity matters: subsequent training steps re-use the same
// grad buffer, so the optimizer must not replace it.
func TestSGDZeroGrad(t *testing.T) {
	p := tensor.NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	p.ZeroGrad()
	p.Grad().AddInPlace(ndarray.FromSlice([]float64{5, 5, 5}, 3))
	gradPtr := p.Grad()

	NewSGD([]*tensor.Tensor{p}, ConstantLR{Rate: 0.1}).ZeroGrad()

	if p.Grad() != gradPtr {
		t.Errorf("ZeroGrad must zero the existing grad buffer, not reallocate")
	}

	for i, v := range slices.Collect(p.Grad().All()) {
		if v != 0 {
			t.Errorf("grad[%d] = %v after ZeroGrad, want 0", i, v)
		}
	}
}
