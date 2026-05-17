package nn

import (
	"math"
	"testing"

	"harryxu.ca/goml/ndarray"
	"harryxu.ca/goml/tensor"
)

// TestCrossEntropyLossKnownValue confirms the nn-layer wrapper produces
// the same hand-computed loss as the underlying tensor op.
func TestCrossEntropyLossKnownValue(t *testing.T) {
	logits := tensor.NewTensor(ndarray.FromSlice([]float64{0, 0}, 1, 2))
	loss := CrossEntropyLoss(logits, []int{0})

	got := loss.Data().Get([]int{0})
	want := math.Log(2)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("loss = %v, want log(2) = %v (|diff| = %g)", got, want, math.Abs(got-want))
	}
}

// TestCrossEntropyLossBackward tests that with a grad-requiring logits
// tensor, Backward must populate logits.Grad() with the correct shape.
func TestCrossEntropyLossBackward(t *testing.T) {
	logits := tensor.NewLeaf(ndarray.FromSlice([]float64{
		1.0, -0.5, 0.2,
		-1.2, 0.8, 0.0,
	}, 2, 3))

	loss := CrossEntropyLoss(logits, []int{0, 2})
	loss.Backward()

	if logits.Grad() == nil {
		t.Fatalf("logits.Grad() is nil after Backward")
	}
	if got, want := logits.Grad().Shape(), []int{2, 3}; got[0] != want[0] || got[1] != want[1] {
		t.Errorf("logits.Grad() shape = %v, want %v", got, want)
	}
}
