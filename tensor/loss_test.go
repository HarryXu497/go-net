package tensor

import (
	"math"
	"testing"

	"harryxu.ca/gonet/ndarray"
)

// TestSoftmaxCrossEntropyKnownLoss pins the forward against a hand-computed
// case. For uniform logits [0, 0] the softmax is [0.5, 0.5], so the
// negative log-likelihood of either class is log(2).
func TestSoftmaxCrossEntropyKnownLoss(t *testing.T) {
	logits := NewTensor(ndarray.FromSlice([]float64{0, 0}, 1, 2))
	loss := SoftmaxCrossEntropy(logits, []int{0})

	got := loss.Data().Get([]int{0})

	want := math.Log(2)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("loss = %v, want log(2) = %v (|diff| = %g)", got, want, math.Abs(got-want))
	}
}

// TestSoftmaxCrossEntropyGrad runs the fused forward + backward against
// finite differences. If both formulas are right (the stable softmax in
// forward and the (softmax - oneHot)/batch in backward) the analytic
// gradient will match the numeric one across every logits entry.
//
// Targets cover three distinct class indices so a bug that hard-coded
// the wrong class (e.g. always subtracting 1 at column 0) surfaces.
func TestSoftmaxCrossEntropyGrad(t *testing.T) {
	logits := NewLeaf(ndarray.FromSlice([]float64{
		1.0, -0.5, 0.2, 0.3,
		-1.2, 0.8, 0.0, 0.5,
		0.4, 0.4, -0.7, 1.1,
	}, 3, 4))
	targets := []int{0, 2, 1}

	gradCheck(t, "SoftmaxCrossEntropy(logits, [0,2,1])", logits, func(x *Tensor) *Tensor {
		return SoftmaxCrossEntropy(x, targets)
	})
}

// TestSoftmaxCrossEntropyPanicOnBatchMismatch locks in the explicit
// shape check at the top of the op. Without it, downstream the loop
// would read targets[i] past the row count of logits and panic with a
// less-clear ndarray index error.
func TestSoftmaxCrossEntropyPanicOnBatchMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on len(targets) != batch, got nil")
		}
	}()

	logits := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4}, 2, 2))
	SoftmaxCrossEntropy(logits, []int{0, 1, 1})
}
