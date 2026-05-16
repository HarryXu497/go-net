package tensor

import (
	"slices"
	"testing"

	"harryxu.ca/goml/ndarray"
)

// TestConstructors locks in the contract that distinguishes the two
// constructors: NewTensor produces a non-grad intermediate, NewLeaf
// produces a grad-requiring parameter/input. Both store the data
// pointer as-is and leave grad nil until something writes to it.
func TestConstructors(t *testing.T) {
	data := ndarray.FromSlice([]float64{1, 2, 3, 4}, 2, 2)

	t.Run("NewTensor: no grad, fields wired up", func(t *testing.T) {
		nt := NewTensor(data)
		if nt.RequiresGrad() {
			t.Errorf("NewTensor must not require grad")
		}
		if nt.Data() != data {
			t.Errorf("Data() should return the same pointer that was passed in")
		}
		if nt.Grad() != nil {
			t.Errorf("Grad() should be nil for a freshly constructed tensor; got %v", nt.Grad())
		}
		if !slices.Equal(nt.Shape(), []int{2, 2}) {
			t.Errorf("Shape() = %v, want [2 2]", nt.Shape())
		}
	})

	t.Run("NewLeaf: requires grad", func(t *testing.T) {
		nl := NewLeaf(data)
		if !nl.RequiresGrad() {
			t.Errorf("NewLeaf must require grad")
		}
		if nl.Grad() != nil {
			t.Errorf("Grad() should still be nil before any backward; got %v", nl.Grad())
		}
	})
}

// TestZeroGrad exercises both paths: the lazy-allocation path on the
// first call and the in-place zero path on subsequent calls. The
// second path is the one training loops hit every step, so it must not
// re-allocate.
func TestZeroGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))

	if x.Grad() != nil {
		t.Fatalf("precondition: grad must start nil")
	}

	x.ZeroGrad()
	gradAfterFirst := x.Grad()
	if gradAfterFirst == nil {
		t.Fatalf("ZeroGrad must allocate when grad is nil")
	}
	if !slices.Equal(gradAfterFirst.Shape(), []int{3}) {
		t.Errorf("allocated shape = %v, want [3]", gradAfterFirst.Shape())
	}
	for i, v := range slices.Collect(gradAfterFirst.All()) {
		if v != 0 {
			t.Errorf("grad[%d] = %v, want 0", i, v)
		}
	}

	// Dirty the gradient, then verify ZeroGrad zeroes it in place
	// (same pointer, all zeros). Pointer identity is the load-bearing
	// check here -- if ZeroGrad reallocated, training-loop callers
	// holding references to params' grads would silently miss updates.
	x.accumulateGrad(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	x.ZeroGrad()
	if x.Grad() != gradAfterFirst {
		t.Errorf("ZeroGrad must zero in place; got new pointer")
	}
	for i, v := range slices.Collect(x.Grad().All()) {
		if v != 0 {
			t.Errorf("after ZeroGrad: grad[%d] = %v, want 0", i, v)
		}
	}
}

// TestAccumulateGradLazyAlloc verifies the two halves of accumulateGrad's
// contract: it allocates on first call, and reuses the buffer on
// subsequent calls (additive accumulation). The broadcast path (g
// smaller than t.grad) is exercised by every reduction backward, so
// only the same-shape path needs a direct test.
func TestAccumulateGradLazyAlloc(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))

	x.accumulateGrad(ndarray.FromSlice([]float64{10, 20, 30}, 3))
	if got := slices.Collect(x.Grad().All()); !slices.Equal(got, []float64{10, 20, 30}) {
		t.Errorf("first accumulate: got %v, want [10 20 30]", got)
	}

	x.accumulateGrad(ndarray.FromSlice([]float64{1, 2, 3}, 3))
	if got := slices.Collect(x.Grad().All()); !slices.Equal(got, []float64{11, 22, 33}) {
		t.Errorf("second accumulate: got %v, want [11 22 33]", got)
	}
}
