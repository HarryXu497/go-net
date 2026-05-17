package nn

import (
	"math/rand"
	"slices"
	"testing"

	"harryxu.ca/goml/ndarray"
	"harryxu.ca/goml/tensor"
)

// TestSequentialEmpty pins the identity contract: zero modules means
// Forward returns its input unchanged and Parameters is empty.
func TestSequentialEmpty(t *testing.T) {
	s := NewSequential()
	x := tensor.NewTensor(ndarray.FromSlice([]float64{1, 2, 3}, 3))

	y := s.Forward(x)
	if y != x {
		t.Errorf("empty Sequential should return input pointer unchanged")
	}

	if got := s.Parameters(); len(got) != 0 {
		t.Errorf("empty Sequential Parameters() length = %d, want 0", len(got))
	}
}

// TestSequentialForwardShape: a (784 -> 128) -> ReLU -> (128 -> 10)
// chain produces the expected output shape.
func TestSequentialForwardShape(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	s := NewSequential(
		NewLinear(rng, 784, 128, He(), Zeros()),
		&ReLU{},
		NewLinear(rng, 128, 10, Glorot(), Zeros()),
	)

	const batch = 4

	x := tensor.NewTensor(ndarray.NewNDArray(batch, 784))

	y := s.Forward(x)
	if !slices.Equal(y.Shape(), []int{batch, 10}) {
		t.Errorf("output shape = %v, want [%d 10]", y.Shape(), batch)
	}
}

// TestSequentialParameters: parameters are concatenated in module
// order, ReLU contributes none, and the returned pointers are the
// SAME tensors held by each module.
func TestSequentialParameters(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	l1 := NewLinear(rng, 4, 3, He(), Zeros())
	l2 := NewLinear(rng, 3, 2, Glorot(), Zeros())
	s := NewSequential(l1, &ReLU{}, l2)

	got := s.Parameters()
	want := []*tensor.Tensor{l1.W, l1.b, l2.W, l2.b}

	if len(got) != len(want) {
		t.Fatalf("Parameters() length = %d, want %d", len(got), len(want))
	}

	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Parameters()[%d] pointer mismatch", i)
		}
	}
}

// TestSequentialBackward tests that after Backward, every
// parameter inside the chain must have a populated grad of the right
// shape. If any backward closure is broken, the corresponding grad
// stays nil or wrong-shaped.
func TestSequentialBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	l1 := NewLinear(rng, 4, 3, He(), Zeros())
	l2 := NewLinear(rng, 3, 2, Glorot(), Zeros())
	s := NewSequential(l1, &ReLU{}, l2)

	xData := ndarray.NewNDArray(5, 4)
	xData.FillNormal(rand.New(rand.NewSource(2)), 0, 1)
	x := tensor.NewTensor(xData)

	loss := tensor.Sum(s.Forward(x), nil, true)
	loss.Backward()

	checks := []struct {
		name string
		t    *tensor.Tensor
		want []int
	}{
		{"l1.W", l1.W, []int{4, 3}},
		{"l1.b", l1.b, []int{3}},
		{"l2.W", l2.W, []int{3, 2}},
		{"l2.b", l2.b, []int{2}},
	}
	for _, c := range checks {
		if c.t.Grad() == nil {
			t.Errorf("%s.Grad() is nil after Backward", c.name)
			continue
		}

		if !slices.Equal(c.t.Grad().Shape(), c.want) {
			t.Errorf("%s.Grad() shape = %v, want %v", c.name, c.t.Grad().Shape(), c.want)
		}
	}
}
