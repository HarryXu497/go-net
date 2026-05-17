package tensor

import (
	"testing"

	"harryxu.ca/gonet/ndarray"
)

// TestAddGrad exercises Add across same-shape, leaf-on-rhs, and the two
// broadcasting patterns that unbroadcast must handle: lower-rank rhs and
// a size-1 axis on rhs. The leaf-on-rhs subtest catches asymmetric bugs
// in the backward (e.g. accumulating into the wrong parent).
func TestAddGrad(t *testing.T) {
	t.Run("same shape", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{6, 5, 4, 3, 2, 1}, 2, 3))

		gradCheck(t, "Add(x, b)", x, func(x *Tensor) *Tensor { return Add(x, b) })
	})

	t.Run("leaf on rhs", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{6, 5, 4, 3, 2, 1}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Add(a, x)", x, func(x *Tensor) *Tensor { return Add(a, x) })
	})

	t.Run("broadcast lower rank rhs", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 3))

		gradCheck(t, "Add(x, b) (2,3)+(3,)", x, func(x *Tensor) *Tensor { return Add(x, b) })
	})

	t.Run("broadcast leaf is lower rank", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 3))
		gradCheck(t, "Add(a, x) (2,3)+(3,)", x, func(x *Tensor) *Tensor { return Add(a, x) })
	})

	t.Run("broadcast size-1 axis", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 1, 3))

		gradCheck(t, "Add(x, b) (2,3)+(1,3)", x, func(x *Tensor) *Tensor { return Add(x, b) })
	})
}

// TestSubGrad mirrors TestAddGrad but the sign flip on b's gradient is
// the load-bearing case to verify -- it's the one detail that diverges
// from Add.
func TestSubGrad(t *testing.T) {
	t.Run("same shape, leaf on lhs", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{6, 5, 4, 3, 2, 1}, 2, 3))

		gradCheck(t, "Sub(x, b)", x, func(x *Tensor) *Tensor { return Sub(x, b) })
	})

	t.Run("same shape, leaf on rhs", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{6, 5, 4, 3, 2, 1}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		gradCheck(t, "Sub(a, x)", x, func(x *Tensor) *Tensor { return Sub(a, x) })
	})

	t.Run("broadcast, leaf on rhs (exercises sign flip + unbroadcast)", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 3))
		gradCheck(t, "Sub(a, x) (2,3)-(3,)", x, func(x *Tensor) *Tensor { return Sub(a, x) })
	})
}

// TestMulGrad: each operand's gradient depends on the other's value, so
// the same-shape and broadcast cases exercise different code paths in
// unbroadcast applied to a non-trivial intermediate (g * other).
func TestMulGrad(t *testing.T) {
	t.Run("same shape", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{0.5, -1, 2, 0.25, -2, 3}, 2, 3))

		gradCheck(t, "Mul(x, b)", x, func(x *Tensor) *Tensor { return Mul(x, b) })
	})

	t.Run("broadcast, leaf on lhs", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 3))

		gradCheck(t, "Mul(x, b) (2,3)*(3,)", x, func(x *Tensor) *Tensor { return Mul(x, b) })
	})

	t.Run("broadcast, leaf is lower rank", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{0.5, -0.5, 2}, 3))
		gradCheck(t, "Mul(a, x) (2,3)*(3,)", x, func(x *Tensor) *Tensor { return Mul(a, x) })
	})
}

// TestDivGrad: b values are bounded away from zero so the central
// difference stays finite. The leaf-on-rhs case is the one that
// exercises the more interesting -g * a / b^2 branch.
func TestDivGrad(t *testing.T) {
	t.Run("same shape, leaf on lhs (numerator)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{2, 4, -1, 0.5, -2, 3}, 2, 3))

		gradCheck(t, "Div(x, b)", x, func(x *Tensor) *Tensor { return Div(x, b) })
	})

	t.Run("same shape, leaf on rhs (denominator)", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{2, 4, -1, 0.5, -2, 3}, 2, 3))
		gradCheck(t, "Div(a, x)", x, func(x *Tensor) *Tensor { return Div(a, x) })
	})

	t.Run("broadcast, leaf on rhs", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{2, -1, 0.5}, 3))
		gradCheck(t, "Div(a, x) (2,3)/(3,)", x, func(x *Tensor) *Tensor { return Div(a, x) })
	})
}

// TestNegGrad: trivial -- gradient is -1 everywhere. Worth one check to
// pin the contract.
func TestNegGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{-2, -1, 0, 1, 2}, 5))
	gradCheck(t, "Neg(x)", x, Neg)
}

// TestExpGrad: magnitudes stay small so exp(x) and its derivative remain
// well within central-difference tolerance.
func TestExpGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{-1.0, -0.3, 0.0, 0.5, 1.2}, 5))
	gradCheck(t, "Exp(x)", x, Exp)
}

// TestLogGrad: strictly positive inputs, none too close to zero (so 1/x
// stays bounded and finite differences are stable).
func TestLogGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{0.5, 1.0, 1.5, 2.0, 3.0}, 5))
	gradCheck(t, "Log(x)", x, Log)
}

// TestReLUGrad: inputs are clearly positive or clearly negative -- no
// values near 0, since the kink in the derivative makes central
// differences unreliable there.
func TestReLUGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{-2.0, -0.5, 0.3, 1.0, 2.5}, 5))
	gradCheck(t, "ReLU(x)", x, ReLU)
}

// TestTanhGrad: moderate-magnitude inputs keep tanh well inside the
// saturating regime where 1 - tanh² is still differentiable enough for
// central differences to be stable. Mixes positive, negative, and zero
// (tanh is smooth at 0, unlike ReLU).
func TestTanhGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{-1.5, -0.4, 0.0, 0.7, 1.2}, 5))
	gradCheck(t, "Tanh(x)", x, Tanh)
}

// TestSigmoidGrad: inputs within (-2, 2) so σ stays away from 0 and 1
// and the σ(1-σ) derivative remains numerically well-conditioned for
// the central-difference comparison.
func TestSigmoidGrad(t *testing.T) {
	x := NewLeaf(ndarray.FromSlice([]float64{-1.8, -0.6, 0.0, 0.5, 1.4}, 5))
	gradCheck(t, "Sigmoid(x)", x, Sigmoid)
}

// TestRequiresGradPropagation locks in the contract that the output of a
// binary op requires grad iff at least one input does, and that the
// non-grad operand is never allocated a grad. Add stands in for all
// binary ops since they share the propagation logic.
func TestRequiresGradPropagation(t *testing.T) {
	t.Run("leaf + const -> requires grad, const stays grad-less", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{1, 2, 3}, 3))
		c := NewTensor(ndarray.FromSlice([]float64{4, 5, 6}, 3))
		out := Add(x, c)

		if !out.RequiresGrad() {
			t.Fatalf("Add(leaf, const) must require grad")
		}

		out.Backward()

		if x.Grad() == nil {
			t.Errorf("x.Grad() should be populated")
		}

		if c.Grad() != nil {
			t.Errorf("c.Grad() should remain nil; got %v", c.Grad())
		}
	})

	t.Run("unary on const -> no grad, no backward closure", func(t *testing.T) {
		c := NewTensor(ndarray.FromSlice([]float64{0.5, 1.0, 1.5}, 3))

		out := Log(c)
		if out.RequiresGrad() {
			t.Fatalf("Log(const) must not require grad")
		}

		out.Backward()

		if c.Grad() != nil {
			t.Errorf("c.Grad() should remain nil; got %v", c.Grad())
		}
	})
}
