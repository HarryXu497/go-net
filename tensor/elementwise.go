package tensor

// Add returns elementwise a + b under numpy broadcasting rules. If either
// operand requires grad, the result requires grad and the backward closure
// un-broadcasts the upstream gradient back to each operand's original
// shape before accumulating.
//
//	∂out/∂a = 1, ∂out/∂b = 1.
func Add(a, b *Tensor) *Tensor {
	out := NewTensor(a.data.Add(b.data))
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a, b}

		aShape := a.Shape()
		bShape := b.Shape()
		out.backward = func() {
			if a.requiresGrad {
				a.accumulateGrad(unbroadcast(out.grad, aShape))
			}

			if b.requiresGrad {
				b.accumulateGrad(unbroadcast(out.grad, bShape))
			}
		}
	}

	return out
}

// Sub returns elementwise a - b. Broadcasting and gradient accumulation
// follow the same pattern as Add; the only difference is the sign flip on
// b's incoming gradient.
//
//	∂out/∂a = 1, ∂out/∂b = -1.
func Sub(a, b *Tensor) *Tensor {
	out := NewTensor(a.data.Sub(b.data))
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a, b}

		aShape := a.Shape()
		bShape := b.Shape()
		out.backward = func() {
			if a.requiresGrad {
				a.accumulateGrad(unbroadcast(out.grad, aShape))
			}

			if b.requiresGrad {
				b.accumulateGrad(unbroadcast(out.grad.Neg(), bShape))
			}
		}
	}

	return out
}

// Mul returns elementwise a * b. Each operand's gradient depends on the
// other operand's forward value, so the backward closure reads both
// parents' data on every invocation.
//
//	∂out/∂a = b, ∂out/∂b = a.
func Mul(a, b *Tensor) *Tensor {
	out := NewTensor(a.data.Mul(b.data))
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a, b}

		aShape := a.Shape()
		bShape := b.Shape()
		out.backward = func() {
			if a.requiresGrad {
				a.accumulateGrad(unbroadcast(out.grad.Mul(b.data), aShape))
			}

			if b.requiresGrad {
				b.accumulateGrad(unbroadcast(out.grad.Mul(a.data), bShape))
			}
		}
	}

	return out
}

// Div returns elementwise a / b. The backward for b reuses the forward
// output to avoid a redundant division: -g * a / b^2 == -g * out / b.
// Callers are responsible for ensuring b is non-zero; zeros propagate as
// +/-Inf or NaN through both forward and backward.
//
//	∂out/∂a = 1/b, ∂out/∂b = -a/b^2.
func Div(a, b *Tensor) *Tensor {
	out := NewTensor(a.data.Div(b.data))
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a, b}

		aShape := a.Shape()
		bShape := b.Shape()
		out.backward = func() {
			if a.requiresGrad {
				a.accumulateGrad(unbroadcast(out.grad.Div(b.data), aShape))
			}

			if b.requiresGrad {
				derivativeB := out.grad.Mul(out.data).Div(b.data).Neg()
				b.accumulateGrad(unbroadcast(derivativeB, bShape))
			}
		}
	}

	return out
}

// Neg returns -t elementwise.
//
//	∂out/∂t = -1.
func Neg(t *Tensor) *Tensor {
	out := NewTensor(t.data.Neg())
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			t.accumulateGrad(out.grad.Neg())
		}
	}

	return out
}

// Exp returns exp(t) elementwise. The backward reuses the forward output
// since d(exp(t))/dt = exp(t).
//
//	∂out/∂t = out.
func Exp(t *Tensor) *Tensor {
	out := NewTensor(t.data.Exp())
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			t.accumulateGrad(out.grad.Mul(out.data))
		}
	}

	return out
}

// Log returns the natural log of t elementwise. Callers are responsible
// for ensuring t > 0; non-positive entries propagate as -Inf or NaN.
//
//	∂out/∂t = 1/t.
func Log(t *Tensor) *Tensor {
	out := NewTensor(t.data.Log())
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			t.accumulateGrad(out.grad.Div(t.data))
		}
	}

	return out
}

// ReLU returns max(t, 0) elementwise. The gradient at exactly zero is
// defined as 0, matching the convention in PyTorch and JAX. gradCheck
// inputs must stay clear of 0 because central differences are unstable
// across the kink in the derivative.
//
//	∂out/∂t = 1 if t > 0 else 0.
func ReLU(t *Tensor) *Tensor {
	out := NewTensor(t.data.ReLU())
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			t.accumulateGrad(out.grad.Mul(t.data.IndicatorPositive()))
		}
	}

	return out
}
