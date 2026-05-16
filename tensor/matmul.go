package tensor

// MatMul returns the matrix product a @ b. For 2D inputs of shape (M, K)
// and (K, N) the result has shape (M, N). Both operands must be rank-2;
// batched matmul (broadcasting over leading axes) is not supported.
//
//	∂out/∂a = g @ b^T
//	∂out/∂b = a^T @ g
func MatMul(a, b *Tensor) *Tensor {
	out := NewTensor(a.data.Matmul(b.data))
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a, b}
		out.backward = func() {
			if a.requiresGrad {
				a.accumulateGrad(out.grad.Matmul(b.data.Transpose()))
			}
			if b.requiresGrad {
				b.accumulateGrad(a.data.Transpose().Matmul(out.grad))
			}
		}
	}
	return out
}