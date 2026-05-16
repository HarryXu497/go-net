package tensor

import (
	"testing"

	"harryxu.ca/goml/ndarray"
)

// TestMatMulGrad verifies both adjoints (g @ b^T and a^T @ g) via finite
// differences. Shapes are rectangular (M != K != N) so a mistakenly
// transposed dim would surface as either a shape mismatch or a wrong
// gradient.
func TestMatMulGrad(t *testing.T) {
	t.Run("leaf on lhs (2,3) @ (3,4)", func(t *testing.T) {
		x := NewLeaf(ndarray.FromSlice([]float64{
			1, 2, 3,
			4, 5, 6,
		}, 2, 3))
		b := NewTensor(ndarray.FromSlice([]float64{
			0.5, -1, 2, 0.25,
			-2, 3, 0.1, -0.5,
			1, 0.5, -1.5, 2,
		}, 3, 4))
		gradCheck(t, "MatMul(x, b)", x, func(x *Tensor) *Tensor { return MatMul(x, b) })
	})

	t.Run("leaf on rhs (2,3) @ (3,4)", func(t *testing.T) {
		a := NewTensor(ndarray.FromSlice([]float64{
			1, 2, 3,
			4, 5, 6,
		}, 2, 3))
		x := NewLeaf(ndarray.FromSlice([]float64{
			0.5, -1, 2, 0.25,
			-2, 3, 0.1, -0.5,
			1, 0.5, -1.5, 2,
		}, 3, 4))
		gradCheck(t, "MatMul(a, x)", x, func(x *Tensor) *Tensor { return MatMul(a, x) })
	})
}
