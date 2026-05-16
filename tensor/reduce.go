package tensor

import "slices"

// Sum reduces t along the given axes. If axes is nil, every axis is
// reduced. With keepDims=false the reduced axes are dropped from the
// output shape; otherwise they remain as size-1.
//
// The backward reshapes the upstream gradient back to t's rank, placing
// 1s in the reduced positions, and relies on accumulateGrad's broadcasting
// to spread the value across the original sizes.
//
//	∂out/∂t = 1 (broadcast over the reduced axes).
func Sum(t *Tensor, axes []int, keepDims bool) *Tensor {
	out := NewTensor(t.data.Sum(axes, keepDims))
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			tShape := slices.Clone(t.Shape())
			// Computes axes for reshaping
			if axes == nil {
				for i := range tShape {
					tShape[i] = 1
				}
			} else {
				for _, ax := range axes {
					tShape[ax] = 1
				}
			}
			t.accumulateGrad(out.grad.Reshape(tShape...))
		}
	}

	return out
}

// Mean reduces t along the given axes by averaging. If axes is nil,
// every axis is reduced. The backward is Sum's gradient scaled by 1/N,
// where N is the product of the original sizes of the reduced axes.
//
// The denominator must be computed before tShape is overwritten with
// 1s, since the rewrite destroys the dimensions we need to read.
//
//	∂out/∂t = 1/N (broadcast over the reduced axes).
func Mean(t *Tensor, axes []int, keepDims bool) *Tensor {
	out := NewTensor(t.data.Mean(axes, keepDims))
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			tShape := slices.Clone(t.Shape())

			// Compute denominator
			count := 1
			if axes == nil {
				count = t.data.Size()
			} else {
				for _, ax := range axes {
					count *= tShape[ax]
				}
			}

			// Computes axes for reshaping
			if axes == nil {
				for i := range tShape {
					tShape[i] = 1
				}
			} else {
				for _, ax := range axes {
					tShape[ax] = 1
				}
			}

			reshaped := out.grad.Reshape(tShape...)
			t.accumulateGrad(reshaped.Scale(1.0 / float64(count)))
		}
	}

	return out
}
