package tensor

// Reshape returns a view of t with the given shape. The total size of
// newShape must equal t.Size(); ndarray.Reshape enforces this and falls
// back to a Copy when t is non-contiguous. The backward closure
// reshapes the upstream gradient back to t's original shape -- a pure
// rewrite, no broadcasting or summing.
//
//	∂out/∂t = identity (shape-only transformation).
func Reshape(t *Tensor, newShape ...int) *Tensor {
	out := NewTensor(t.data.Reshape(newShape...))
	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		tShape := t.Shape()
		out.backward = func() {
			t.accumulateGrad(out.grad.Reshape(tShape...))
		}
	}

	return out
}

// inversePerm returns q such that perm[q[i]] = i -- the permutation that
// undoes perm. For perm=[2,0,1] the inverse is [1,2,0]; verify by
// composing: perm[q[0]]=perm[1]=0, perm[q[1]]=perm[2]=1, perm[q[2]]=perm[0]=2.
func inversePerm(perm []int) []int {
	q := make([]int, len(perm))
	for i, axis := range perm {
		q[axis] = i
	}

	return q
}

// Transpose permutes t's axes by perm. With no perm provided, the
// underlying ndarray reverses every axis. The backward applies the
// inverse permutation to the upstream gradient; for the default
// reverse-all case, the inverse is also reverse-all, so the same
// no-args Transpose call works on the gradient too.
//
//	∂out/∂t = identity (axis-permutation only).
func Transpose(t *Tensor, perm ...int) *Tensor {
	out := NewTensor(t.data.Transpose(perm...))

	if t.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{t}
		out.backward = func() {
			t.accumulateGrad(out.grad.Transpose(inversePerm(perm)...))
		}
	}

	return out
}
