package tensor

import "harryxu.ca/gonet/ndarray"

// Tensor is a node in the autograd computation graph. It pairs a forward
// data buffer with an optional gradient buffer and the closure required
// to populate parent gradients during the backward pass.
//
// Fields:
//   - data: forward values.
//   - grad: accumulated gradient, lazily allocated by accumulateGrad or
//     ZeroGrad; nil until something writes to it.
//   - parents: tensors whose forward values were consumed to produce
//     this one.
//   - backward: closure that reads out.grad and writes parent grads.
//     nil for leaves and for nodes whose subgraph has no grad.
//   - requiresGrad: true if and only if a backward pass should populate
//     this tensor's grad and walk back through its parents.
type Tensor struct {
	data         *ndarray.NDArray
	grad         *ndarray.NDArray
	backward     func()
	parents      []*Tensor
	requiresGrad bool
}

// NewTensor wraps data as a non-leaf tensor that does not require grad.
// Used by op constructors for intermediate results; the op then sets
// requiresGrad to true if any parent requires grad.
func NewTensor(data *ndarray.NDArray) *Tensor {
	return &Tensor{
		data:         data,
		requiresGrad: false,
	}
}

// NewLeaf wraps data as a leaf tensor that requires grad. Use this for
// model parameters and any input whose gradient is needed.
func NewLeaf(data *ndarray.NDArray) *Tensor {
	return &Tensor{
		data:         data,
		requiresGrad: true,
	}
}

// Data returns the underlying ndarray. The pointer is live: mutating
// the returned array mutates the tensor's forward values.
func (t *Tensor) Data() *ndarray.NDArray {
	return t.data
}

// Grad returns the accumulated gradient, or nil if no backward pass has
// touched this tensor yet.
func (t *Tensor) Grad() *ndarray.NDArray {
	return t.grad
}

// Shape returns the shape of the underlying ndarray.
func (t *Tensor) Shape() []int {
	return t.data.Shape()
}

// RequiresGrad reports whether this tensor participates in backward.
func (t *Tensor) RequiresGrad() bool {
	return t.requiresGrad
}

// ZeroGrad clears the gradient: if grad is nil it allocates a fresh
// zero-filled buffer of the tensor's shape; otherwise it zeroes in
// place.
func (t *Tensor) ZeroGrad() {
	if t.grad == nil {
		t.grad = ndarray.NewNDArray(t.Shape()...)
	} else {
		t.grad.Zero()
	}
}

// accumulateGrad adds g into this tensor's gradient, allocating a
// fresh zero buffer first if grad is still nil. g may have a smaller
// shape than t.grad; AddInPlace broadcasts the rhs up to the lhs.
func (t *Tensor) accumulateGrad(g *ndarray.NDArray) {
	if t.grad == nil {
		t.grad = ndarray.NewNDArray(t.Shape()...)
	}

	t.grad.AddInPlace(g)
}
