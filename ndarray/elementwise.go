package ndarray

import (
	"fmt"
	"iter"
	"math"
	"slices"
)

// unaryOp applies op elementwise to a and returns a fresh contiguous
// tensor with the same shape. The receiver is unchanged.
//
// unaryOp walks a in shape order and writes results sequentially into a
// fresh row-major buffer, so it is correct for any view of a.
func (a *NDArray) unaryOp(op func(float64) float64) *NDArray {
	out := NewNDArray(a.shape...)

	i := 0
	for _, aOff := range a.indicesAndOffsets() {
		out.data[i] = op(a.data[aOff])
		i++
	}

	return out
}

// Neg returns a fresh contiguous tensor whose elements are the negation
// of a's elements. The receiver is unchanged.
func (a *NDArray) Neg() *NDArray {
	return a.unaryOp(func(x float64) float64 { return -x })
}

// Exp returns a fresh contiguous tensor whose elements are e raised to
// the corresponding element of a (the natural exponential). The receiver
// is unchanged.
//
// Inputs near +∞ overflow to +Inf and very negative inputs underflow to 0,
// per IEEE 754. Used in softmax.
func (a *NDArray) Exp() *NDArray {
	return a.unaryOp(math.Exp)
}

// Log returns a fresh contiguous tensor whose elements are the natural
// logarithm (base e) of a's elements. The receiver is unchanged.
//
// log(0) is -Inf and log(x) for x < 0 is NaN, per IEEE 754; neither
// panics. Used in cross-entropy loss.
func (a *NDArray) Log() *NDArray {
	return a.unaryOp(math.Log)
}

// ReLU returns a fresh contiguous tensor whose elements are max(a, 0).
// The receiver is unchanged.
//
// ReLU is the standard rectified-linear nonlinearity for MLPs: positive
// inputs pass through, negative inputs are clamped to zero.
func (a *NDArray) ReLU() *NDArray {
	return a.unaryOp(func(x float64) float64 { return max(x, 0) })
}

// IndicatorPositive returns a fresh contiguous tensor whose elements
// are 1.0 where the corresponding element of a is strictly greater
// than 0, and 0.0 otherwise. The receiver is unchanged.
//
// This is the elementwise derivative of ReLU and is intended as the
// mask factor in ReLU's backward pass. The value at 0 is taken to be
// 0, matching PyTorch's convention for the ReLU subgradient.
func (a *NDArray) IndicatorPositive() *NDArray {
	return a.unaryOp(func(x float64) float64 {
		if x > 0 {
			return 1
		}

		return 0
	})
}

// binaryOp applies op elementwise to a and b under numpy's broadcasting
// rules and returns a fresh contiguous tensor with the result.
//
// The output shape is broadcastShape(a.shape, b.shape): the two shapes are
// right-aligned, the shorter is padded with size-1 axes on the left, and
// any size-1 axis is stretched against the other side. op is invoked once
// per output cell with the corresponding broadcast values from a and b.
//
// It panics if the shapes are not broadcast-compatible (delegated to
// broadcastShape).
func (a *NDArray) binaryOp(b *NDArray, op func(float64, float64) float64) *NDArray {
	outShape := broadcastShape(a.shape, b.shape)
	aView := a.BroadcastTo(outShape...)
	bView := b.BroadcastTo(outShape...)
	out := NewNDArray(outShape...)

	// All three views share outShape and are walked in shape order.
	aPull, aStop := iter.Pull2(aView.indicesAndOffsets())
	defer aStop()

	bPull, bStop := iter.Pull2(bView.indicesAndOffsets())
	defer bStop()

	for _, outOff := range out.indicesAndOffsets() {
		_, aOff, _ := aPull()
		_, bOff, _ := bPull()
		out.data[outOff] = op(aView.data[aOff], bView.data[bOff])
	}

	return out
}

// binaryOpInPlace applies op elementwise and writes the result back into
// a in place: a[i] = op(a[i], b[i]). No allocation is performed and the
// receiver's shape is unchanged.
//
// b is broadcast up to a.shape. in particular b cannot be
// strictly larger than a on any axis, since the result must fit into a.
// Panics if the shapes are incompatible.
//
// Aliasing: callers must not pass a b that aliases a's backing buffer
// in a different iteration order (e.g. a transposed view of a). The
// loop reads from b after writing to a, so reordered aliases produce
// undefined results. b == a is safe because both iterators advance
// through the buffer in the same order.
func (a *NDArray) binaryOpInPlace(b *NDArray, op func(float64, float64) float64) {
	bView := b.BroadcastTo(a.shape...)

	bPull, bStop := iter.Pull2(bView.indicesAndOffsets())
	defer bStop()

	for _, aOff := range a.indicesAndOffsets() {
		_, bOff, _ := bPull()
		a.data[aOff] = op(a.data[aOff], bView.data[bOff])
	}
}

// Add returns a fresh contiguous tensor whose elements are a + b, combined
// under numpy's broadcasting rules. The receivers are unchanged. Add
// panics if a.shape and b.shape are not broadcast-compatible.
//
// Examples:
//
//	x.Add(y)   // same-shape elementwise sum
//	logits.Add(bias)   // (batch, 10) + (10,) — bias stretches across the batch axis
func (a *NDArray) Add(b *NDArray) *NDArray {
	return a.binaryOp(b, func(x, y float64) float64 { return x + y })
}

// AddInPlace adds b into a in place: a = a + b. The receiver is
// modified and no allocation is performed.
//
// b is broadcast up to a.shape. AddInPlace panics if b cannot be broadcast
// to a.shape; in particular if b is strictly larger than a on any axis.
//
// Callers must not pass a b that aliases a's backing buffer in a
// different iteration order; b == a is the only safe self-aliasing case.
func (a *NDArray) AddInPlace(b *NDArray) {
	a.binaryOpInPlace(b, func(x, y float64) float64 { return x + y })
}

// Sub returns a fresh contiguous tensor whose elements are a - b, combined
// under numpy's broadcasting rules. The receivers are unchanged. Sub
// panics if a.shape and b.shape are not broadcast-compatible.
//
// Note that Sub is not commutative: a.Sub(b) and b.Sub(a) differ in sign.
func (a *NDArray) Sub(b *NDArray) *NDArray {
	return a.binaryOp(b, func(x, y float64) float64 { return x - y })
}

// Mul returns a fresh contiguous tensor whose elements are a * b
// (Hadamard product), combined under numpy's broadcasting rules. The
// receivers are unchanged. Mul panics if a.shape and b.shape are not
// broadcast-compatible.
//
// Mul is elementwise multiplication, not matrix multiplication; for the
// latter use Matmul.
func (a *NDArray) Mul(b *NDArray) *NDArray {
	return a.binaryOp(b, func(x, y float64) float64 { return x * y })
}

// Div returns a fresh contiguous tensor whose elements are a / b,
// combined under numpy's broadcasting rules. The receivers are
// unchanged. Div panics if a.shape and b.shape are not
// broadcast-compatible.
//
// Division by zero is not a panic: results follow IEEE 754 semantics
// (±Inf for nonzero / 0, NaN for 0 / 0). Callers that need different
// behavior should guard their inputs.
func (a *NDArray) Div(b *NDArray) *NDArray {
	return a.binaryOp(b, func(x, y float64) float64 { return x / y })
}

// Fill sets every element of a's view to a scalar value, in place. The receiver is
// modified and no allocation is performed.
//
// Because views share their backing data, this also affects
// every other view of the same tensor.
func (a *NDArray) Fill(value float64) {
	if a.IsContiguous() && a.offset == 0 && len(a.data) == a.Size() {
		for i := range a.data {
			a.data[i] = value
		}

		return
	}

	for _, offset := range a.indicesAndOffsets() {
		a.data[offset] = value
	}
}

// Zero sets every element of a's view to 0, in place. The receiver is
// modified and no allocation is performed.
//
// Because views share their backing data, zeroing a view also zeros
// every other view of the same tensor. Zeroing a broadcast view
// (which has stride 0 on stretched axes) zeros the underlying
// unbroadcast data, which is rarely what the caller wants.
func (a *NDArray) Zero() {
	a.Fill(0)
}

// Scale returns a fresh contiguous tensor whose elements are c * a's
// elements. The receiver is unchanged.
//
// This is the scalar-broadcast form of Mul: c is a plain float64, so
// no 1-element ndarray needs to be allocated to carry it. Useful for
// gradient scaling (e.g. dividing by N in a Mean reduction) and for
// SGD parameter updates.
//
// Special values follow IEEE 754: 0 * ±Inf = NaN, c * NaN = NaN.
func (a *NDArray) Scale(c float64) *NDArray {
	return a.unaryOp(func(x float64) float64 { return c * x })
}

// AxpyInPlace overwrites a with a + alpha*x, elementwise.
// Gradient descent uses this with alpha = -lr to step parameters
// against their gradient without allocating additional memory.
//
// Requires a.shape == x.shape. Iterates the backing slices directly,
// so a and x must both be contiguous with offset 0.
//
// Panics on shape mismatch.
func (a *NDArray) AxpyInPlace(alpha float64, x *NDArray) {
	if !slices.Equal(a.shape, x.shape) {
		panic(fmt.Sprintf("ndarray: AxpyInPlace shape mismatch: %v vs %v", a.shape, x.shape))
	}

	if !a.IsContiguous() || a.offset != 0 || len(a.data) != a.Size() {
		panic("ndarray: AxpyInPlace requires a contiguous receiver with offset 0")
	}

	if !x.IsContiguous() || x.offset != 0 || len(x.data) != x.Size() {
		panic("ndarray: AxpyInPlace requires a contiguous argument with offset 0")
	}

	for i := range a.data {
		a.data[i] += alpha * x.data[i]
	}
}
