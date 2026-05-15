package ndarray

import (
	"iter"
	"math"
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
