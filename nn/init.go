package nn

import (
	"math"
	"math/rand"

	"harryxu.ca/goml/ndarray"
)

// Initializer produces a fresh ndarray of the given shape, optionally
// using rng. It exists so that parameter-bearing modules (Linear, ...)
// can decouple weight construction from layer construction: callers
// pick He, Glorot, Zeros, or any custom strategy without the module
// caring.
//
// Each Initializer documents its own shape contract -- e.g. He treats
// shape[0] as fan_in, so it expects a rank >= 1 shape.
type Initializer func(rng *rand.Rand, shape ...int) *ndarray.NDArray

// He returns an initializer that samples from N(0, 2/fan_in), the
// canonical scheme for layers whose outputs feed a ReLU. The factor
// of 2 compensates for ReLU zeroing roughly half the activations and
// keeps signal variance stable through depth.
//
// fan_in is read from shape[0], so this works directly for Linear
// weights of shape (in, out). Higher-rank shapes (e.g. conv kernels)
// would need a different fan_in calculation. Panics if shape is empty.
func He() Initializer {
	return func(rng *rand.Rand, shape ...int) *ndarray.NDArray {
		if len(shape) < 1 {
			panic("nn: He initializer must have shape with at least 1 dim")
		}

		in := shape[0]
		stddev := math.Sqrt(2.0 / float64(in))
		a := ndarray.NewNDArray(shape...)
		a.FillNormal(rng, 0, stddev)

		return a
	}
}

// Glorot (a.k.a. Xavier) returns an initializer that samples from
// N(0, 2/(fan_in + fan_out)). Use for layers whose outputs feed a
// linear/softmax/tanh activation -- balances forward and backward
// signal variance when the activation is roughly linear at 0.
//
// fan_in and fan_out are read from shape[0] and shape[1], so this
// works directly for Linear weights of shape (in, out). Panics if
// shape has fewer than two dimensions.
func Glorot() Initializer {
	return func(rng *rand.Rand, shape ...int) *ndarray.NDArray {
		if len(shape) < 2 {
			panic("nn: Glorot initializer must have shape with at least 1 dim")
		}

		in, out := shape[0], shape[1]
		stddev := math.Sqrt(2.0 / float64(in+out))
		a := ndarray.NewNDArray(shape...)
		a.FillNormal(rng, 0, stddev)

		return a
	}
}

// Zeros returns an initializer that produces a fresh zero-filled
// ndarray of the given shape. Standard choice for biases (no symmetry
// to break, since the weights they're added to are already random).
func Zeros() Initializer {
	return func(_ *rand.Rand, shape ...int) *ndarray.NDArray {
		// Already zero-filled
		return ndarray.NewNDArray(shape...)
	}
}
