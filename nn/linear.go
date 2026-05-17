package nn

import (
	"fmt"
	"math/rand"

	"harryxu.ca/gonet/tensor"
)

// Linear is a fully-connected layer: y = x @ W + b. W and b are leaf
// tensors that participate in backward and are returned by Parameters
// so the optimizer can update them.
//
// W has shape (inFeatures, outFeatures); b has shape (outFeatures,)
// and broadcasts across the batch axis in Forward.
type Linear struct {
	W, b *tensor.Tensor
}

// NewLinear constructs a Linear layer with weights filled by initW
// and biases filled by initB. Typical choice: initW = He() for layers
// feeding a ReLU, initW = Glorot() for layers feeding softmax/linear,
// initB = Zeros() in either case.
//
// rng is threaded through the initializers so a single seed produces
// reproducible weights for the entire network.
//
// Panics if either dimension is non-positive.
func NewLinear(rng *rand.Rand, inFeatures, outFeatures int, initW, initB Initializer) *Linear {
	if inFeatures <= 0 {
		panic(fmt.Sprintf("nn: number of input features must be greater than 0, got %d", inFeatures))
	}

	if outFeatures <= 0 {
		panic(fmt.Sprintf("nn: number of output features must be greater than 0, got %d", outFeatures))
	}

	return &Linear{
		W: tensor.NewLeaf(initW(rng, inFeatures, outFeatures)),
		b: tensor.NewLeaf(initB(rng, outFeatures)),
	}
}

// Forward computes x @ W + b. x is expected to have shape
// (batch, inFeatures); the result has shape (batch, outFeatures). The
// bias broadcasts across the batch axis; unbroadcast handles summing
// it back to (outFeatures,) in backward.
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Add(tensor.MatMul(x, l.W), l.b)
}

// Parameters returns W and b in stable order so optimizers can rely
// on the layout (e.g. for indexed momentum buffers).
func (l *Linear) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.W, l.b}
}
