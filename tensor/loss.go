package tensor

import (
	"fmt"

	"harryxu.ca/goml/ndarray"
)

// SoftmaxCrossEntropy returns the mean cross-entropy loss of softmax(logits)
// against integer class targets, fused for numerical stability and a
// simplified gradient.
//
//   - logits: shape (batch, numClasses). The only tensor in the graph; if
//     it requires grad, the result requires grad too.
//   - targets: length batch. Each entry is an integer class index in
//     [0, numClasses).
//
// The returned tensor is a scalar (shape [1]) equal to
//
//	-mean_i log(softmax(logits[i])[targets[i]])
//
// Forward stability: the per-row max is subtracted from logits before
// exp, which keeps exp inputs in (-inf, 0] and prevents overflow.
//
// Backward uses the closed-form fused gradient:
//
//	dLogits = (softmax - oneHot(targets)) / batch
//
// Computed in-place by copying softmax and subtracting 1 at each
// [i, targets[i]] slot, then scaling by upstream / batch.
func SoftmaxCrossEntropy(logits *Tensor, targets []int) *Tensor {
	numTargets := len(targets)
	if numTargets != logits.Shape()[0] {
		panic(fmt.Sprintf("tensor: targets has length %d, expected batch size %d", numTargets, logits.Shape()[0]))
	}

	classAxis := []int{1}
	// Compute softmax shift for numerical stability
	shift := logits.data.Max(classAxis, true)
	shifted := logits.data.Sub(shift)

	expShifted := shifted.Exp()
	sumExp := expShifted.Sum(classAxis, true)
	softmax := expShifted.Div(sumExp)
	logSoftmax := shifted.Sub(sumExp.Log())

	totalLoss := 0.0

	for i := range targets {
		totalLoss += -logSoftmax.Get([]int{i, targets[i]})
	}

	outData := ndarray.NewNDArray(1)
	outData.Set([]int{0}, totalLoss/float64(numTargets))
	out := NewTensor(outData)

	if logits.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{logits}
		out.backward = func() {
			// Scalar
			gradient := out.grad.Get([]int{0})
			derivativeLogits := softmax.Copy()

			for i, target := range targets {
				v := derivativeLogits.Get([]int{i, target}) - 1
				derivativeLogits.Set([]int{i, target}, v)
			}

			derivativeLogits = derivativeLogits.Scale(gradient / float64(numTargets))
			logits.accumulateGrad(derivativeLogits)
		}
	}

	return out
}
