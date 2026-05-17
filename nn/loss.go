package nn

import "harryxu.ca/gonet/tensor"

// CrossEntropyLoss returns the mean cross-entropy loss of softmax(logits)
// against integer class targets.
//
//   - logits: shape (batch, numClasses), typically the output of a
//     Linear final layer.
//   - targets: length batch, each entry an integer class index in
//     [0, numClasses) representing the ground-truth class.
//
// Returns a scalar tensor.
func CrossEntropyLoss(logits *tensor.Tensor, targets []int) *tensor.Tensor {
	return tensor.SoftmaxCrossEntropy(logits, targets)
}
