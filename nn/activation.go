package nn

import "harryxu.ca/gonet/tensor"

// ReLU is a parameter-less module that applies the rectified linear
// unit elementwise. Thin wrapper around tensor.ReLU.
type ReLU struct{}

// Forward computes an elementwise ReLU on all elements in x.
func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.ReLU(x)
}

// Parameters returns an empty slice since ReLU has no
// trainable parameters.
func (r *ReLU) Parameters() []*tensor.Tensor {
	return nil
}
