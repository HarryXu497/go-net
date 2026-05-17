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

// Tanh is a parameter-less module that applies tanh
// elementwise. Thin wrapper around tensor.Tanh.
type Tanh struct{}

// Forward computes an elementwise Tanh on all elements in x.
func (t *Tanh) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Tanh(x)
}

// Parameters returns an empty slice since Tanh has no
// trainable parameters.
func (t *Tanh) Parameters() []*tensor.Tensor {
	return nil
}

// Sigmoid is a parameter-less module that applies the sigmoid
// function elementwise. Thin wrapper around tensor.Sigmoid.
type Sigmoid struct{}

// Forward computes an elementwise Sigmoid on all elements in x.
func (s *Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Sigmoid(x)
}

// Parameters returns an empty slice since Sigmoid has no
// trainable parameters.
func (s *Sigmoid) Parameters() []*tensor.Tensor {
	return nil
}
