package nn

import "harryxu.ca/goml/tensor"

type Module interface {
	Forward(x tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}