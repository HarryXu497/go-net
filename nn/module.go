package nn

import "harryxu.ca/gonet/tensor"

type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}
