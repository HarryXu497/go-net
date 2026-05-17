package nn

import "harryxu.ca/gonet/tensor"

// Sequential composes a list of modules into a single Module, threading
// the forward signal through each in order.
//
// Implements the Module interface itself, so a Sequential can compose
// other Sequential layers.
type Sequential struct {
	modules []Module
}

// NewSequential wraps modules into a Sequential that runs them in the
// order given. The modules slice is held by reference; callers should
// not mutate it after construction.
func NewSequential(modules ...Module) *Sequential {
	return &Sequential{modules: modules}
}

// Forward runs each module's Forward in sequence, threading the output
// of one as the input of the next. An empty Sequential is the identity:
// it returns x unchanged.
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	current := x
	for _, module := range s.modules {
		current = module.Forward(current)
	}

	return current
}

// Parameters returns every child module's parameters concatenated in
// module order. Pointers are returned as-is (no copying) so the
// optimizer's in-place mutations land on the same tensors held by
// each module.
func (s *Sequential) Parameters() []*tensor.Tensor {
	var allParameters []*tensor.Tensor
	for _, module := range s.modules {
		allParameters = append(allParameters, module.Parameters()...)
	}

	return allParameters
}
