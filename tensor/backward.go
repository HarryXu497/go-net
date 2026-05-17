package tensor

import "slices"

import "harryxu.ca/goml/ndarray"

// Backward computes gradients of t (treated as the scalar loss) w.r.t.
// every parameter reachable through the parents chain. Gradients
// accumulate into each leaf's grad: callers must ZeroGrad between
// training steps.
func (t *Tensor) Backward() {
	if !t.requiresGrad {
		return
	}

	order := topoSort(t)

	t.grad = ndarray.NewNDArray(t.Shape()...)
	t.grad.Fill(1)

	for _, v := range slices.Backward(order) {
		tensor := v

		if tensor.backward != nil {
			tensor.backward()
		}
	}
}

// topoSort returns the nodes reachable from root in post-order, so that
// every parent appears before its children. Iterating the result in
// reverse therefore processes children before their parents.
func topoSort(root *Tensor) []*Tensor {
	visited := make(map[*Tensor]bool)

	var order []*Tensor

	var visit func(*Tensor)

	visit = func(t *Tensor) {
		if visited[t] {
			return
		}

		visited[t] = true

		for _, parent := range t.parents {
			visit(parent)
		}

		order = append(order, t)
	}
	visit(root)

	return order
}
