package nn

import "harryxu.ca/gonet/tensor"

// SGD is a stochastic gradient descent optimizer. On each Step it pulls
// every parameter in the direction opposite its accumulated gradient:
//
//	p.data -= lr * p.grad
//
// The learning rate comes from an LRSchedule queried once per Step
// with a monotonically increasing step counter, so schedules from a
// flat constant to exponential decay all plug in without changing the
// optimizer.
//
// The update bypasses autograd. Step writes directly into each
// parameter's backing ndarray via AxpyInPlace, so it does not extend
// the computation graph and is safe to call between Backward and the
// next forward pass.
type SGD struct {
	schedule LRSchedule
	params   []*tensor.Tensor
	step     int
}

// NewSGD wires an optimizer to a list of parameters and an LR
// schedule. Typically params is the return value of
// model.Parameters(); the pointers are held by reference, so Step's
// mutations land on the same tensors the model uses in its forward
// pass on the next iteration.
//
// The internal step counter starts at 0 and increments once per call
// to Step, so the very first update uses schedule.LR(0).
func NewSGD(params []*tensor.Tensor, schedule LRSchedule) *SGD {
	return &SGD{params: params, schedule: schedule}
}

// Step applies one optimizer update to every parameter that has a
// gradient. The current learning rate is read from the schedule
// before the step counter advances, so schedule.LR(0) drives the
// first call, schedule.LR(1) the second, and so on.
//
// Parameters whose Grad() is still nil are skipped, not panicked on.
// This keeps frozen branches and modules-not-used-this-step harmless.
func (s *SGD) Step() {
	lr := s.schedule.LR(s.step)

	s.step++
	for _, p := range s.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}

		p.Data().AxpyInPlace(-lr, grad)
	}
}

// ZeroGrad resets every parameter's gradient buffer to zero (allocating
// it first if it was still nil). Call this between training steps;
// without it, the next Backward would accumulate on top of the
// previous step's gradient instead of starting fresh.
//
// The grad buffer's identity is preserved -- existing pointers held
// elsewhere (e.g. by another consumer) keep pointing at the same
// underlying storage.
func (s *SGD) ZeroGrad() {
	for _, p := range s.params {
		p.ZeroGrad()
	}
}
