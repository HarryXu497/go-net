package optim

import (
	"math"

	"harryxu.ca/gonet/ndarray"
	"harryxu.ca/gonet/tensor"
)

// Default hyperparameters as recommended by Kingma & Ba (2014).
const (
	DefaultLR    = 1e-3
	DefaultBeta1 = 0.9
	DefaultBeta2 = 0.999
	DefaultEps   = 1e-8
)

// Adam is the Adaptive Moment Estimation optimizer. On each Step it maintains
// per-parameter exponentially-decaying averages of the gradient (m, the
// first moment) and the squared gradient (v, the second moment), then
// updates each parameter against a bias-corrected, normalized step:
//
//	m_t   = β1·m_{t-1} + (1 − β1)·g
//	v_t   = β2·v_{t-1} + (1 − β2)·g²
//	m̂_t  = m_t / (1 − β1^t)
//	v̂_t  = v_t / (1 − β2^t)
//	p    -= lr · m̂_t / (√v̂_t + ε)
//
type Adam struct {
	params                []*tensor.Tensor
	m, v                  []*ndarray.NDArray
	lr, beta1, beta2, eps float64
	step                  int
}

// NewAdam creates an Adam optimizer with a list of parameters and the four
// hyperparameters.
func NewAdam(params []*tensor.Tensor, lr, beta1, beta2, eps float64) *Adam {
	m := make([]*ndarray.NDArray, len(params))
	v := make([]*ndarray.NDArray, len(params))

	for i, p := range params {
		m[i] = ndarray.NewNDArray(p.Shape()...)
		v[i] = ndarray.NewNDArray(p.Shape()...)
	}

	return &Adam{
		params: params,
		m:      m,
		v:      v,
		lr:     lr,
		beta1:  beta1,
		beta2:  beta2,
		eps:    eps,
	}
}

// NewAdamDefault is NewAdam with the paper's recommended hyperparameters
// (lr=1e-3, β1=0.9, β2=0.999, ε=1e-8). Use when there is no
// specific reason to deviate.
func NewAdamDefault(params []*tensor.Tensor) *Adam {
	return NewAdam(
		params,
		DefaultLR,
		DefaultBeta1,
		DefaultBeta2,
		DefaultEps,
	)
}

// Step applies one Adam update to every parameter that has a gradient.
// The internal step counter advances first, so the very first call uses
// t=1 in the bias-correction terms.
//
// Parameters whose Grad() is nil are skipped.
func (a *Adam) Step() {
	a.step++
	t := a.step

	biasC1 := 1 - math.Pow(a.beta1, float64(t))
	biasC2 := 1 - math.Pow(a.beta2, float64(t))

	for i, p := range a.params {
		g := p.Grad()
		if g == nil {
			continue
		}

		a.m[i] = a.m[i].Scale(a.beta1).Add(g.Scale(1 - a.beta1))
		a.v[i] = a.v[i].Scale(a.beta2).Add(g.Mul(g).Scale(1 - a.beta2))

		mHat := a.m[i].Scale(1 / biasC1)
		vHat := a.v[i].Scale(1 / biasC2)

		denom := vHat.Sqrt().AddScalar(a.eps)
		update := mHat.Div(denom)

		p.Data().AxpyInPlace(-a.lr, update)
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
func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}
