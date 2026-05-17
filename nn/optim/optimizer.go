package optim

type Optimizer interface {
	Step()
	ZeroGrad()
}
