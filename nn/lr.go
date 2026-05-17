package nn

import "math"

// LRSchedule maps a step counter to a learning rate. SGD calls LR once
// per Step() with a monotonically increasing step value, so schedules
// can be stateless functions of step alone -- no internal mutation
// needed.
//
// A schedule that needs per-epoch decay should divide the incoming
// step by batches-per-epoch internally; the optimizer counts batches,
// not epochs.
type LRSchedule interface {
	LR(step int) float64
}

// ConstantLR returns Rate at every step.
type ConstantLR struct {
	Rate float64
}

func (c ConstantLR) LR(_ int) float64 {
	return c.Rate
}

// ExpDecayLR multiplies the learning rate by DecayRate every step:
//
//	lr(step) = Initial * DecayRate^step
type ExpDecayLR struct {
	Initial   float64
	DecayRate float64
}

func (e ExpDecayLR) LR(step int) float64 {
	return e.Initial * math.Pow(e.DecayRate, float64(step))
}

// StepDecayLR keeps the rate flat for DecayEvery steps, then multiplies
// it by DecayFactor:
//
//	lr(step) = Initial * DecayFactor^floor(step / DecayEvery)
type StepDecayLR struct {
	Initial     float64
	DecayFactor float64
	DecayEvery  int
}

func (s StepDecayLR) LR(step int) float64 {
	return s.Initial * math.Pow(s.DecayFactor, float64(step/s.DecayEvery))
}
