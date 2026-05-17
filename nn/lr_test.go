package nn

import (
	"math"
	"testing"
)

// TestConstantLR: same rate at every step, including step 0 and a
// large step.
func TestConstantLR(t *testing.T) {
	c := ConstantLR{Rate: 0.01}
	for _, step := range []int{0, 1, 100, 9999} {
		if got := c.LR(step); got != 0.01 {
			t.Errorf("ConstantLR.LR(%d) = %v, want 0.01", step, got)
		}
	}
}

// TestExpDecayLR: lr(0) = Initial, lr(n) = Initial * DecayRate^n.
func TestExpDecayLR(t *testing.T) {
	e := ExpDecayLR{Initial: 1.0, DecayRate: 0.9}
	cases := []struct {
		step int
		want float64
	}{
		{0, 1.0},
		{1, 0.9},
		{2, 0.81},
		{10, math.Pow(0.9, 10)},
	}
	for _, c := range cases {
		got := e.LR(c.step)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("ExpDecayLR.LR(%d) = %v, want %v", c.step, got, c.want)
		}
	}
}

// TestStepDecayLR: rate is piecewise constant. step=0..DecayEvery-1
// stays at Initial; at DecayEvery it drops by DecayFactor; etc.
func TestStepDecayLR(t *testing.T) {
	s := StepDecayLR{Initial: 1.0, DecayFactor: 0.5, DecayEvery: 10}
	cases := []struct {
		step int
		want float64
	}{
		{0, 1.0},
		{9, 1.0},   // still in the first bucket
		{10, 0.5},  // first drop
		{19, 0.5},
		{20, 0.25}, // second drop
	}
	for _, c := range cases {
		got := s.LR(c.step)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("StepDecayLR.LR(%d) = %v, want %v", c.step, got, c.want)
		}
	}
}
