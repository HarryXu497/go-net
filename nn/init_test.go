package nn

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// empiricalVariance returns the sample variance of vs, using the
// sample mean as the centering point.
func empiricalVariance(vs []float64) float64 {
	var sum float64
	for _, v := range vs {
		sum += v
	}

	mean := sum / float64(len(vs))

	var sq float64

	for _, v := range vs {
		d := v - mean
		sq += d * d
	}

	return sq / float64(len(vs))
}

// TestHeShape pins the contract that He returns a tensor of the
// requested shape, even when shape is rank-2 with non-square dims.
func TestHeShape(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	a := He()(rng, 3, 4)
	if !slices.Equal(a.Shape(), []int{3, 4}) {
		t.Errorf("shape = %v, want [3 4]", a.Shape())
	}
}

// TestHeVariance checks that He's empirical variance lands near
// 2/fan_in. With ~20000 samples the sample variance is well-
// concentrated, so a 10% tolerance is loose enough to be robust to
// RNG changes but tight enough to catch bugs.
func TestHeVariance(t *testing.T) {
	const (
		fanIn  = 400
		fanOut = 50
	)

	rng := rand.New(rand.NewSource(42))
	a := He()(rng, fanIn, fanOut)

	got := empiricalVariance(slices.Collect(a.All()))

	want := 2.0 / float64(fanIn)
	if math.Abs(got-want)/want > 0.10 {
		t.Errorf("empirical variance = %g, want close to %g (|rel diff| = %g)",
			got, want, math.Abs(got-want)/want)
	}
}

// TestGlorotVariance: Glorot's empirical variance should approach
// 2/(fan_in + fan_out). Same tolerance reasoning as He.
func TestGlorotVariance(t *testing.T) {
	const (
		fanIn  = 300
		fanOut = 100
	)

	rng := rand.New(rand.NewSource(42))
	a := Glorot()(rng, fanIn, fanOut)

	got := empiricalVariance(slices.Collect(a.All()))

	want := 2.0 / float64(fanIn+fanOut)
	if math.Abs(got-want)/want > 0.10 {
		t.Errorf("empirical variance = %g, want close to %g (|rel diff| = %g)",
			got, want, math.Abs(got-want)/want)
	}
}

// TestZerosAllZero: every element is exactly 0, and the shape is
// preserved. Also asserts that rng is allowed to be nil, since Zeros
// must not need a random source.
func TestZerosAllZero(t *testing.T) {
	a := Zeros()(nil, 4, 3)
	if !slices.Equal(a.Shape(), []int{4, 3}) {
		t.Errorf("shape = %v, want [4 3]", a.Shape())
	}

	for i, v := range slices.Collect(a.All()) {
		if v != 0 {
			t.Errorf("a[%d] = %g, want 0", i, v)
		}
	}
}

// TestInitializersDeterministic locks in that the same seed produces
// the same weights for both He and Glorot. This is the property
// callers rely on to reproduce training runs from a single seed.
func TestInitializersDeterministic(t *testing.T) {
	t.Run("He", func(t *testing.T) {
		a := He()(rand.New(rand.NewSource(7)), 4, 3)

		b := He()(rand.New(rand.NewSource(7)), 4, 3)
		if !slices.Equal(slices.Collect(a.All()), slices.Collect(b.All())) {
			t.Errorf("same seed produced different output")
		}
	})

	t.Run("Glorot", func(t *testing.T) {
		a := Glorot()(rand.New(rand.NewSource(7)), 4, 3)

		b := Glorot()(rand.New(rand.NewSource(7)), 4, 3)
		if !slices.Equal(slices.Collect(a.All()), slices.Collect(b.All())) {
			t.Errorf("same seed produced different output")
		}
	})
}
