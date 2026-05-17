package ndarray

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// TestFillNormalEmpiricalMoments draws a large sample and checks that
// the empirical mean and variance land near the requested parameters.
// Tolerances are wide enough that a future RNG implementation change
// in math/rand won't break the test, but tight enough to catch the
// wrong scale or wrong offset.
func TestFillNormalEmpiricalMoments(t *testing.T) {
	const (
		n      = 10000
		mean   = 3.0
		stddev = 2.0
	)

	a := NewNDArray(n)
	a.FillNormal(rand.New(rand.NewSource(42)), mean, stddev)

	values := slices.Collect(a.All())

	var sum float64
	for _, v := range values {
		sum += v
	}

	gotMean := sum / float64(n)
	if math.Abs(gotMean-mean) > 0.1 {
		t.Errorf("empirical mean = %g, want close to %g", gotMean, mean)
	}

	var sq float64

	for _, v := range values {
		d := v - gotMean
		sq += d * d
	}

	gotVar := sq / float64(n)

	wantVar := stddev * stddev
	if math.Abs(gotVar-wantVar) > 0.2 {
		t.Errorf("empirical variance = %g, want close to %g", gotVar, wantVar)
	}
}

// TestFillNormalDeterministic locks in that the same seed produces the
// same values. Tests in nn/ will depend on this to keep parameter
// inits reproducible.
func TestFillNormalDeterministic(t *testing.T) {
	a := NewNDArray(2, 3)
	b := NewNDArray(2, 3)

	a.FillNormal(rand.New(rand.NewSource(42)), 0, 1)
	b.FillNormal(rand.New(rand.NewSource(42)), 0, 1)

	if !slices.Equal(slices.Collect(a.All()), slices.Collect(b.All())) {
		t.Errorf("same seed produced different output:\na=%v\nb=%v",
			slices.Collect(a.All()), slices.Collect(b.All()))
	}
}

// TestFillNormalZeroStddevIsConstant: with stddev=0 every sample
// collapses to exactly mean. Cheap edge case that catches a missing
// "* stddev" multiplication.
func TestFillNormalZeroStddevIsConstant(t *testing.T) {
	a := NewNDArray(5)
	a.FillNormal(rand.New(rand.NewSource(1)), 7.0, 0.0)

	for i, v := range slices.Collect(a.All()) {
		if v != 7.0 {
			t.Errorf("a[%d] = %g, want 7.0", i, v)
		}
	}
}
