package ndarray

import "math/rand"

// FillNormal overwrites every element of a with a fresh sample from
// the normal distribution N(mean, stddev^2), drawn from rng. Intended
// for parameter initialization at network construction time.
//
// The implementation iterates the backing slice directly, so it is
// only correct for contiguous tensors e.g. fresh NewNDArray results.
// Calling it on a view (transposed, broadcast, or sliced) writes to
// underlying buffer positions that do not correspond one-to-one with
// the view's logical elements.
//
// Pass a seeded *rand.Rand for reproducible inits across runs.
func (a *NDArray) FillNormal(rng *rand.Rand, mean, stddev float64) {
	for i := range a.data {
		a.data[i] = (rng.NormFloat64() * stddev) + mean
	}
}
