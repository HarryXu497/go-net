package data

import (
	"fmt"
	"iter"
	"math/rand"
)

// DataLoader iterates batches of samples drawn from a Dataset. Each call
// to Batches starts a new epoch; if shuffle is true, indices are
// reshuffled at the start of every epoch using the provided RNG.
type DataLoader[T any] struct {
	dataset   Dataset[T]
	rng       *rand.Rand
	batchSize int
	shuffle   bool
}

// NewDataLoader constructs a DataLoader over dataset. batchSize must be
// > 0; if shuffle is true, rng must be non-nil. Both preconditions panic
// at construction.
func NewDataLoader[T any](dataset Dataset[T], batchSize int, shuffle bool, rng *rand.Rand) *DataLoader[T] {
	if batchSize <= 0 {
		panic(fmt.Sprintf("data: batch size must be a positive integer, got %d", batchSize))
	}

	if shuffle && rng == nil {
		panic("data: rng must be non-nil when shuffle is true")
	}

	return &DataLoader[T]{
		dataset:   dataset,
		batchSize: batchSize,
		shuffle:   shuffle,
		rng:       rng,
	}
}

// indexOrder returns the iteration order for one epoch: a fresh
// permutation when shuffling, or the natural order 0..Len()-1 otherwise.
func (d *DataLoader[T]) indexOrder() []int {
	if d.shuffle {
		return d.rng.Perm(d.dataset.Len())
	}

	indices := make([]int, d.dataset.Len())
	for i := range indices {
		indices[i] = i
	}

	return indices
}

// Batches yields one epoch of batches as amn iterator.
// The last batch may be shorter than batchSize if the dataset
// doesn't divide evenly. Each yielded slice is freshly allocated.
// Calling Batches again starts a new epoch with a fresh shuffle (if
// enabled).
func (d *DataLoader[T]) Batches() iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		indices := d.indexOrder()

		for start := 0; start < len(indices); start += d.batchSize {
			end := min(start+d.batchSize, len(indices))

			batch := make([]T, 0, end-start)
			for _, i := range indices[start:end] {
				batch = append(batch, d.dataset.Get(i))
			}

			if !yield(batch) {
				return
			}
		}
	}
}
