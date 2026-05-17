package data

// Dataset is an indexed collection of samples of type T. Implementations
// expose their length and access by index.
type Dataset[T any] interface {
	Len() int
	Get(i int) T
}

// Sample is the shape of a supervised-learning data point: a
// numeric input vector X and an integer class label Y.
// Regression or multi-target tasks should define a sibling
// sample type rather than overload this one.
type Sample struct {
	X []float64
	Y int
}

// SliceDataset is a Dataset[T] backed by an in-memory slice. It is the
// simplest possible Dataset implementation: useful for tests and small
// debug subsets of larger datasets. Larger datasets that don't fit in
// memory should implement Dataset directly rather than wrap a slice.
type SliceDataset[T any] struct {
	items []T
}

// NewSliceDataset wraps an existing slice as a SliceDataset[T]. The slice
// is retained, not copied: mutations to items after construction are
// visible through Get, and the caller should not mutate items while the
// dataset is in use by a DataLoader.
func NewSliceDataset[T any](items []T) *SliceDataset[T] {
	return &SliceDataset[T]{
		items: items,
	}
}

// Len reports the number of samples in the dataset.
func (s *SliceDataset[T]) Len() int {
	return len(s.items)
}

// Get returns the sample at index i. It panics if i is out of range,
// matching the convention of the underlying slice; the DataLoader only
// ever calls Get with indices in [0, Len()).
func (s *SliceDataset[T]) Get(i int) T {
	return s.items[i]
}
