package ndarray

import (
	"iter"
	"slices"
)

// indices returns an iterator that iterates over the NDArray
// in shape order, yielding the multi-index and the corresponding offset
// of each value.
func (a *NDArray) indicesAndOffsets() iter.Seq2[[]int, int] {
	return func(yield func([]int, int) bool) {
		if a.Size() == 0 {
			return
		}

		indices := make([]int, len(a.shape))

		offset := a.offset
		for {
			if !yield(indices, offset) {
				return
			}

			k := len(indices) - 1
			for k >= 0 {
				indices[k]++

				offset += a.strides[k]
				if indices[k] < a.shape[k] {
					break
				}

				offset -= a.shape[k] * a.strides[k]
				indices[k] = 0
				k--
			}

			if k < 0 {
				return
			}
		}
	}
}

// All returns an iterator over the values in the NDArray
// in shape order.
func (a *NDArray) All() iter.Seq[float64] {
	return func(yield func(float64) bool) {
		for _, offset := range a.indicesAndOffsets() {
			if !yield(a.data[offset]) {
				return
			}
		}
	}
}

// AllIndexed returns an iterator over the values in the NDArray
// in shape order.
func (a *NDArray) AllIndexed() iter.Seq2[[]int, float64] {
	return func(yield func([]int, float64) bool) {
		for index, offset := range a.indicesAndOffsets() {
			if !yield(slices.Clone(index), a.data[offset]) {
				return
			}
		}
	}
}
