package data

import (
	"fmt"

	"harryxu.ca/gonet/ndarray"
)

// Collate stacks a batch of Samples into a contiguous (batchSize, dim)
// ndarray of inputs and a parallel []int of labels. Every sample's X
// must have the same length as the first sample's; mismatches and empty
// batches panic.
//
// The returned ndarray owns a freshly-allocated buffer copied from the
// samples, so callers may mutate the input slices afterward without
// affecting the output.
func Collate(batch []Sample) (*ndarray.NDArray, []int) {
	if len(batch) == 0 {
		panic("data: cannot collate an empty batch")
	}

	dim := len(batch[0].X)
	xData := make([]float64, len(batch)*dim)

	y := make([]int, len(batch))
	for i, sample := range batch {
		if len(sample.X) != dim {
			panic(fmt.Sprintf("data: collate requires sample %d have len(X)=%d, got %d", i, dim, len(sample.X)))
		}

		copy(xData[i*dim:(i+1)*dim], sample.X)
		y[i] = sample.Y
	}

	return ndarray.FromSlice(xData, len(batch), dim), y
}
