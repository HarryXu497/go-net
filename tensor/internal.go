package tensor

import (
	"slices"

	"harryxu.ca/goml/ndarray"
)

// unbroadcast collapses g back to targetShape by summing over the axes
// that were stretched during forward broadcasting. It is the dual of
// numpy-style broadcasting.
//
// The collapse runs in two steps:
//  1. Sum away leading axes when a has higher rank than targetShape
//     (broadcasting prepends size-1 axes; this strips them).
//  2. Sum, with keepDims=true, over axes where targetShape is 1 but a
//     is N (broadcasting stretched a size-1 axis; this folds it back).
//
// When the shapes already match, a is returned unchanged; callers
// should not assume the result is a fresh allocation.
func unbroadcast(a *ndarray.NDArray, targetShape []int) *ndarray.NDArray {
	if slices.Equal(a.Shape(), targetShape) {
		return a
	}
	
	// Sum along leading axes to "collapse" them
	extra := a.Ndim() - len(targetShape)
	if extra > 0 {
		axes := make([]int, extra)
		for i := range axes {
			axes[i] = i
		}
		a = a.Sum(axes, false)
	}

	// Sum along size-1 axes to "collapse" them
	axes := make([]int, 0, a.Ndim())
	gShape := a.Shape()
	for i, dim := range targetShape {
		if dim == 1 && gShape[i] != 1 {
			axes = append(axes, i)
		}
	}

	if len(axes) > 0 {
		a = a.Sum(axes, true)
	}

	return a
} 